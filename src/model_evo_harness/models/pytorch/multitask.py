"""Independent PyTorch references for multi-task, scenario and slate models."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _expert(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())


class SharedBottom(nn.Module):
    """Dense features [B,D] -> one raw logit per observed task [B,T]."""

    def __init__(self, input_dim: int, num_tasks: int, hidden_dim: int):
        super().__init__()
        self.bottom = _expert(input_dim, hidden_dim)
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(num_tasks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.bottom(x)
        return torch.cat([head(shared) for head in self.heads], dim=1)


class ESMM(nn.Module):
    """Impression features [B,D] -> click, post-click and joint probabilities [B]."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.click_tower = nn.Sequential(_expert(input_dim, hidden_dim), nn.Linear(hidden_dim, 1))
        self.conversion_tower = nn.Sequential(_expert(input_dim, hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        click = torch.sigmoid(self.click_tower(x).squeeze(-1))
        post_click = torch.sigmoid(self.conversion_tower(x).squeeze(-1))
        return {"click": click, "post_click": post_click, "joint": click * post_click}


class PLE(nn.Module):
    """CGC extraction: shared and task experts, then raw task logits [B,T]."""

    def __init__(self, input_dim: int, num_tasks: int, shared_experts: int,
                 task_experts: int, expert_dim: int, *, levels: int = 2):
        super().__init__()
        if min(num_tasks, shared_experts, task_experts, levels) < 1:
            raise ValueError("PLE needs positive tasks, experts and levels")
        self.num_tasks = num_tasks
        self.shared_experts = nn.ModuleList()
        self.task_experts = nn.ModuleList()
        self.task_gates = nn.ModuleList()
        self.shared_gates = nn.ModuleList()
        for level in range(levels):
            width = input_dim if level == 0 else expert_dim
            self.shared_experts.append(nn.ModuleList(
                _expert(width, expert_dim) for _ in range(shared_experts)))
            self.task_experts.append(nn.ModuleList(
                nn.ModuleList(_expert(width, expert_dim) for _ in range(task_experts))
                for _ in range(num_tasks)))
            self.task_gates.append(nn.ModuleList(
                nn.Linear(width, task_experts + shared_experts) for _ in range(num_tasks)))
            if level < levels - 1:
                self.shared_gates.append(nn.Linear(width, shared_experts + num_tasks * task_experts))
        self.heads = nn.ModuleList(nn.Linear(expert_dim, 1) for _ in range(num_tasks))

    @staticmethod
    def _mix(values: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return (values * torch.softmax(gate, dim=1).unsqueeze(-1)).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = x
        tasks = [x] * self.num_tasks
        for level, (shared_group, task_groups, task_gates) in enumerate(
                zip(self.shared_experts, self.task_experts, self.task_gates)):
            shared_values = torch.stack([expert(shared) for expert in shared_group], dim=1)
            task_values = [[expert(tasks[t]) for expert in task_groups[t]]
                           for t in range(self.num_tasks)]
            next_tasks = []
            for t in range(self.num_tasks):
                values = torch.cat([torch.stack(task_values[t], dim=1), shared_values], dim=1)
                next_tasks.append(self._mix(values, task_gates[t](tasks[t])))
            if level < len(self.shared_gates):
                all_values = torch.cat([shared_values, *[torch.stack(v, dim=1)
                                                         for v in task_values]], dim=1)
                shared = self._mix(all_values, self.shared_gates[level](shared))
            tasks = next_tasks
        return torch.cat([head(value) for head, value in zip(self.heads, tasks)], dim=1)


class AITM(nn.Module):
    """Attention transfer across ordered tasks; features [B,D] -> logits [B,T]."""

    def __init__(self, input_dim: int, num_tasks: int, hidden_dim: int):
        super().__init__()
        if num_tasks < 2:
            raise ValueError("AITM needs at least two ordered tasks")
        self.task_encoders = nn.ModuleList(_expert(input_dim, hidden_dim)
                                           for _ in range(num_tasks))
        self.transfer = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim)
                                      for _ in range(num_tasks - 1))
        self.queries = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim)
                                     for _ in range(num_tasks - 1))
        self.keys = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim)
                                  for _ in range(num_tasks - 1))
        self.values = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim)
                                    for _ in range(num_tasks - 1))
        self.hidden_dim = hidden_dim
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(num_tasks))

    def _attend(self, previous: torch.Tensor, current: torch.Tensor,
                step: int) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = torch.stack([self.transfer[step](previous), current], dim=1)
        score = (self.queries[step](candidates) * self.keys[step](candidates)).sum(
            dim=-1) / math.sqrt(self.hidden_dim)
        weights = torch.softmax(score, dim=1)
        return (self.values[step](candidates) * weights.unsqueeze(-1)).sum(1), weights

    def attention_weights(self, previous: torch.Tensor, current: torch.Tensor,
                          step: int) -> torch.Tensor:
        """Return the two candidate weights for an adjacent task transition."""
        return self._attend(previous, current, step)[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous = None
        outputs = []
        for t, (encoder, head) in enumerate(zip(self.task_encoders, self.heads)):
            current = encoder(x)
            if previous is not None:
                current, _ = self._attend(previous, current, t - 1)
            outputs.append(head(current))
            previous = current
        return torch.cat(outputs, dim=1)


def aitm_loss(logits: torch.Tensor, labels: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    """Binary loss plus ordered funnel penalty for fully observed end-to-end labels."""
    probabilities = torch.sigmoid(logits)
    order_penalty = F.relu(probabilities[:, 1:] - probabilities[:, :-1]).mean()
    return F.binary_cross_entropy_with_logits(logits, labels) + alpha * order_penalty


class M2M(nn.Module):
    """Task/expert/scenario views; (features [B,D], scenario [B,S]) -> [B,T] logits."""

    def __init__(self, input_dim: int, scenario_dim: int, num_tasks: int,
                 num_experts: int, expert_dim: int, hidden_dim: int):
        super().__init__()
        self.num_tasks, self.num_experts = num_tasks, num_experts
        self.experts = nn.ModuleList(_expert(input_dim, expert_dim)
                                     for _ in range(num_experts))
        self.meta_gates = nn.Linear(scenario_dim, num_tasks * num_experts)
        self.meta_towers = nn.ModuleList(nn.Sequential(
            nn.Linear(expert_dim + scenario_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)) for _ in range(num_tasks))

    def forward(self, x: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        values = torch.stack([expert(x) for expert in self.experts], dim=1)
        gates = torch.softmax(self.meta_gates(scenario).reshape(
            -1, self.num_tasks, self.num_experts), dim=-1)
        outputs = []
        for t, tower in enumerate(self.meta_towers):
            mixture = (values * gates[:, t, :, None]).sum(dim=1)
            outputs.append(tower(torch.cat([mixture, scenario], dim=-1)))
        return torch.cat(outputs, dim=1)


class APG(nn.Module):
    """Scenario-generated low-rank center matrix; (features, scenario) -> [B] logits."""

    def __init__(self, input_dim: int, scenario_dim: int, rank: int, hidden_dim: int):
        super().__init__()
        self.rank = rank
        self.input_projection = nn.Linear(input_dim, rank)
        self.generator = nn.Linear(scenario_dim, rank * rank)
        self.output_projection = nn.Linear(rank, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def generated_matrix(self, scenario: torch.Tensor) -> torch.Tensor:
        return self.generator(scenario).reshape(-1, self.rank, self.rank)

    def forward(self, x: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        center = torch.bmm(self.generated_matrix(scenario),
                           self.input_projection(x).unsqueeze(-1)).squeeze(-1)
        return self.head(F.relu(self.output_projection(center))).squeeze(-1)


class HMoE(nn.Module):
    """Domain-routed experts; cross-domain tower scores have stopped gradients."""

    def __init__(self, input_dim: int, num_domains: int, num_experts: int, expert_dim: int):
        super().__init__()
        self.num_domains = num_domains
        self.experts = nn.ModuleList(_expert(input_dim, expert_dim)
                                     for _ in range(num_experts))
        self.domain_gates = nn.ModuleList(nn.Linear(input_dim, num_experts)
                                          for _ in range(num_domains))
        self.domain_towers = nn.ModuleList(nn.Linear(expert_dim, 1)
                                           for _ in range(num_domains))
        self.domain_weights = nn.Linear(input_dim, num_domains)
        self.domain_bias = nn.Embedding(num_domains, num_domains)

    def forward(self, x: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        experts = torch.stack([expert(x) for expert in self.experts], dim=1)
        scores = []
        for gate, tower in zip(self.domain_gates, self.domain_towers):
            mixture = (experts * torch.softmax(gate(x), dim=1).unsqueeze(-1)).sum(dim=1)
            scores.append(tower(mixture))
        scores = torch.cat(scores, dim=1)
        own = F.one_hot(domain.long(), self.num_domains).bool()
        routed = torch.where(own, scores, scores.detach())
        weights = torch.softmax(self.domain_weights(x) + self.domain_bias(domain.long()), dim=1)
        return (weights * routed).sum(dim=1)


class PEPNet(nn.Module):
    """EPNet input gate and per-task PPNet hidden gates; outputs [B,T] logits."""

    def __init__(self, input_dim: int, context_dim: int, num_tasks: int, hidden_dim: int):
        super().__init__()
        self.embedding_gate = nn.Linear(context_dim, input_dim)
        self.bottom = _expert(input_dim, hidden_dim)
        self.task_gates = nn.ModuleList(nn.Linear(context_dim, hidden_dim)
                                        for _ in range(num_tasks))
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(num_tasks))

    def forward(self, x: torch.Tensor, ep_context: torch.Tensor,
                pp_context: torch.Tensor) -> torch.Tensor:
        personalized = x * (2 * torch.sigmoid(self.embedding_gate(ep_context)))
        hidden = self.bottom(personalized)
        return torch.cat([head(hidden * (2 * torch.sigmoid(gate(pp_context))))
                          for gate, head in zip(self.task_gates, self.heads)], dim=1)


class _PartitionedNorm(nn.Module):
    """Normalize each domain over its batch in training and running moments in eval."""

    def __init__(self, input_dim: int, num_domains: int, *, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.eps = 1e-5
        self.global_scale = nn.Parameter(torch.ones(input_dim))
        self.global_shift = nn.Parameter(torch.zeros(input_dim))
        self.domain_scale = nn.Embedding(num_domains, input_dim)
        self.domain_shift = nn.Embedding(num_domains, input_dim)
        nn.init.ones_(self.domain_scale.weight)
        nn.init.zeros_(self.domain_shift.weight)
        self.register_buffer("running_mean", torch.zeros(num_domains, input_dim))
        self.register_buffer("running_var", torch.ones(num_domains, input_dim))
        self.register_buffer("running_batches", torch.zeros(num_domains, dtype=torch.long))

    def forward(self, x: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        domain = domain.long()
        normalized = torch.empty_like(x)
        for d in range(self.running_mean.shape[0]):
            selected = domain == d
            count = int(selected.sum())
            if not count:
                continue
            group = x[selected]
            if self.training and count >= 2:
                mean = group.mean(dim=0)
                var = group.var(dim=0, unbiased=False)
                with torch.no_grad():
                    if self.running_batches[d] == 0:
                        self.running_mean[d].copy_(mean)
                        self.running_var[d].copy_(var)
                    else:
                        self.running_mean[d].mul_(self.momentum).add_(
                            mean * (1 - self.momentum))
                        self.running_var[d].mul_(self.momentum).add_(
                            var * (1 - self.momentum))
                    self.running_batches[d] += 1
                values = (group - mean) / torch.sqrt(var + self.eps)
            elif self.running_batches[d] > 0:
                values = (group - self.running_mean[d]) / torch.sqrt(
                    self.running_var[d] + self.eps)
            else:
                values = group
            normalized[selected] = values
        return (normalized * self.global_scale * self.domain_scale(domain) +
                self.global_shift + self.domain_shift(domain))


class STAR(nn.Module):
    """Partitioned normalization and shared × domain star layer; [B,D], [B] -> [B]."""

    def __init__(self, input_dim: int, num_domains: int, hidden_dim: int):
        super().__init__()
        self.norm = _PartitionedNorm(input_dim, num_domains)
        self.shared_weight = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.domain_weight = nn.Parameter(torch.ones(num_domains, input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.shared_weight)
        self.star_head = nn.Linear(hidden_dim, 1)
        self.aux_head = nn.Linear(input_dim, 1)
        self.aux_domain = nn.Embedding(num_domains, 1)

    def forward(self, x: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        domain = domain.long()
        normalized = self.norm(x, domain)
        weights = self.shared_weight.unsqueeze(0) * self.domain_weight[domain]
        hidden = F.relu(torch.bmm(normalized.unsqueeze(1), weights).squeeze(1))
        return (self.star_head(hidden) + self.aux_head(normalized) +
                self.aux_domain(domain)).squeeze(-1)


class PRM(nn.Module):
    """Slate candidates [B,S,D], user [B,U], mask [B,S] -> position softmax [B,S]."""

    def __init__(self, input_dim: int, user_dim: int, hidden_dim: int,
                 max_positions: int, num_heads: int):
        super().__init__()
        self.item = nn.Linear(input_dim, hidden_dim)
        self.user = nn.Linear(user_dim, hidden_dim)
        self.position = nn.Embedding(max_positions, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, candidates: torch.Tensor, user: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any(dim=1).all()):
            raise ValueError("PRM needs at least one eligible candidate per slate")
        sequence = self.item(candidates) + self.user(user)[:, None, :] + \
            self.position(torch.arange(candidates.shape[1], device=candidates.device))[None, :, :]
        attended, _ = self.attention(sequence, sequence, sequence, key_padding_mask=~mask.bool())
        logits = self.head(self.norm(sequence + attended)).squeeze(-1)
        return torch.softmax(logits.masked_fill(~mask.bool(), -torch.inf), dim=1)


class PRS(nn.Module):
    """Slate candidates [B,S,D] -> per-position CTR/continuation probabilities [B,S]."""

    def __init__(self, input_dim: int, hidden_dim: int, max_positions: int):
        super().__init__()
        self.item = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Embedding(max_positions, hidden_dim)
        self.ctr_head = nn.Linear(hidden_dim, 1)
        self.continuation_head = nn.Linear(hidden_dim, 1)

    def forward(self, candidates: torch.Tensor) -> dict[str, torch.Tensor]:
        position = self.position(torch.arange(candidates.shape[1], device=candidates.device))
        hidden = F.relu(self.item(candidates) + position[None, :, :])
        return {"ctr": torch.sigmoid(self.ctr_head(hidden).squeeze(-1)),
                "continuation": torch.sigmoid(self.continuation_head(hidden).squeeze(-1))}

    @staticmethod
    def slate_reward(ctr: torch.Tensor, continuation: torch.Tensor) -> torch.Tensor:
        reach = torch.cumprod(torch.cat([torch.ones_like(continuation[:, :1]),
                                         continuation[:, :-1]], dim=1), dim=1)
        return (ctr * reach).sum(dim=1)

    @torch.no_grad()
    def rerank(self, candidates: torch.Tensor, *, beam_size: int = 3) -> torch.Tensor:
        """Beam-propose permutations, then rescore complete slates; inference only."""
        if beam_size < 1:
            raise ValueError("beam_size must be positive")
        orders = []
        for slate in candidates:
            beams = [((), 0.0)]
            for _ in range(len(slate)):
                expanded = []
                for prefix, _ in beams:
                    for index in range(len(slate)):
                        if index in prefix:
                            continue
                        order = (*prefix, index)
                        output = self.forward(slate[list(order)][None, :, :])
                        value = self.slate_reward(output["ctr"], output["continuation"]).item()
                        expanded.append((order, value))
                beams = sorted(expanded, key=lambda pair: (-pair[1], pair[0]))[:beam_size]
            rescored = []
            for order, _ in beams:
                output = self.forward(slate[list(order)][None, :, :])
                value = self.slate_reward(output["ctr"], output["continuation"]).item()
                rescored.append((order, value))
            orders.append(max(rescored, key=lambda pair: pair[1])[0])
        return torch.tensor(orders, dtype=torch.long, device=candidates.device)
