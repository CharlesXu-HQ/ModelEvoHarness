"""Paper-derived recommendation heads over supplied representations, without encoders."""

import torch
from torch import nn
from torch.nn import functional as F


def _attention(logits, mask):
    active = mask.bool()
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    masked = work.masked_fill(~active, float("-inf"))
    masked = torch.where(active.any(-1, keepdim=True), masked, torch.zeros_like(masked))
    return torch.where(active, torch.softmax(masked, -1), 0).to(logits.dtype)


class DICM(nn.Module):
    """Image Matters ranking head: two query channels over behavior images.

    forward(history[B,T,D], image_query[B,D], id_query[B,Q], mask[B,T]) -> [B,2D].
    The two scoring MLPs are a declared compact choice; the paper does not fully
    specify their layers. This candidate-conditioned output is not indexable.
    The host owns image/ID encoders, CTR head and AMS infrastructure.
    """

    def __init__(self, image_dim: int, id_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.image_attention = nn.Sequential(nn.Linear(2 * image_dim, hidden_dim),
                                             nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.id_attention = nn.Sequential(nn.Linear(image_dim + id_dim, hidden_dim),
                                          nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, history, image_query, id_query, history_mask):
        outputs = []
        for query, scorer in ((image_query, self.image_attention), (id_query, self.id_attention)):
            repeated = query[:, None, :].expand(-1, history.shape[1], -1)
            logits = scorer(torch.cat((history, repeated), -1)).squeeze(-1)
            outputs.append((_attention(logits, history_mask)[..., None] * history).sum(1))
        return torch.cat(outputs, -1)


class MMRec(nn.Module):
    """MM-Rec Eqs. 3–4: four independently normalized cross-modal attention streams.

    Text/image histories [B,T,D], candidate text/image [B,D], mask [B,T]
    -> score [B]. All embeddings share D. Input encoders, ROI detection and
    ViLBERT are excluded; caller-provided embeddings may be trained upstream.
    """

    def representation(self, history_text, history_image, candidate_text, candidate_image, history_mask):
        user = torch.zeros_like(candidate_text)
        for history in (history_text, history_image):
            for candidate in (candidate_text, candidate_image):
                logits = (history * candidate[:, None, :]).sum(-1)
                user = user + (_attention(logits, history_mask)[..., None] * history).sum(1)
        return user

    def forward(self, history_text, history_image, candidate_text, candidate_image, history_mask):
        user = self.representation(history_text, history_image, candidate_text, candidate_image, history_mask)
        return (user * (candidate_text + candidate_image)).sum(-1)


class DMTLDistillation(nn.Module):
    """Bernoulli KL from product teacher to student; [B] logits -> scalar mean.

    Teacher click and conditional long-read probabilities form a joint product.
    Distillation detaches the teacher. Host supplies MMoE teacher, two-tower
    student, impression labels and training schedule; no duration binning here.
    """

    def forward(self, click_logits, conditional_read_logits, student_logits):
        teacher = (torch.sigmoid(click_logits) * torch.sigmoid(conditional_read_logits)).detach()
        entropy = -torch.xlogy(teacher, teacher) - torch.xlogy(1 - teacher, 1 - teacher)
        cross_entropy = F.binary_cross_entropy_with_logits(student_logits, teacher, reduction="none")
        return (cross_entropy - entropy).mean()

    def teacher_loss(self, click_logits, conditional_read_logits, click_labels, joint_read_labels):
        """Joint-read labels are impression-level; do not train CVR on unclicked negatives.

        Eq. 8's apparent label typo is resolved using click labels as stated in
        the paper's prose and Figure 1. Both terms use mean BCE.
        """
        a, b = click_logits, conditional_read_logits
        joint_logits = a + b - torch.logsumexp(torch.stack((torch.zeros_like(a), a, b)), 0)
        return (F.binary_cross_entropy_with_logits(a, click_labels)
                + F.binary_cross_entropy_with_logits(joint_logits, joint_read_labels))
