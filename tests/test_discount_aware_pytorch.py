"""Independent mathematical and masking checks for DANet-inspired components."""

import importlib
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _classes():
    path = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models/pytorch/discount_aware.py"
    assert path.is_file(), f"Missing standalone discount component: {path}"
    module = importlib.import_module("model_evo_harness.models.pytorch.discount_aware")
    return module.DiscountPreferenceBranch, module.DiscountAwareConversionHead


def _args(length=5, purchases=2, batch=1):
    return [torch.full((batch, length), .5), torch.ones(batch, length, dtype=torch.bool),
            torch.ones(batch, 1), torch.zeros(batch, purchases, 1),
            torch.ones(batch, purchases, 1), torch.ones(batch, purchases, dtype=torch.bool),
            torch.zeros(batch, 1)]


def _linear(layer, weight, bias):
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight, dtype=layer.weight.dtype))
        layer.bias.copy_(torch.tensor(bias, dtype=layer.bias.dtype))


def _zero(model):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()


@pytest.mark.parametrize("length", [3, 5, 8])
def test_spectral_filter_preserves_explicit_length_and_has_known_dc_output(length):
    branch = _classes()[0](length, 1, 1, 1, 2)
    _zero(branch)
    # Input .5 has only a DC coefficient N/2; retaining 40% gives .2 everywhere.
    with torch.no_grad():
        branch.spectral_filter[0].weight[0, 0] = 1
        branch.spectral_filter[2].weight[0, 0] = .4
    output = branch(*_args(length))
    torch.testing.assert_close(output["filtered_history"], torch.full((1, length), .2))
    torch.testing.assert_close(output["residual_history"], torch.full((1, length), .3))
    expected = torch.cat((torch.zeros(1, 2), torch.full((1, length), .3), torch.full((1, length), .2)), 1)
    torch.testing.assert_close(output["representation"], expected)


def test_spectral_filter_receives_real_then_imaginary_coefficients():
    branch = _classes()[0](4, 1, 1, 1, 2)
    _zero(branch)
    # rFFT([0,1,0,0]) = [1,0,-1] + i[0,-1,0]. Route only imaginary bin 1.
    with torch.no_grad():
        branch.spectral_filter[0].weight[0, 4] = -1
        branch.spectral_filter[2].weight[4, 0] = -1
    args = _args(4)
    args[0] = torch.tensor([[0., 1., 0., 0.]])
    output = branch(*args)
    torch.testing.assert_close(output["filtered_history"], torch.tensor([[0., .5, 0., -.5]]))
    torch.testing.assert_close(output["residual_history"], torch.tensor([[0., .5, 0., .5]]))


def test_target_attention_context_gate_and_representation_order_have_known_values():
    branch = _classes()[0](3, 1, 1, 1, 2)
    _zero(branch)
    _linear(branch.attribute_attention[0], [[1., -1.], [0., 0.]], [0., 0.])
    _linear(branch.attribute_attention[2], [[1., 0.]], [0.])
    _linear(branch.user_projection[0], [[1.], [2.]], [0., 0.])
    _linear(branch.context_gate[0], [[1.], [-1.]], [0., 0.])
    args = _args(3)
    args[3] = torch.tensor([[[0.], [2.]]])
    args[4] = torch.tensor([[[2.], [4.]]])
    result = branch(*args)
    p = 1 / (1 + math.e)
    pooled = 2 * p + 4 * (1 - p)
    torch.testing.assert_close(result["attention"], torch.tensor([[p, 1 - p]]))
    torch.testing.assert_close(result["user_preference"], torch.tensor([[pooled / 2, pooled]]))
    args[2] = torch.tensor([[3.]])
    torch.testing.assert_close(branch(*args)["attention"], torch.tensor([[.5, .5]]))
    args[2] = torch.tensor([[1.]])
    args[6] = torch.tensor([[math.log(3)]])
    torch.testing.assert_close(branch(*args)["user_preference"], torch.tensor([[.75 * pooled, .5 * pooled]]))


@pytest.mark.parametrize("purchases", [0, 3])
def test_empty_purchase_and_history_signals_stay_zero_even_with_learned_biases(purchases):
    branch = _classes()[0](5, 1, 1, 1, 2)
    with torch.no_grad():
        for name, parameter in branch.named_parameters():
            if name.endswith("bias"):
                parameter.fill_(2)
    args = _args(5, purchases, batch=2)
    args[1].fill_(False)
    args[5].fill_(False)
    for i in (0, 3, 4):
        args[i].fill_(float("nan"))
    output = branch(*args)
    assert output["attention"].shape == (2, purchases)
    for value in output.values():
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, torch.zeros_like(value))
    output["representation"].square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())


def test_padding_is_removed_before_fft_attention_and_gradient_computation():
    torch.manual_seed(17)
    branch = _classes()[0](5, 1, 1, 1, 3)
    args = _args(5, 3, batch=2)
    args[0] = torch.tensor([[.1, float("nan"), .4, .3, .2], [float("nan")] * 5], requires_grad=True)
    args[1] = torch.tensor([[1, 0, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
    args[3] = torch.tensor([[[1.], [float("inf")], [2.]], [[float("nan")]] * 3], requires_grad=True)
    args[4] = torch.tensor([[[2.], [float("nan")], [3.]], [[float("nan")]] * 3], requires_grad=True)
    args[5] = torch.tensor([[1, 0, 1], [0, 0, 0]], dtype=torch.bool)
    output = branch(*args)
    clean = [torch.nan_to_num(value) if value.is_floating_point() else value for value in args]
    cleaned = branch(*clean)
    for name in output:
        torch.testing.assert_close(output[name], cleaned[name])
    torch.testing.assert_close(output["attention"][0].sum(), torch.tensor(1.))
    torch.testing.assert_close(output["representation"][1], torch.zeros(13))
    output["representation"].square().sum().backward()
    for index, mask_index in ((0, 1), (3, 5), (4, 5)):
        gradient = args[index].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient[~args[mask_index]], torch.zeros_like(gradient[~args[mask_index]]))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())


@pytest.mark.parametrize("index,replacement", [
    (0, torch.zeros(1, 4)), (1, torch.ones(1, 5)), (2, torch.zeros(1, 2)),
    (3, torch.zeros(1, 2)), (4, torch.zeros(1, 3, 1)), (5, torch.ones(1, 2)),
    (6, torch.zeros(2, 1)), (0, torch.full((1, 5), 1.1)),
    (0, torch.full((1, 5), -.1)), (2, torch.full((1, 1), float("nan"))),
    (4, torch.full((1, 2, 1), float("nan"))), (6, torch.full((1, 1), float("inf"))),
])
def test_branch_rejects_incompatible_shapes_masks_and_observed_values(index, replacement):
    branch = _classes()[0](5, 1, 1, 1, 2)
    args = _args()
    args[index] = replacement
    with pytest.raises((ValueError, TypeError)):
        branch(*args)


def test_half_inputs_promote_to_default_float_and_unsupported_parameter_dtype_is_explicit():
    branch = _classes()[0](5, 1, 1, 1, 2)
    args = _args()
    for dtype in (torch.float16, torch.bfloat16):
        converted = [x.to(dtype) if x.is_floating_point() else x for x in args]
        output = branch(*converted)
        assert output["representation"].dtype == torch.float32
        torch.testing.assert_close(output["representation"], branch(*args)["representation"])
    with pytest.raises(ValueError, match="float32.*float64"):
        branch.half()(*args)


def test_autocast_keeps_odd_length_fft_in_full_precision_with_finite_gradients():
    branch = _classes()[0](5, 1, 1, 1, 2)
    args = _args(5)
    args[0].requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = branch(*args)
        loss = output["representation"].square().sum()
    assert output["filtered_history"].dtype == torch.float32
    assert output["filtered_history"].shape == (1, 5)
    assert torch.isfinite(output["representation"]).all()
    loss.backward()
    assert args[0].grad is not None and torch.isfinite(args[0].grad).all()
    assert all(p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all()
               for p in branch.parameters())


def test_independent_conversion_and_discount_heads_match_assigned_weights():
    head = _classes()[1](1, 2, 2)
    _linear(head.conversion_head[0], [[1., 2., 0.], [-1., 0., 1.]], [0., 0.])
    _linear(head.conversion_head[2], [[1., 2.]], [-.5])
    _linear(head.discount_head[0], [[1., 0.], [0., 1.]], [0., 0.])
    _linear(head.discount_head[2], [[1., -1.]], [0.])
    interest = torch.tensor([[1.], [2.]])
    discount = torch.tensor([[.2, .4], [.5, .3]])
    output = head(interest, discount)
    torch.testing.assert_close(output["conversion_logit"], torch.tensor([.9, 2.5]))
    torch.testing.assert_close(output["discount_prediction"], torch.sigmoid(torch.tensor([-.2, .2])))
    changed = head(interest + 1, discount)
    torch.testing.assert_close(changed["discount_prediction"], output["discount_prediction"])
    assert not torch.equal(changed["conversion_logit"], output["conversion_logit"])


def test_objective_uses_only_valid_purchases_and_normalizes_by_positive_count():
    head = _classes()[1](1, 2, 2)
    predictions = torch.tensor([.1, .2, .3, .4], requires_grad=True)
    outputs = {"conversion_logit": torch.tensor([0., math.log(3), -math.log(3), 0.], requires_grad=True),
               "discount_prediction": predictions}
    labels = torch.tensor([1., 0., 1., 0.], requires_grad=True)
    discount_labels = torch.tensor([.4, float("nan"), .1, float("nan")], requires_grad=True)
    result = head.objective(outputs, labels, discount_labels, torch.tensor([1, 1, 1, 0], dtype=torch.bool), alpha=2)
    torch.testing.assert_close(result["conversion_loss"], torch.tensor((math.log(2) + math.log(4)) / 2))
    torch.testing.assert_close(result["discount_loss"], torch.tensor(.065))
    torch.testing.assert_close(result["loss"], result["conversion_loss"] + .13)
    assert result["discount_count"].dtype == torch.int64 and result["discount_count"].item() == 2
    result["discount_loss"].backward()
    torch.testing.assert_close(predictions.grad, torch.tensor([-.3, 0., .2, 0.]))
    assert labels.grad is None and discount_labels.grad is None


@pytest.mark.parametrize("labels,mask", [([0, 0], [1, 1]), ([1, 1], [0, 0])])
def test_no_valid_positive_auxiliary_objective_is_differentiable_zero(labels, mask):
    head = _classes()[1](1, 2, 2)
    predictions = torch.tensor([.2, .8], requires_grad=True)
    outputs = {"conversion_logit": torch.zeros(2, requires_grad=True), "discount_prediction": predictions}
    result = head.objective(outputs, torch.tensor(labels), torch.full((2,), float("nan")),
                            torch.tensor(mask, dtype=torch.bool))
    assert result["discount_count"].item() == 0
    assert result["discount_loss"].item() == 0 and torch.isfinite(result["loss"])
    result["discount_loss"].backward()
    torch.testing.assert_close(predictions.grad, torch.zeros(2))


@pytest.mark.parametrize("prediction", [-.1, 1.1])
def test_objective_rejects_out_of_range_discount_predictions(prediction):
    head = _classes()[1](1, 2, 2)
    outputs = {"conversion_logit": torch.zeros(2),
               "discount_prediction": torch.tensor([.5, prediction])}
    # The prediction contract applies to every row, including non-purchases.
    with pytest.raises(ValueError, match="discount_prediction.*\\[0,1\\]"):
        head.objective(outputs, torch.tensor([1., 0.]), torch.tensor([.4, float("nan")]),
                       torch.tensor([True, False]))


@pytest.mark.parametrize("field,value", [
    ("labels", torch.tensor([1., .5])), ("labels", torch.tensor([1., float("nan")])),
    ("discounts", torch.tensor([1.1, .5])), ("discounts", torch.tensor([float("nan"), .5])),
    ("mask", torch.ones(2)), ("alpha", -1), ("alpha", float("inf")), ("alpha", float("nan")),
])
def test_objective_rejects_invalid_observed_labels_mask_and_alpha(field, value):
    head = _classes()[1](1, 2, 2)
    args = {"labels": torch.tensor([1., 0.]), "discounts": torch.tensor([.4, .5]),
            "mask": torch.ones(2, dtype=torch.bool), "alpha": 1.}
    args[field] = value
    outputs = {"conversion_logit": torch.zeros(2), "discount_prediction": torch.full((2,), .5)}
    with pytest.raises((ValueError, TypeError)):
        head.objective(outputs, args["labels"], args["discounts"], args["mask"], args["alpha"])


def test_heads_objective_and_constructors_validate_shapes_and_dimensions():
    branch_cls, head_cls = _classes()
    for dims in ((0, 1, 1, 1, 2), (5, 1, -1, 1, 2), (5, 1, 1, 1, 1.5)):
        with pytest.raises(ValueError):
            branch_cls(*dims)
    with pytest.raises(ValueError):
        head_cls(1, 0, 2)
    head = head_cls(1, 2, 2)
    for interest, discount in ((torch.zeros(2), torch.zeros(2, 2)),
                               (torch.zeros(2, 1), torch.zeros(1, 2)),
                               (torch.zeros(2, 1), torch.full((2, 2), float("nan")))):
        with pytest.raises(ValueError):
            head(interest, discount)
    for length in (0, 2):
        outputs = {"conversion_logit": torch.zeros(length), "discount_prediction": torch.zeros(length)}
        with pytest.raises(ValueError):
            head.objective(outputs, torch.zeros(1 if length else 0), torch.zeros(length),
                            torch.ones(length, dtype=torch.bool))


def test_composed_branch_and_head_have_finite_gradients():
    torch.manual_seed(44)
    branch_cls, head_cls = _classes()
    branch, head = branch_cls(5, 1, 1, 1, 2), head_cls(2, 12, 3)
    args = _args(5, batch=2)
    args[4] = torch.tensor([[[2.], [1.]], [[1.], [3.]]])
    interest = torch.tensor([[1., 2.], [2., 1.]], requires_grad=True)
    representation = branch(*args)["representation"]
    output = head(interest, representation)
    loss = head.objective(output, torch.tensor([1., 0.]), torch.tensor([.3, float("nan")]),
                          torch.tensor([True, False]))["loss"]
    loss.backward()
    assert interest.grad is not None and torch.isfinite(interest.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for model in (branch, head) for p in model.parameters())
