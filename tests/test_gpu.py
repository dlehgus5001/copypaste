"""Real CUDA checks; a CPU runner must report these as skipped, never passed."""
import pytest
import torch
import torchvision

from offline_yolox.bootstrap import activate_vendor
from yolox.exp import get_exp
from yolox.layers import FastCOCOEvalOp

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="No NVIDIA GPU available")]


def test_cuda_tensor_cudnn_and_nms():
    assert torch.version.cuda == "12.9"
    assert torch.backends.cudnn.version() is not None
    device = torch.device("cuda:0")
    x = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    assert torch.allclose((x @ x.T).cpu(), x.cpu() @ x.cpu().T)
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]],
                         dtype=torch.float32, device=device)
    scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    assert torchvision.ops.nms(boxes, scores, 0.5).cpu().tolist() == [0, 2]
    conv = torch.nn.Conv2d(3, 8, 3).to(device)
    out = conv(torch.randn(2, 3, 32, 32, device=device))
    out.square().mean().backward()
    assert conv.weight.grad.isfinite().all()


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_official_yolox_loss_backward_cuda(mixed_precision):
    torch.manual_seed(37)
    exp = get_exp(exp_name="yolox-nano")
    exp.num_classes = 1
    exp.warmup_epochs = 0
    model = exp.get_model().cuda().train()
    optimizer = exp.get_optimizer(batch_size=2)
    tracked_parameter = next(model.parameters())
    before = tracked_parameter.detach().clone()
    images = torch.rand(2, 3, 128, 128, device="cuda") * 255
    labels = torch.zeros(2, 50, 5, device="cuda")
    labels[:, 0] = torch.tensor([0, 64, 64, 24, 20], device="cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    with torch.amp.autocast("cuda", enabled=mixed_precision):
        outputs = model(images, labels)
    loss = outputs["total_loss"]
    assert loss.isfinite() and loss.item() > 0
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(g.isfinite().all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    assert not torch.equal(before, tracked_parameter.detach())


def test_official_cpp_extension_load():
    # This YOLOX extension is CPU C++, not a custom CUDA kernel.
    module = FastCOCOEvalOp().load()
    assert hasattr(module, "COCOevalEvaluateImages")
