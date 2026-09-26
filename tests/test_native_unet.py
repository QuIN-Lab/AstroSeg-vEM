import torch

from segmentation.models.native_unet import NativeUNet


def test_forward_pass_shape_default():
    model = NativeUNet(in_channels=3, num_classes=2)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2, 64, 64)


def test_forward_pass_shape_custom_channels_and_classes():
    model = NativeUNet(in_channels=1, num_classes=4)
    model.eval()
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, 32, 32)


def test_output_is_finite():
    model = NativeUNet(in_channels=3, num_classes=2)
    model.eval()
    x = torch.randn(1, 3, 48, 48)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()
