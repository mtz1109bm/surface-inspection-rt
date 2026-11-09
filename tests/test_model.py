import pytest

torch = pytest.importorskip("torch")

from inspector.model import ConvAutoencoder, count_parameters  # noqa: E402


def test_autoencoder_output_matches_input_shape():
    model = ConvAutoencoder(64).eval()
    x = torch.rand(2, 1, 128, 128)
    y = model(x)
    assert y.shape == x.shape
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0


def test_model_is_small_enough_for_edge():
    assert count_parameters(ConvAutoencoder(64)) < 2_000_000
