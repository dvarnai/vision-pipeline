import torch
from PIL import Image

from src.inference.prediction import InferenceBundle, predict_image


class FixedLogitModel(torch.nn.Module):
    def forward(self, images):
        batch_size = images.shape[0]
        logits = torch.tensor([[0.0, 2.0, 1.0]], dtype=torch.float32)
        return logits.repeat(batch_size, 1)


def _bundle():
    return InferenceBundle(
        checkpoint_path="checkpoints/example.pt",
        config_module_name="tests.fake_config",
        checkpoint_epoch=1,
        config=None,
        runtime_config={},
        class_weights=None,
        model=FixedLogitModel(),
        transform=lambda image: torch.zeros((3, 4, 4), dtype=torch.float32),
        device=torch.device("cpu"),
        class_names=("buildings", "forest", "glacier"),
        label_contract_version="intel-scene-v1",
        model_version="example-model",
        preprocessing_version="example-preprocess",
    )


def test_predict_image_returns_top_k_predictions_and_versions():
    result = predict_image(_bundle(), Image.new("RGB", (4, 4)), top_k=2)

    assert result["predicted_label"] == "forest"
    assert result["confidence"] == result["top_k"][0]["confidence"]
    assert [item["label"] for item in result["top_k"]] == ["forest", "glacier"]
    assert result["model_version"] == "example-model"
    assert result["preprocessing_version"] == "example-preprocess"
    assert result["label_contract_version"] == "intel-scene-v1"
    assert result["latency_ms"] >= 0


def test_predict_image_clamps_top_k_to_class_count():
    result = predict_image(_bundle(), Image.new("RGB", (4, 4)), top_k=10)

    assert len(result["top_k"]) == 3
