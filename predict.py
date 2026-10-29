"""Run one RGB tile through the exported QAC-Net TorchScript model."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an RGB satellite tile")
    parser.add_argument("--model", help="TorchScript path (defaults to the packaged model)")
    parser.add_argument("--classes", help="Class mapping JSON (defaults to the packaged mapping)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    packaged = base if (base / "qacnet_torchscript.pt").exists() else base / "submission"
    model_path = Path(args.model) if args.model else packaged / "qacnet_torchscript.pt"
    classes_path = Path(args.classes) if args.classes else packaged / "model_config.json"
    config = json.loads(classes_path.read_text())
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    image = Image.open(args.image).convert("RGB")
    tensor = Compose([Resize((64, 64)), ToTensor()])(image).unsqueeze(0)
    with torch.inference_mode():
        landcover_logits, quality_logits = model(tensor)
        landcover_prob = landcover_logits.softmax(1)[0]
        quality_prob = quality_logits.softmax(1)[0]
    class_id = int(landcover_prob.argmax())
    quality_id = int(quality_prob.argmax())
    # Normalized predictive entropy summarizes how spread out each softmax is.
    # It is an ambiguity indicator, not a calibrated probability of error.
    tiny = torch.finfo(landcover_prob.dtype).tiny
    landcover_entropy = -(landcover_prob * landcover_prob.clamp_min(tiny).log()).sum()
    quality_entropy = -(quality_prob * quality_prob.clamp_min(tiny).log()).sum()
    landcover_uncertainty = landcover_entropy / torch.log(torch.tensor(float(landcover_prob.numel())))
    quality_uncertainty = quality_entropy / torch.log(torch.tensor(float(quality_prob.numel())))
    print(json.dumps({
        "land_cover": config["classes"][class_id],
        "land_cover_confidence": round(float(landcover_prob[class_id]), 4),
        "land_cover_uncertainty_normalized_entropy": round(float(landcover_uncertainty), 4),
        "input_quality": ["clean", "synthetically_corrupted"][quality_id],
        "synthetic_corruption_confidence": round(float(quality_prob[quality_id]), 4),
        "quality_uncertainty_normalized_entropy": round(float(quality_uncertainty), 4),
        "uncertainty_note": "Normalized softmax entropy measures prediction ambiguity; it is not calibrated to error probability.",
        "quality_head_note": "Trained on synthetic corruption labels; this is not cloud detection.",
    }, indent=2))


if __name__ == "__main__":
    main()
