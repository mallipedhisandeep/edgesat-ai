"""Train and evaluate EdgeSat QAC-Net. See README.md for commands."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from data import Corruption, EuroSATSplit, load_splits
from model import QACNet, QACMobileNet


def build_model(backbone, width=32, pretrained=True):
    if backbone == "mobilenet_v3_small":
        return QACMobileNet(pretrained=pretrained)
    return QACNet(width=width)


def batch_corrupt(x, kind, severity, seed):
    gen = torch.Generator().manual_seed(seed)
    if kind == "channel_dropout":
        channel = int(torch.randint(0, x.shape[1], (), generator=gen).item())
        x = x.clone()
        x[:, channel] = 0.0
        return x
    return torch.stack([Corruption.apply(image, kind, severity, gen) for image in x])


def evaluate(model, loader, device, classes):
    model.eval(); y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y, _ in loader:
            land, _ = model(x.to(device))
            y_true.extend(y.numpy()); y_pred.extend(land.argmax(1).cpu().numpy())
    return {"macro_f1": f1_score(y_true, y_pred, average="macro"),
            "accuracy": accuracy_score(y_true, y_pred),
            "per_class_f1": dict(zip(classes, f1_score(y_true, y_pred, average=None).tolist())),
            "classification_report": classification_report(y_true, y_pred, target_names=classes,
                                                              output_dict=True, zero_division=0)}


def evaluate_quality(model, loader, device, seed):
    truth, pred = [], []
    model.eval()
    with torch.inference_mode():
        for batch_id, (x, _, _) in enumerate(loader):
            _, clean_logits = model(x.to(device))
            _, corrupt_logits = model(batch_corrupt(x, "gaussian", 0.15, seed + batch_id).to(device))
            truth.extend([0] * len(x) + [1] * len(x))
            pred.extend(clean_logits.argmax(1).cpu().tolist() + corrupt_logits.argmax(1).cpu().tolist())
    return f1_score(truth, pred, average="macro")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/eurosat", help="Dataset root (download destination or ImageFolder root)")
    p.add_argument("--output", default="outputs")
    p.add_argument("--download", action="store_true", help="Download EuroSAT if not present")
    p.add_argument("--mode", choices=["clean", "noise"], default="noise")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--width", type=int, default=48, help="Channel width of the compact network")
    p.add_argument("--epoch-offset", type=int, default=0,
                   help="Offset augmentation randomness when continuing from a checkpoint")
    p.add_argument("--backbone", choices=["qacnet", "mobilenet_v3_small"], default="qacnet")
    p.add_argument("--no-pretrained", action="store_true", help="Do not initialize MobileNet from ImageNet weights")
    p.add_argument("--freeze-backbone", action="store_true", help="Train only the new heads for faster transfer learning")
    p.add_argument("--unfreeze-last", type=int, default=0,
                   help="For MobileNet transfer learning, train only this many final feature blocks")
    p.add_argument("--init-checkpoint", help="Initialize from a prior best.pt checkpoint and continue fine-tuning")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    samples, split = load_splits(args.data, args.download, seed=args.seed)
    classes = split["class_names"]
    train_ds = EuroSATSplit(samples, split["train_indices"], True, args.mode == "noise", args.seed)
    val_ds = EuroSATSplit(samples, split["val_indices"])
    test_ds = EuroSATSplit(samples, split["test_indices"])
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=args.workers)
    pretrained = args.backbone == "mobilenet_v3_small" and not args.no_pretrained
    model = build_model(args.backbone, args.width, pretrained).to(device)
    if args.backbone == "mobilenet_v3_small":
        if args.freeze_backbone and args.unfreeze_last:
            p.error("--freeze-backbone and --unfreeze-last cannot be used together")
        if args.unfreeze_last < 0 or args.unfreeze_last > len(model.backbone.features):
            p.error("--unfreeze-last must be between 0 and the number of MobileNet feature blocks")
        if args.freeze_backbone:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False
        elif args.unfreeze_last:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False
            for block in model.backbone.features[-args.unfreeze_last:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        optimizer = torch.optim.AdamW([
            {"params": [v for v in model.backbone.parameters() if v.requires_grad], "lr": 1e-4},
            {"params": list(model.landcover_head.parameters()) + list(model.quality_head.parameters()), "lr": 1e-3},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    land_loss, quality_loss = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if initial.get("backbone", "qacnet") != args.backbone:
            p.error("--init-checkpoint backbone does not match --backbone")
        model.load_state_dict(initial["state_dict"])
        if args.epoch_offset:
            torch.manual_seed(args.seed + args.epoch_offset)
            np.random.seed(args.seed + args.epoch_offset)
    best_f1 = evaluate(model, val_loader, device, classes)["macro_f1"]
    torch.save({"state_dict": model.cpu().state_dict(), "classes": classes,
                "mode": args.mode, "seed": args.seed, "width": args.width,
                "backbone": args.backbone, "pretrained": pretrained}, out / "best.pt")
    model.to(device)
    for epoch in range(args.epochs):
        model.train()
        if args.backbone == "mobilenet_v3_small" and args.freeze_backbone:
            model.backbone.eval()
        elif args.backbone == "mobilenet_v3_small" and args.unfreeze_last:
            model.backbone.eval()
            for block in model.backbone.features[-args.unfreeze_last:]:
                block.train()
        train_ds.epoch = epoch + args.epoch_offset
        running = 0.0
        for x, y, q in train_loader:
            x, y, q = x.to(device), y.to(device), q.to(device)
            optimizer.zero_grad(set_to_none=True)
            land, quality = model(x)
            loss = land_loss(land, y) + 0.3 * quality_loss(quality, q)
            loss.backward(); optimizer.step(); running += loss.item() * len(x)
        scheduler.step()
        metrics = evaluate(model, val_loader, device, classes)
        print(f"epoch {epoch+1:02d}/{args.epochs} loss={running/len(train_ds):.4f} val_macro_f1={metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save({"state_dict": model.cpu().state_dict(), "classes": classes,
                        "mode": args.mode, "seed": args.seed, "width": args.width,
                        "backbone": args.backbone, "pretrained": pretrained}, out / "best.pt")
            model.to(device)
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    results = {"mode": args.mode, "target_is_result": False, "split_counts": {
        "official_train_pool": len(split["train_pool_indices"]),
        "fit_train": len(train_ds), "validation": len(val_ds), "test": len(test_ds)},
        "split_protocol": split["split_protocol"]}
    results["clean"] = evaluate(model, test_loader, device, classes)
    results["quality_head_synthetic_clean_vs_gaussian_f1"] = evaluate_quality(
        model, test_loader, device, args.seed)
    corruption_results = {}
    for kind, severity in [("gaussian", 0.15), ("salt_pepper", 0.15), ("channel_dropout", 1.0)]:
        ys, ps = [], []
        with torch.inference_mode():
            for batch_id, (x, y, _) in enumerate(test_loader):
                x = batch_corrupt(x, kind, severity, args.seed + batch_id).to(device)
                pred, _ = model(x)
                ys.extend(y.numpy()); ps.extend(pred.argmax(1).cpu().numpy())
        score = f1_score(ys, ps, average="macro")
        corruption_results[kind] = {"severity": severity, "macro_f1": score,
                                     "drop_percentage_points": (results["clean"]["macro_f1"] - score) * 100}
    results["noise"] = corruption_results
    model_cpu = build_model(args.backbone, args.width, pretrained=False)
    model_cpu.load_state_dict(ckpt["state_dict"]); model_cpu.eval()
    scripted = torch.jit.trace(model_cpu, torch.zeros(1, 3, 64, 64))
    scripted.save(str(out / "qacnet_torchscript.pt"))
    size_mb = os.path.getsize(out / "qacnet_torchscript.pt") / (1024 * 1024)
    x = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        for _ in range(20): model_cpu(x)
        start = time.perf_counter()
        for _ in range(200): model_cpu(x)
        latency_ms = (time.perf_counter() - start) * 1000 / 200
    results["deployment"] = {"torchscript_size_mb": size_mb, "cpu_latency_ms_per_tile": latency_ms,
                             "benchmark_device": "cpu", "latency_batch_size": 1}
    parameter_count = sum(parameter.numel() for parameter in model_cpu.parameters())
    results["deployment"]["parameter_count"] = parameter_count
    results["deployment"]["fp32_parameter_footprint_mb"] = parameter_count * 4 / (1024 * 1024)
    results["targets"] = {"clean_macro_f1_min": 0.85, "model_size_mb_max": 5,
                          "cpu_latency_ms_max": 10, "moderate_noise_f1_drop_pp_max": 8}
    with (out / "results.json").open("w") as f: json.dump(results, f, indent=2)
    print(json.dumps({"clean_macro_f1": results["clean"]["macro_f1"], "noise": results["noise"],
                      "deployment": results["deployment"], "saved_to": str(out)}, indent=2))


if __name__ == "__main__": main()
