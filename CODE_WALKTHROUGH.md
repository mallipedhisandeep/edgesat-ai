# Code walkthrough

## How the model runs

1. `data.py` loads EuroSAT (downloaded when needed), selects the shared 2,000-tile subset, reproduces the organizer’s split, and creates training augmentations. In noise mode it can add Gaussian, salt-and-pepper, or RGB-channel dropout corruption.
2. `model.py` defines QAC-Net: a compact convolutional network with depthwise-separable blocks, a 10-class land-cover head, and a 2-class clean/synthetic-corruption quality head.
3. `run.py` trains a selected mode, chooses a checkpoint from the internal validation set, evaluates on the held-out test set, exports TorchScript, and writes JSON metrics for macro-F1, accuracy, size, and latency.
4. `predict.py` loads a TorchScript model and an RGB image, resizes it to 64×64, and prints the predicted land-cover class, confidence, and synthetic-quality score.
5. `EdgeSatAI_Colab.ipynb` provides the organizer-requested Colab workflow for both clean-only and noise-augmented runs, results comparison, downloading artifacts, and a prediction demo.

## Try the supplied model

From this repository folder, run:

```bash
python3 predict.py --model artifacts/noise/qacnet_torchscript.pt --classes artifacts/model_config.json --image "data/eurosat/eurosat/2750/Pasture/Pasture_509.jpg"
```

The dataset path above exists after a training run downloads EuroSAT. You can supply any local RGB image with `--image`. The quality head does not detect clouds.

## Train again

```bash
python3 -m pip install -r requirements.txt
python3 run.py --download --mode noise --epochs 80 --width 48 --batch-size 64 --output outputs/noise
```

Use `--mode clean` for the clean-only baseline.



## Prediction uncertainty indicator

`predict.py` computes normalized entropy from each head’s softmax probabilities. It adds `land_cover_uncertainty_normalized_entropy` and `quality_uncertainty_normalized_entropy` to the JSON output (0 means concentrated probabilities; values nearer 1 mean a more spread-out distribution). These are uncalibrated ambiguity indicators, not probabilities that a prediction is wrong. Calibration and reliability have not been evaluated.
