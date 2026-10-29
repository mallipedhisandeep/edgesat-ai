# EdgeSat AI / QAC-Net

A compact EuroSAT RGB land-cover classifier with a second head for recognizing synthetic sensor corruption. It does **not** detect clouds.

## What is included

- Training, data, model, and prediction scripts.
- Colab notebook and its source ZIP for the organizer’s execution workflow.
- Both measured Colab checkpoints, TorchScript exports, split manifest, and JSON results.
- `SUBMISSION_REPORT.md` with honest target comparisons and `CODE_WALKTHROUGH.md` explaining the flow.

The EuroSAT image dataset is not committed. The training script downloads it when run with `--download`.

## Run in Google Colab

Open `EdgeSatAI_Colab.ipynb` in Colab, choose a GPU runtime if available, upload `EdgeSatAI_Submission.zip` when prompted, then use **Runtime → Run all**. The notebook trains clean-only and noise-augmented models and downloads results.

## Interactive dashboard

Install the dependencies above, then launch the local browser dashboard:

```bash
python3 -m streamlit run dashboard.py
```

The page lets you upload an image, select the clean-only or noise-trained model, and view land-cover scores and uncalibrated entropy indicators. The synthetic-corruption output is shown only for the noise-trained model; the clean-only quality head was not trained on corrupted examples. It runs locally and does not detect clouds.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 run.py --download --mode noise --epochs 80 --width 48 --batch-size 64 --output outputs/noise
```

## Try the saved Colab model

After EuroSAT has been downloaded, from this folder run:

```bash
python3 predict.py --model artifacts/noise/qacnet_torchscript.pt --classes artifacts/model_config.json --image "data/eurosat/eurosat/2750/Pasture/Pasture_509.jpg"
```

For a different tile, replace the image path with a local RGB image. The quality output means clean or synthetically corrupted only. Predictions also include normalized softmax entropy as an uncalibrated ambiguity indicator, not an error probability. The JSON also includes normalized softmax entropy for each prediction; it indicates ambiguity but is not calibrated to error probability.

## Measured results

See [`SUBMISSION_REPORT.md`](SUBMISSION_REPORT.md). The clean-only model reaches 82.09% clean macro-F1; the noise-augmented model reaches 77.05%. The 85% target is not met. Noise augmentation helps under all tested corruptions, but only the channel-drop F1 drop meets the 8-point target. The TorchScript model is 0.449 MB and Colab CPU latency is 2.23 ms/tile at batch 1; timing must be rechecked on the target onboard hardware.



## Hackathon submission files

The required two-page note and five-slide deck are in [`deliverables/`](deliverables/). The note reports held-out metrics, identifies unmet team goals, and states that the synthetic quality head is not a cloud detector. The handout gives qualitative trade-offs; numeric thresholds are team goals, not organizer requirements.

The current model does not detect clouds: EuroSAT has no cloud labels. A cloud-labelled dataset and a trained/evaluated cloud head are still needed.
# edgesat-ai
By Cosmic Pulse
