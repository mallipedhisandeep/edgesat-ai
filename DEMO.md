# Live prediction demo

## Setup

Extract the GitHub repository ZIP and open Terminal in the extracted `EdgeSatAI_GitHub` folder. Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The example image is part of EuroSAT and is downloaded the first time you train with `--download`. Alternatively, use any local RGB image with the `--image` option.

## Run the trained model

```bash
python3 predict.py --model artifacts/noise/qacnet_torchscript.pt --classes artifacts/model_config.json --image "data/eurosat/eurosat/2750/Pasture/Pasture_509.jpg"
```

The saved example output was `Pasture` at 92.63% land-cover confidence, with the quality head reporting `clean` at 99.87% synthetic-corruption confidence. The predictor now also reports normalized softmax entropy for both heads as an uncalibrated ambiguity indicator. This is one demo tile and not an aggregate score.

## Explain what happens

1. The image is converted to RGB and resized to 64×64.
2. TorchScript runs QAC-Net on the CPU and returns land-cover and quality logits.
3. The script converts logits to confidence scores and prints the top land-cover label and synthetic quality label.

The quality head detects synthetic corruption patterns only; it does not detect clouds. Entropy is not calibrated to the chance of error.



## Browser dashboard

From the repository root, install dependencies with `python3 -m pip install -r requirements.txt`, then launch `python3 -m streamlit run dashboard.py`. Upload a JPG, PNG, or TIFF tile and select either model variant. The page displays the image, class scores, and uncalibrated entropy indicators. It shows synthetic-corruption output only for the noise-trained variant; the clean-only quality head was not trained on corrupted examples.
