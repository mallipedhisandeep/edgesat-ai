# EdgeSat AI / QAC-Net — measured Google Colab results

The event handout describes trade-offs but does not specify numeric thresholds; the thresholds below are team goals adopted from project materials. Both runs use the organizer’s fixed seed-42 subset of 2,000 EuroSAT RGB tiles and exact 1,600/400 train/test split. The training pool is subdivided into 1,440 fit and 160 validation tiles; the 400 test tiles remain held out.

| Measure | Clean-only | Noise-augmented | Target |
|---|---:|---:|---:|
| Clean macro-F1 | 82.09% | 77.05% | ≥85% |
| Clean accuracy | 83.25% | 78.75% | — |
| Gaussian macro-F1 | 2.14% | 59.75% | — |
| Gaussian F1 drop | 79.95 pp | 17.30 pp | ≤8 pp |
| Salt-pepper macro-F1 | 1.69% | 57.25% | — |
| Salt-pepper F1 drop | 80.39 pp | 19.80 pp | ≤8 pp |
| Channel-drop macro-F1 | 9.01% | 70.38% | — |
| Channel-drop F1 drop | 73.07 pp | 6.67 pp | ≤8 pp |

Deployment: 89,820 parameters; 0.449 MB TorchScript export; 2.23 ms/tile at batch 1 on the Colab CPU runtime.

## Interpretation

- The clean-only model has higher clean macro-F1 (82.09% vs. 77.05%), but it collapses under the specified synthetic corruptions.
- Noise augmentation improves corrupted-input scores. The channel-drop loss meets the 8-point goal (6.67 points); Gaussian and salt-and-pepper drops do not (17.30 and 19.80 points).
- Neither model reaches the 85% clean macro-F1 target. The size and measured Colab CPU latency are within the stated targets; confirm timing on target hardware.
- EuroSAT supplies land-cover labels, not cloud labels. The quality head recognizes synthetic corruption only and must not be described as a cloud detector. A labeled cloud dataset is needed for cloud evaluation.
- The example local prediction on Pasture_509.jpg returned Pasture at 92.63% confidence, and clean at 99.87% synthetic-corruption confidence. This is a demo output, not an aggregate accuracy metric.


The prediction script additionally reports normalized softmax entropy for both output heads. It is an uncalibrated ambiguity indicator, not an error probability; calibration metrics are not yet available.
