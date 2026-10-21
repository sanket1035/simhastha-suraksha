# How it works

One flow: **frame -> CSRNet -> density map -> count -> density (people per m2) -> risk level -> alert**.
The model only counts people. The safety decision comes from density and the thresholds in `config/risk_thresholds.yaml`.

## Files

| File | What it does | Key idea |
|------|--------------|----------|
| `src/density.py` | Builds a density map from head points | A Gaussian blob at every head, each summing to 1, so the map sums to the people count. Geometry-adaptive sigma for dense crowds, fixed sigma for sparse ones |
| `src/data_prep.py` | Converts ShanghaiTech annotations into density maps and index CSVs | Run once before training |
| `src/dataset.py` | PyTorch datasets | Density mode downsamples by 8 with sum pooling, which keeps the count exact |
| `src/models/mlp.py`, `cnn.py` | Baselines that predict the count directly | Weak on purpose, to show why density maps are needed |
| `src/models/csrnet.py` | CSRNet | Pretrained VGG16 frontend (first 10 conv layers) plus dilated convolutions that see a wide area without shrinking the map |
| `src/train.py` | Trains any of the three models | Options for Dropout, L2, activation, optimizer, freezing the frontend, mixed precision, WandB. Best checkpoint by validation MAE |
| `src/evaluate.py` | Test MAE and RMSE, predictions.csv, worst-case overlays | Reports what was measured, nothing else |
| `src/export_onnx.py` | CSRNet to ONNX and a check against PyTorch | ONNX Runtime keeps the web app light on CPU |
| `src/predict.py` | `predict(image) -> (count, density_map)` | ONNX model if present, otherwise a clearly flagged demo stub |
| `src/risk.py` | Density, risk level, alert smoother | Escalate fast, de-escalate slowly |
| `src/pipeline.py` | Frames to history and alerts for one zone | UI-free, so it is unit-tested |
| `src/heatmap.py` | Heatmap overlay on a frame | Colours are relative to the densest spot in that frame |
| `src/video.py` | Samples frames from a video | Stand-in for a camera feed |
| `src/perceptron.py` | Perceptron from scratch | Learns AND/OR and the unsafe-zone rule (a straight line in count/area space) |
| `app/app.py` | Streamlit control-room dashboard | Zone tabs, heatmap, count, density, risk badge, alert log |

## Known limits

- Trained on public datasets, not on real Kumbh footage.
- Density is only meaningful if the area entered matches the ground visible to the camera.
- Risk thresholds are placeholders to be validated.
- Decision-support aid; it does not identify individuals.
