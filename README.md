# सिंहस्थ Suraksha

**AI-based crowd density estimation and risk alert system for the Simhastha Kumbh Mela, Nashik**

> सिंहस्थ सुरक्षा: गर्दीवर एआयची नजर

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white) ![Model](https://img.shields.io/badge/Model-CSRNet_VGG16-8A2BE2?style=flat-square) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white) ![CI](https://github.com/sanket1035/simhastha-suraksha/actions/workflows/ci.yml/badge.svg) ![Status](https://img.shields.io/badge/Status-In_Development-orange?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> 🚧 **Work in progress.** Results, demo link and setup steps are added only after they exist and have been measured.

## Why this project

The Simhastha Kumbh Mela at Nashik-Trimbakeshwar brings millions of pilgrims to narrow ghats and roads, where a crowd that builds up unnoticed can turn into a stampede. Operators watching many camera feeds at once cannot judge density by eye across every zone, every minute.

## What it does

- Reads a camera or video feed and estimates how many people are in each frame.
- Produces a **density heatmap** overlaid on the frame, so crowded spots are visible at a glance.
- Converts the count into **people per square metre** using the area of the zone being watched.
- Classifies each zone as **Low, Medium, High or Critical** and raises an alert when the level changes.

## Who it is for

Control-room operators, police and event administrators. It is **not** a pilgrim-facing app: the input comes from fixed cameras, not from people uploading photos.

## How it works

```
Camera / video frame
        |
        v
  CSRNet model  -->  density map  -->  sum = people count
        |
        v
  count / zone area (m2)  =  density (people per m2)
        |
        v
  Risk level: Low | Medium | High | Critical
        |
        v
  Dashboard: heatmap overlay, count, risk badge, alert log
```

The model only counts people. The safety decision comes from density and thresholds, which are kept in a config file.

| Level | Density (people per m2) |
|-------|-------------------------|
| Low | below 1 |
| Medium | 1 to 2.5 |
| High | 2.5 to 4 |
| Critical | 4 or more |

> These thresholds are **placeholders** based on common crowd-safety rules of thumb. They are configurable and need validation before any real use.

## Models compared

| Step | Model | Purpose |
|------|-------|---------|
| 1 | MLP | Baseline that predicts the count directly from the image |
| 2 | Custom CNN | Stronger baseline with data augmentation |
| 3 | CSRNet (VGG16 frontend + dilated backend) | Density map estimation, using pretrained weights from [rootstrap-org/crowd-counting](https://huggingface.co/rootstrap-org/crowd-counting) (MIT license). This project did not train these weights |

`src/train.py` also supports Dropout, L2 regularization, Sigmoid/Tanh/ReLU, freezing the CSRNet frontend and Weights and Biases logging. These experiments have not been run yet.

## Results

Measured on the ShanghaiTech Part A test split (182 images) with `src/evaluate.py`. Lower is better.

| Model | MAE | RMSE |
|-------|-----|------|
| MLP (baseline) | 312.36 | 425.08 |
| Custom CNN | 259.37 | 360.53 |
| CSRNet (pretrained weights, used as is) | 66.68 | 106.92 |

Notes:

- The CSRNet weights come from [rootstrap-org/crowd-counting](https://huggingface.co/rootstrap-org/crowd-counting). Its model card says they were trained on ShanghaiTech Part B. This project did not verify that and did not train them; it only evaluated them.
- A 3-epoch fine-tune of those weights on Part A gave MAE 70.86 and RMSE 105.16 on the same test split, which is no improvement, so the weights are used as they are.
- The MLP and CNN were trained for 15 epochs each on 270 Part A training images (30 held out for validation) on a Kaggle T4 x2 GPU session. Training took about 1.5 minutes for the MLP and under 1 minute for the CNN.
- Single runs, no repeated seeds.

## Dashboard

Screenshots and a demo link will be added here.

The app has two views. The pilgrim view is public and read-only: a map of the zones, a plain-language
crowd level per zone and a heatmap (never the raw photo). The control room needs a staff password
(`SS_STAFF_PASSWORD`) and lets staff upload camera photos and see counts, density, alerts and trends.

A zone can have several camera angles. Each angle keeps its own latest photo and ground area; the zone's
risk level uses the total people over the total area of its angles, so one busy corner does not hide a quiet
one. Uploading again to the same angle replaces only that angle.

### Camera ingest

`python -m src.ingest --config config/cameras.yaml` polls each camera (snapshot URL, RTSP stream,
video file or image folder), runs the model and updates the dashboard. Run it on the same machine
as the app and point both at one folder with `SS_LIVE_DIR`. Edit `config/cameras.yaml` with the real
cameras and measured ground areas. Streamlit Community Cloud cannot run this worker; use the
Docker image or a server for a live deployment.

## Project structure

```
simhastha-suraksha/
├─ src/
│  ├─ predict.py        # predict(image) -> (count, density_map); ONNX model or demo stub
│  ├─ risk.py           # risk_level(count, area_m2) and the alert smoother
│  ├─ density.py        # head points -> density maps
│  ├─ data_prep.py      # ShanghaiTech -> density maps + index CSVs
│  ├─ dataset.py        # PyTorch datasets
│  ├─ models/           # MLP, CNN, CSRNet
│  ├─ train.py          # training (Dropout, L2, activations, freeze/fine-tune, WandB)
│  ├─ evaluate.py       # MAE, RMSE, predictions.csv, worst cases
│  ├─ export_onnx.py    # CSRNet -> ONNX (verified against PyTorch)
│  ├─ pipeline.py       # frames -> risk history and alerts
│  ├─ heatmap.py, video.py, metrics.py, perceptron.py
├─ app/app.py           # Streamlit control-room dashboard
├─ config/              # risk thresholds, WandB sweep
├─ notebooks/           # Kaggle training notebook
├─ experiments/         # perceptron demo
├─ tests/               # unit tests (python -m unittest discover -s tests)
├─ docs/HOW_IT_WORKS.md
├─ Dockerfile, requirements.txt, requirements-train.txt
```

## Getting started

```bash
git clone https://github.com/sanket1035/simhastha-suraksha.git
cd simhastha-suraksha
pip install -r requirements.txt
python -m unittest discover -s tests -v     # run the tests
streamlit run app/app.py                    # dashboard
```

Until a trained model is added, the dashboard runs in **demo mode** with simulated counts (it says so on screen).
To use a real model, place `models/csrnet.onnx` in the repo folder, or set `SS_MODEL_URL` to a download link (for example a GitHub Release asset) and the app fetches it on start.

**Training** (GPU, for example Kaggle): open `notebooks/train_on_kaggle.ipynb`, or run:

```bash
pip install -r requirements-train.txt
python -m src.data_prep --root <ShanghaiTech folder> --part A --out data/processed
python -m src.train --model csrnet --train-index data/processed/part_A/train_index.csv --out runs/csrnet_A --amp
python -m src.evaluate --checkpoint runs/csrnet_A/best.pth --test-index data/processed/part_A/test_index.csv --out runs/csrnet_A/eval
python -m src.export_onnx --checkpoint runs/csrnet_A/best.pth --out models/csrnet.onnx
```

**Docker:**

```bash
docker build -t simhastha-suraksha .
docker run -p 8501:8501 simhastha-suraksha
```

## Datasets and references

Datasets are not included in this repository. Download them from their original sources and follow their licenses.

- ShanghaiTech crowd counting (Part A and Part B): Zhang et al., *Single-Image Crowd Counting via Multi-Column Convolutional Neural Network*, CVPR 2016 ([arXiv:1604.02683](https://arxiv.org/abs/1604.02683))
- UCF-QNRF (optional): [crcv.ucf.edu/data/ucf-qnrf](https://www.crcv.ucf.edu/data/ucf-qnrf/)
- CSRNet: Li et al., *CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes*, CVPR 2018 ([arXiv:1802.10062](https://arxiv.org/abs/1802.10062))

## Future scope

- Live camera feeds on a hosted deployment: cameras push frames to an API with shared storage, so the
  public dashboard updates without a staff upload. Today the ingest worker (`src.ingest`) runs on the
  same machine as the app.
- Per-camera positions and fields of view on the map, and more zones.
- Alerts to control-room staff by SMS or WhatsApp when a zone crosses a risk level.
- Perspective-aware counting for far-away crowds, which the current model undercounts.

## Limitations and responsible use

- Evaluated only on a public dataset (ShanghaiTech Part A), **not on real Kumbh footage**. Accuracy on Kumbh scenes is unknown until tested on local images.
- Density is only meaningful when the area matches the ground actually visible to the camera.
- This is a **decision-support aid**. It does not replace administrative or police judgement.
- It counts people in aggregate and does not identify individuals.

## Team

- [Atharv](https://github.com/ajxlv)
- [Sanket](https://github.com/sanket1035)
- [Purva](https://github.com/purvachopade1665)
- [Rahil](https://github.com/Rahilshah2234)

## License

Released under the [MIT License](LICENSE).
