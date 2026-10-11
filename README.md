# सिंहस्थ Suraksha

**AI-Based Crowd Density Estimation and Risk Alert System for Simhastha Kumbh Mela**

> सिंहस्थ सुरक्षा: गर्दीवर एआयची नजर



![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)




![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)




![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?logo=opencv&logoColor=white)




![Model](https://img.shields.io/badge/Model-CSRNet%20(VGG16)

-8A2BE2)


![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)




![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Tracked-FFBE00?logo=weightsandbiases&logoColor=black)




![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)




![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?logo=huggingface&logoColor=black)




![Status](https://img.shields.io/badge/Status-In%20Development-orange)




![License](https://img.shields.io/badge/License-MIT-yellow)



> 🚀 **Live Demo →** Coming soon

## Quick Links
- 🚀 Live Demo (coming soon)
- 🎥 Demo Video (coming soon)
- 📊 Results (see below)
- 📖 Datasets: ShanghaiTech, UCF-QNRF

## Overview
Estimates crowd count and density from images/video and classifies each zone as
Low / Medium / High / Critical risk. Intended as a decision-support aid, not a
replacement for administrative or police judgement.

## Approach
1. Baseline MLP
2. Custom CNN with augmentation
3. CSRNet (VGG16 transfer learning) on density maps
4. Risk classification: people/m² thresholds

## Results
| Model | MAE | RMSE |
|-------|-----|------|
| MLP (baseline) | TBD | TBD |
| Custom CNN | TBD | TBD |
| CSRNet | TBD | TBD |

## Limitations
Trained on public datasets (ShanghaiTech, UCF-QNRF), not on real Kumbh footage.
Performance may drop on a different camera angle or crowd density.

## Setup
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Team
| Name | Role |
|------|------|
| Atharv | Data + baseline models |
| Sanket | CSRNet training + tuning |
| Purva | Risk logic + evaluation |
| Rahil | App + deployment |

## Acknowledgements
ShanghaiTech and UCF-QNRF datasets, CSRNet (Li et al., 2018)
