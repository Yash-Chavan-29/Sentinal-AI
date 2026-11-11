---
title: School Violence Detection
emoji: 🏫
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
models:
  - visionlab-ai/school-violence-detection-models
tags:
  - video-classification
  - violence-detection
  - 3d-cnn
  - x3d
  - i3d
  - slowfast
  - r2plus1d
  - school-safety
---

# 🏫 Safe School AI — Violence Detection Demo

Hệ thống phát hiện hành vi bạo lực trong video bằng 4 kiến trúc 3D CNN khác nhau (X3D-M, I3D-R50, SlowFast-R50, R(2+1)D-18). Đây là demo trực quan của đồ án tốt nghiệp tại UIT.

[![Models](https://img.shields.io/badge/🤗%20Models-visionlab--ai/school--violence--detection--models-yellow)](https://huggingface.co/visionlab-ai/school-violence-detection-models)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.1-orange.svg)](https://gradio.app)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red.svg)](https://pytorch.org)

## 🎯 Mô tả

Upload một video ngắn (4-10 giây), hệ thống sẽ phân loại thành **Violent** hoặc **Non-Violent** kèm xác suất confidence và thời gian inference.

Có **6 model** để chọn — 2 **final** (sau ablation) và 4 **baseline** (để đối chứng):

### ⭐ Final (sau ablation, F1 = 0.93)

| Model | Backbone | Frames | Threshold | GPU latency | Note |
|-------|----------|--------|-----------|-------------|------|
| **X3D-M Realtime** | x3d_m | **16** | **0.4** | ~3.77 ms | Default, nhanh nhất |
| **R(2+1)D-18 Best** | r2plus1d_18 | 32 | 0.5 | ~21 ms | Accuracy cao, ổn định |

### Baseline (trước ablation)

| Model | Accuracy | Speed (CPU) | Memory |
|-------|----------|-------------|--------|
| X3D-M | ~92.0% | ⚡ Fastest (~0.3-0.5s) | 💚 ~35 MB |
| I3D-R50 | ~90.75% | 🟡 Medium (~1.5-2.5s) | 🟠 ~312 MB |
| SlowFast-R50 | ~92.5% | 🔴 Slow (~3-5s) | 🔴 ~386 MB |
| R(2+1)D-18 | ~93.0% | 🟠 (~2-3s) | 🟠 ~358 MB |

> Trên HF Spaces free tier (2 vCPU), khuyến nghị dùng **X3D-M Realtime** cho phản hồi nhanh.

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐
│ Video Input │───▶│ Frame Sampling   │───▶│ Normalization│
│   (MP4)     │    │ (uniform 32 fr.) │    │ (mean/std)   │
└─────────────┘    └──────────────────┘    └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Gradio UI       │◀───│  Softmax + Label │◀───│ 3D CNN Model │
│  (label + bar)   │    │                  │    │ (4 backbones)│
└──────────────────┘    └──────────────────┘    └──────────────┘
```

- **Input**: Video clip → 32 frames uniformly sampled → resize 224×224 → tensor `(1, 3, 32, 224, 224)`
- **Inference**: Forward pass → softmax → 2 logits (non-violent, violent)
- **Output**: Top-1 label, confidence bar, latency

## 📁 Files

```
school-violence-detection-app/  (HF Space)
├── README.md                 # This file (YAML frontmatter for HF)
├── app.py                    # Gradio app entry point
├── model_loader.py           # Fetch checkpoint from Hub + LRU cache
├── temporal_analyzer.py      # Sliding-window inference + early-warning
├── video_utils.py            # Video preprocessing
├── sv_model/                 # 3D CNN wrappers (X3D / I3D / SlowFast / R(2+1)D)
├── requirements.txt          # Production deps (Linux, Python 3.12)
├── requirements-local.txt    # macOS dev deps (no decord)
├── .gitattributes            # Git LFS config
├── DEPLOYMENT_INFO.txt       # Push instructions
└── examples/                 # Sample videos (optional)
```

Checkpoints **không** nằm trong Space repo - được auto-download từ model repo
[`visionlab-ai/school-violence-detection-models`](https://huggingface.co/visionlab-ai/school-violence-detection-models) lần đầu load.

## 🚀 Run locally

### Prerequisites

- **Python 3.11** (recommended). `pytorchvideo==0.1.5` does not ship
  wheels for Python 3.13 yet.
  ```bash
  brew install python@3.11   # macOS
  ```
- ~3 GB disk free for torch + pytorchvideo.

### Setup

```bash
git clone https://huggingface.co/spaces/visionlab-ai/school-violence-detection-app
cd school-violence-detection-app

python3.11 -m venv .venv-space
source .venv-space/bin/activate          # Linux / macOS
# .venv-space\Scripts\activate           # Windows

pip install --upgrade pip
```

### Linux

```bash
pip install -r requirements.txt
```

### macOS (Apple Silicon / Intel)

The production `requirements.txt` pins `decord`, which has **no macOS
wheel**. Use `requirements-local.txt` instead — it drops `decord`
(the app falls back to OpenCV for video decoding) and loosens a few
Linux-only pins.

```bash
pip install -r requirements-local.txt
```

If `pytorchvideo` fails to build, install it from git:

```bash
pip install git+https://github.com/facebookresearch/pytorchvideo.git
```

### Run

```bash
python app.py          # one-shot, opens at http://127.0.0.1:7860
# OR — hot reload while editing CSS / app.py:
gradio app.py
```

The first analysis downloads the default checkpoint
(`final/final_x3d_realtime.pt`, ~12 MB) from the model repo into
`~/.cache/huggingface/hub/`. Subsequent runs are instant.

### UI-only mode (skip model inference)

If you just want to iterate on layout/CSS without installing the full
ML stack:

```bash
pip install "gradio==5.50.0" matplotlib opencv-python-headless \
            "huggingface_hub<1.0" numpy pyyaml
gradio app.py
```

The UI renders fully; clicking **Analyze** will error out (no torch
installed) — but it's enough to test layout, panels, dropdowns,
animations, etc.

### Pointing at a different model repo

```bash
VIOLENCE_MODEL_REPO_ID=your-user/your-models python app.py
```

## 🤖 About the Models

4 mô hình đều fine-tuned từ pretrain **Kinetics-400** trên tập **RWF-2000**:

- **X3D-M**: Efficient mobile-friendly 3D CNN (Facebook AI, CVPR 2020)
- **I3D-R50**: Inflated 3D ConvNet với ResNet-50 (Carreira & Zisserman, CVPR 2017)
- **SlowFast-R50**: Dual-pathway architecture (Feichtenhofer et al., ICCV 2019)
- **R(2+1)D-18**: Factorized 3D convolution (Tran et al., CVPR 2018)

Chi tiết kiến trúc và metrics: xem [model repo](https://huggingface.co/visionlab-ai/school-violence-detection-models).

## ⚠️ Limitations & Ethical Considerations

- **Dataset bias**: Train trên RWF-2000 (YouTube videos), có thể chệch khi áp dụng cho camera giám sát trường học (góc cao, độ phân giải thấp, ánh sáng đêm).
- **False positives**: Hoạt động thể thao mạnh (võ thuật, bóng đá) có thể bị nhận nhầm.
- **Không quyết định pháp lý**: Output mô hình chỉ là **gợi ý**, cần con người review trước khi hành động.
- **Privacy**: Demo không lưu trữ video; tuy nhiên việc deploy trong môi trường thật cần tuân thủ quy định bảo vệ dữ liệu cá nhân.

## 📚 Citation

```bibtex
@misc{school_violence_detection_2026,
  title  = {School Violence Detection: A Comparative Study of 3D CNN Architectures},
  author = {Nguyen, Nauthui7},
  year   = {2026},
  school = {University of Information Technology (UIT), VNU-HCM}
}
```

## 📝 License

MIT License.

## 👤 Author

- **visionlab-ai** — UIT Graduation Project, 2026
- Models: [visionlab-ai/school-violence-detection-models](https://huggingface.co/visionlab-ai/school-violence-detection-models)
