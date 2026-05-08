<div align="center">

# 🪪 Egyptian ID Card Recognition System

### AI-Powered OCR, Field Extraction & Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLO](https://img.shields.io/badge/YOLO-v8-00FFFF?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-Arabic%20%7C%20English-green?style=for-the-badge)](https://github.com/JaidedAI/EasyOCR)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<p>A production-ready Python application that detects, reads, and verifies Egyptian National ID cards using a multi-model AI pipeline — combining YOLO object detection, TrOCR, PaddleOCR, and EasyOCR for maximum accuracy.</p>

![Demo Banner](https://via.placeholder.com/900x300/1a1a2e/ffffff?text=Egyptian+ID+Card+Recognition+System)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Extracted Fields](#-extracted-fields)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## 🔍 Overview

The **Egyptian ID Card Recognition System** automates the process of scanning, reading, and validating Egyptian National ID cards. It is designed for use in government portals, banking KYC pipelines, and identity verification workflows.

The system uses a **multi-stage AI pipeline**:
1. **YOLO** detects and crops the ID card from any uploaded image
2. **Field Detection** locates individual fields on the card (name, NID, address, etc.)
3. **OCR Ensemble** (TrOCR + PaddleOCR + EasyOCR) extracts Arabic & English text
4. **ID Decoder** parses the 14-digit national number to derive birth date, governorate, and gender
5. **Fraud Detection** flags potentially fake or tampered documents

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **AI Card Detection** | Automatically locates and crops ID cards from any photo angle |
| 🔤 **Multi-Engine OCR** | Ensemble of TrOCR, PaddleOCR & EasyOCR for Arabic + English |
| 🧠 **Smart ID Decoding** | Deciphers the 14-digit NID to extract birth info, governorate & gender |
| 🛡️ **Fraud Detection** | Validates ID photo authenticity and cross-checks personal details |
| 📁 **Batch Processing** | Process hundreds of ID images in one run with a progress tracker |
| 👁️ **Auto Folder Monitor** | Watches a folder and auto-processes newly added images in real time |
| 🗄️ **SQLite Database** | Persistent storage with confidence scores, OCR method, and status tracking |
| 📊 **Analytics Dashboard** | Visual breakdowns by governorate, gender, age group, and processing method |
| 📤 **Excel Export** | One-click export of all records to `.xlsx` |
| 🌐 **Streamlit Web UI** | Clean, responsive dashboard — no frontend code needed |

---

## 🏗️ System Architecture

```
Input Image
     │
     ▼
┌─────────────────────┐
│  YOLO ID Card       │  ← detect_id_card.pt
│  Detection & Crop   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  YOLO Field         │  ← detect_objects.pt
│  Detection          │  (firstName, lastName, address, nid…)
└────────┬────────────┘
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────┐                   ┌──────────────────────┐
│  OCR Ensemble   │                   │  NID Region          │
│  TrOCR          │                   │  YOLO Digit Detect   │
│  PaddleOCR      │                   │  ← detect_id.pt      │
│  EasyOCR        │                   │  EasyOCR Fallback    │
└────────┬────────┘                   └──────────┬───────────┘
         │                                       │
         └──────────────┬────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  ID Decoder      │  Birth Date · Governorate · Gender
              │  Fraud Validator │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  SQLite DB       │
              │  Streamlit UI    │
              └──────────────────┘
```

---

## 📋 Extracted Fields

| Field | Source |
|---|---|
| **First Name** | OCR on field region |
| **Last Name** | OCR on field region |
| **Full Name** | Merged from first + last |
| **National ID Number** | YOLO digit detection + OCR fallback |
| **Address** | OCR on address region |
| **Birth Date** | Decoded from NID digits 2–7 |
| **Governorate** | Decoded from NID digits 8–9 (all 27 governorates supported) |
| **Gender** | Decoded from NID digit 13 |

---

## ⚙️ Installation

### Prerequisites

- Python **3.8+**
- pip
- (Optional) CUDA-capable GPU for faster inference

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/NASO7Y/ocr_egyptian_ID.git
cd ocr_egyptian_ID

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run APP.py
```

> 💡 **Tip:** Make sure the three YOLO model files (`detect_id_card.pt`, `detect_objects.pt`, `detect_id.pt`) are placed in the project root before running.

---

## 🚀 Usage

### Single Image Processing
Navigate to the **Home** tab → upload any image containing an Egyptian ID card → the system extracts and displays all fields instantly.

### Batch Processing
Go to **Batch Processing** → upload multiple images → click **Process All** → view per-image results and aggregate metrics (success rate, average confidence, field completeness).

### Automatic Folder Monitoring
Open the **Auto Monitor** tab → enter a local folder path → click **Start Monitoring**. Any new image dropped into that folder will be automatically processed and saved to the database.

### Database & Export
The **Database** tab shows all processed records with color-coded confidence levels. Use **Export to Excel** to download a full `.xlsx` report.

---

## 🧠 Model Details

| Model | Purpose | Notes |
|---|---|---|
| `detect_id_card.pt` | Detect & crop ID card from image | Custom-trained YOLO |
| `detect_objects.pt` | Detect field regions on the card | Custom-trained YOLO |
| `detect_id.pt` | Digit-by-digit NID detection | Custom-trained YOLO |
| `TrOCR` | High-accuracy printed text OCR | `microsoft/trocr-base-printed` |
| `PaddleOCR` | Multilingual OCR with Arabic | `lang='ar'` + `lang='en'` |
| `EasyOCR` | Robust fallback OCR | Arabic + English readers |

The OCR pipeline uses an **ensemble strategy**: all available models run on each region, and the best result is selected — maximizing robustness against poor scan quality, skew, and lighting variation.

---

## 📁 Project Structure

```
ocr_egyptian_ID/
│
├── APP.py                  # Original Streamlit application
├── improved_app.py         # Enhanced v3.0 Streamlit application
├── utils.py                # Core OCR & YOLO processing utilities
├── improved_utils.py       # Modern multi-engine OCR utilities
│
├── detect_id_card.pt       # YOLO model: card detection
├── detect_objects.pt       # YOLO model: field detection
├── detect_id.pt            # YOLO model: digit detection
│
├── id_cards_database.db    # SQLite database (auto-created)
├── settings.json           # Saved app settings (auto-created)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🗺️ Roadmap

- [x] Single image processing with YOLO + EasyOCR
- [x] Batch processing with progress tracking
- [x] Automatic folder monitoring
- [x] Multi-engine OCR ensemble (TrOCR + PaddleOCR + EasyOCR)
- [x] SQLite database with confidence scoring
- [x] Excel export
- [ ] REST API endpoint (FastAPI)
- [ ] Docker containerization
- [ ] Cloud deployment guide (AWS / GCP)
- [ ] Support for both sides of the ID card
- [ ] Improved fraud detection with deep learning

---

## 🤝 Contributing

Contributions are welcome and appreciated! Here's how to get started:

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
# Open a Pull Request
```

Please make sure your code:
- Follows PEP 8 style guidelines
- Includes relevant docstrings
- Passes existing tests (or adds new ones where appropriate)

---

## 🙏 Acknowledgments

This project is built on top of excellent open-source tools:

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection backbone
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — Arabic & English text recognition
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — multilingual OCR engine
- [TrOCR](https://huggingface.co/microsoft/trocr-base-printed) — transformer-based OCR by Microsoft
- [Streamlit](https://streamlit.io/) — interactive web interface framework

---

<div align="center">

Made with ❤️ in Egypt 🇪🇬

⭐ Star this repo if you found it useful!

</div>
