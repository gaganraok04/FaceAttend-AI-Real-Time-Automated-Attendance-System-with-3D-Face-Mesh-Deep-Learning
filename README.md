# 🎓 FaceAttend AI: Real-Time Automated Attendance System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.14-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-3D%20Mesh-00bcd4.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, contactless student attendance management system powered by **Deep Learning (YOLOv8 & MediaPipe Face Landmarker)**, **Computer Vision (OpenCV Haar Cascades)**, and **Machine Learning (LBPH)** with an executive **Desktop GUI Dashboard**.

---

## 📌 Overview

**FaceAttend AI** eliminates manual roll-calls and proxy attendance by automating the entire lifecycle of student attendance using computer vision:
1. **Interactive Desktop GUI:** Modern, responsive Tkinter dashboard built in an executive Roasted Espresso & Warm Caramel palette.
2. **Student Face Enrollment:** Captures 20 normalized facial crops per student and maps identities in structured databases.
3. **1-Click Model Training:** Rapidly compiles Local Binary Patterns Histograms (LBPH) in seconds on standard CPUs.
4. **Live Attendance with Geometric 3D Mesh:** Simultaneously performs student detection, live identity verification, distance scoring, and visual wireframe tracking, automatically logging attendance to CSV records.

---

## ✨ Key Features

* **🖥️ Executive Graphical Dashboard (`gui_app.py`):**
  * Real-time metrics bar (Model size, Student count, Today's attendance counter).
  * 1-Click triggers for Live Attendance, Face Mesh webcam, and Model Training.
  * Inline student registration with automated camera capture.
  * Tabbed data browser for Live Attendance records and Registered Students with compact action buttons (`🔄 Refresh`, `📂 Open CSV`).
  * Real-time scrolling system activity log.

* **🧠 Hybrid AI Architecture:**
  * **YOLOv8 Deep CNN:** Custom-trained object detection model (`best.pt`) for student detection.
  * **MediaPipe Face Landmarker:** High-fidelity 478-landmark 3D facial regression network for real-time spatial mesh tracking.
  * **LBPH Texture Recognizer:** Local Binary Patterns Histograms for instantaneous on-device classification without GPU dependencies.
  * **Haar Cascade:** Fast frontal face localization for sample acquisition.

* **📐 Geometric Wireframe HUD:**
  * Clean, spacious light-brown geometric wireframe with open eye and mouth cutouts (no distracting facial dots).
  * Real-time status cards, student badges, and match confidence distances.

* **📊 Automated Data Persistence:**
  * Daily CSV attendance logging with timestamp tracking (`First Seen`, `Last Seen`, `Status`).
  * Duplicate prevention logic within session windows.

---

## 🏗️ System Architecture

```text
               ┌──────────────────────────────────────────┐
               │         Video Acquisition (Webcam)       │
               └────────────────────┬─────────────────────┘
                                    │
                                    ▼
               ┌──────────────────────────────────────────┐
               │    Face Localization (Haar / YOLOv8)     │
               └────────────────────┬─────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            │                                               │
            ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│ MediaPipe 3D Face Landmarker  │               │   LBPH Texture Feature Match  │
│ 478 Landmarks & Mesh Geometry │               │  Euclidean Distance Scoring   │
└───────────────┬───────────────┘               └───────────────┬───────────────┘
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        │
                                        ▼
               ┌──────────────────────────────────────────┐
               │       Identity Verification & HUD        │
               │    (Recognized Badge vs Unknown Alert)   │
               └────────────────────┬─────────────────────┘
                                    │
                                    ▼
               ┌──────────────────────────────────────────┐
               │      Automated Persistence (CSV)         │
               │   attendance.csv  │  students.csv        │
               └──────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Deep Learning** | YOLOv8 (Ultralytics), MediaPipe Face Landmarker, PyTorch |
| **Computer Vision** | OpenCV (`cv2`, `opencv-contrib-python`), Haar Cascade Classifiers |
| **Machine Learning** | Local Binary Patterns Histograms (LBPH) |
| **Desktop UI** | Python Tkinter, Custom `ttk` Stylers, Multithreading |
| **Data Storage** | Tabular CSV (`attendance.csv`, `students.csv`), Serialized YAML (`face_model.yml`) |

---

## 📂 Project Structure

```text
Student-attendance.v1i.yolov8/
├── registered_faces/                # Student face image dataset (20 images/student)
│   ├── 101/                         # Student ID folder
│   └── .gitkeep
├── runs/
│   └── detect/
│       └── train/weights/best.pt    # Custom trained YOLOv8 model weights
├── attendance.csv                   # Live attendance log records
├── attendance.py                    # Unified live attendance engine with 3D wireframe
├── face_labels.txt                  # ID-to-index label mappings
├── face_landmarker.task             # Google MediaPipe 3D face landmarker model
├── face_model.yml                   # Trained OpenCV LBPH face model
├── gui_app.py                       # Executive Desktop GUI Application
├── recognize_face.py                # Standalone face recognition tester
├── register_student.py              # CLI student enrollment script
├── requirements.txt                 # Python project dependencies
├── run_app.bat                      # 1-Click Windows batch launcher
├── students.csv                     # Student master roster
├── test.py                          # YOLOv8 evaluation script
├── train.py                         # YOLOv8 fine-tuning script
├── train_face_model.py              # LBPH face model trainer
├── webcam.py                        # Standalone 3D mesh wireframe demo
├── yolov8n.pt                       # Pre-trained base YOLOv8 model
├── LICENSE                          # MIT License
└── README.md                        # Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10 or higher installed.
* A working webcam.
* Git installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Student-Attendance-Face-Recognition.git
cd Student-Attendance-Face-Recognition
```

### 3. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Method 1: All-In-One Desktop GUI (Recommended)
Launch the graphical dashboard:
```bash
python gui_app.py
```
*(On Windows, you can also simply double-click **`run_app.bat`**)*

From the GUI, you can:
1. **Register Students:** Enter Student ID & Name, then click **`Capture Face & Register`** (takes 20 photos automatically).
2. **Train Model:** Click **`Train LBPH Face Model`** (takes ~2 seconds).
3. **Start Live Attendance:** Click **`Start Live Attendance`** to run real-time recognition with the live 3D geometric wireframe.
4. **View Records:** Switch tabs to inspect attendance logs or registered students, and click **`Open attendance.csv`** to open in Excel.

---

### Method 2: Command-Line Interface (Step-by-Step)

If you prefer running components individually via terminal:

1. **Register a new student:**
   ```bash
   python register_student.py
   ```
2. **Train the recognition model:**
   ```bash
   python train_face_model.py
   ```
3. **Run live attendance:**
   ```bash
   python attendance.py
   ```
4. **Test recognition confidence & distance:**
   ```bash
   python recognize_face.py
   ```
5. **Run geometric face mesh demo:**
   ```bash
   python webcam.py
   ```

---

## 📊 Sample Attendance Output (`attendance.csv`)

| Student_ID | Name | Date | Time | Status |
| :---: | :---: | :---: | :---: | :---: |
| 101 | GAGAN RAO K | 2026-11-08 | 08:35:12 | Present |
| 909 | Ramesh | 2026-11-08 | 08:36:04 | Present |

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).

## 👤 Author

* **GAGAN RAO K**
