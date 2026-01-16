# 🛡️ DeepGuard - AI Deepfake Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.109-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/TensorFlow-2.15-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

**DeepGuard** is a comprehensive AI-powered deepfake detection system that analyzes videos and audio files to identify synthetic media manipulation. Using CNN-based visual analysis, spectral audio processing, and 468-point face landmark tracking, it provides multi-modal detection with high accuracy.

![DeepGuard Screenshot](frontend/assets/screenshot.png)

## ✨ Features

- 🎬 **Video Analysis** - CNN-based detection of visual artifacts, compression anomalies, and color inconsistencies
- 🎵 **Audio Analysis** - Spectral analysis using MFCC features to detect synthetic speech
- 👤 **Face Landmarks** - 468-point facial mesh analysis for blink patterns, lip-sync, and micro-expressions
- 📊 **Confidence Scoring** - Combined authenticity score with detailed breakdowns
- 🎨 **Modern UI** - Premium dark theme with glassmorphism and smooth animations
- 🔒 **Privacy First** - Files are deleted immediately after analysis
- 🚀 **Fast Processing** - Optimized for quick results without sacrificing accuracy

## 🖥️ Demo

| Frontend (GitHub Pages) | Backend API (Render) |
|-------------------------|----------------------|
| [![Frontend](https://img.shields.io/badge/Live-App-blue?style=for-the-badge&logo=github)](https://capgarrick.github.io/deepfake-detector/) | [![Backend](https://img.shields.io/badge/API-Server-green?style=for-the-badge&logo=render)](https://deepguard-api-d568.onrender.com) |

> **Note:** The backend on Render may spin down after inactivity. Allow 50-60 seconds for the first request.

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- FFmpeg (optional, for audio extraction from videos)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deepfake-detector.git
   cd deepfake-detector
   ```

2. **Set up Python virtual environment**
   ```bash
   cd backend
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the backend server**
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:8000`

5. **Open the frontend**
   
   Open `frontend/index.html` in your web browser, or serve it with a local server:
   ```bash
   cd ../frontend
   python -m http.server 3000
   ```
   Then visit `http://localhost:3000`

## 📁 Project Structure

```
deepfake-detector/
├── backend/
│   ├── app.py                    # FastAPI main application
│   ├── requirements.txt          # Python dependencies
│   ├── models/
│   │   ├── video_detector.py     # CNN video analysis
│   │   ├── audio_detector.py     # Spectral audio analysis
│   │   └── face_analyzer.py      # Face landmark detection
│   └── utils/
│       ├── preprocessing.py      # Media preprocessing
│       └── helpers.py            # Utility functions
│
├── frontend/
│   ├── index.html               # Main HTML
│   ├── css/
│   │   └── styles.css           # Premium styling
│   └── js/
│       └── app.js               # Frontend logic
│
├── README.md
├── .gitignore
└── LICENSE
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/api/health` | GET | Health check |
| `/api/analyze/video` | POST | Analyze video for deepfakes |
| `/api/analyze/audio` | POST | Analyze audio for deepfakes |
| `/api/analyze/full` | POST | Complete multi-modal analysis |
| `/api/tips` | GET | Get protection tips |

### Example API Usage

```python
import requests

# Analyze a video
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/analyze/full',
        files={'file': f}
    )
    result = response.json()
    print(f"Authenticity Score: {result['overall_result']['authenticity_score']}%")
```

## 🔬 Detection Methods

### Video Analysis (CNN)
- **Compression Artifacts**: Detects DCT block boundaries and JPEG artifacts
- **Color Consistency**: Analyzes LAB color space for blending anomalies
- **Noise Patterns**: Identifies unnatural noise characteristics in facial regions
- **Temporal Coherence**: Checks frame-to-frame consistency

### Audio Analysis (Spectral)
- **MFCC Features**: 20 Mel-frequency cepstral coefficients
- **Spectral Analysis**: Centroid, bandwidth, and rolloff detection
- **Voice Quality**: Jitter and shimmer measurements
- **Temporal Patterns**: Rhythm and pause analysis

### Face Landmark Analysis
- **468 Facial Points**: MediaPipe Face Mesh tracking
- **Blink Patterns**: Eye aspect ratio and blink frequency
- **Lip Sync**: Mouth movement correlation
- **Micro-expressions**: Facial movement dynamics

## 🎨 UI Features

- **Glassmorphism Design**: Modern frosted glass effect
- **Dark Theme**: Easy on the eyes with vibrant accents
- **Smooth Animations**: Engaging micro-interactions
- **Responsive**: Works on desktop and mobile
- **Drag & Drop**: Easy file upload experience
- **Real-time Progress**: Visual feedback during analysis

## 🛡️ Privacy & Security

- **No Storage**: Files are deleted immediately after analysis
- **No Logging**: We don't log any analyzed content
- **Local Processing**: All analysis happens on your server
- **CORS Enabled**: Configurable cross-origin settings

## 🚀 Deployment

### Docker

```bash
docker build -t deepguard .
docker run -p 8000:8000 deepguard
```

### Render / Railway

1. Connect your GitHub repository
2. Set the build command: `pip install -r backend/requirements.txt`
3. Set the start command: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is designed for educational and research purposes. While it uses advanced techniques to detect potential deepfakes, no detection system is 100% accurate. Always verify content through multiple sources and use critical thinking when evaluating media authenticity.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for face mesh detection
- [librosa](https://librosa.org/) for audio analysis
- [OpenCV](https://opencv.org/) for video processing
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework

---

<p align="center">
  Made with ❤️ to protect truth in the digital age
</p>
