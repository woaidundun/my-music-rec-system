# Music Screenshot OCR Demo

A minimal local web app for your MSc project.

## What it does
- Upload a Spotify/playlist screenshot
- Run OCR with PaddleOCR
- Show all detected text lines
- Guess likely `song title + artist` pairs from the OCR output

## Project structure
```text
music_ocr_app/
├── app.py
├── ocr_service.py
├── requirements.txt
├── README.md
├── static/
│   └── style.css
├── templates/
│   └── index.html
└── uploads/
```

## Recommended environment
Python 3.10 or 3.11 is the safest choice for local setup.

## Installation
Create and activate a virtual environment first.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## Install PaddlePaddle
Install the backend first, then PaddleOCR.

### CPU example
```bash
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

For GPU, use the matching command from the official PaddlePaddle install guide for your CUDA version.

## Install app dependencies
```bash
pip install -r requirements.txt
```

## Run the app
```bash
python app.py
```

Then open:
```text
http://127.0.0.1:5000
```

## Notes
- The first OCR run may be slow because PaddleOCR may download model files.
- This demo currently uses `lang="en"` because Spotify screenshots are usually English-heavy.
- The parser is heuristic only. It does not yet call Spotify APIs or fuzzy-match OCR results.

## Suggested next upgrades
1. Add Spotify Search API lookup after OCR.
2. Add fuzzy matching for OCR correction.
3. Add cropping or region selection for playlist area only.
4. Save OCR results into SQLite for evaluation.
5. Compare PaddleOCR with EasyOCR for your dissertation experiments.
