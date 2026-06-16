from __future__ import annotations
from pathlib import Path
from uuid import uuid4
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ocr_service import OCRService, extract_music_candidates

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["TEMPLATES_AUTO_RELOAD"] = True

ocr_service = OCRService(lang="en")

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _track_key(track: dict) -> tuple[str, str]:
    title = " ".join(str(track.get("title", "")).casefold().split())
    artist = " ".join(str(track.get("artist", "")).casefold().split())
    return title, artist


def _dedupe_tracks_for_export(tracks: list[dict]) -> list[dict]:
    unique: dict[tuple[str, str], dict] = {}

    for track in tracks:
        key = _track_key(track)
        if not any(key):
            continue

        if key not in unique:
            unique[key] = dict(track)
            continue

        existing = unique[key]
        if track.get("recommended") and not existing.get("recommended"):
            existing["recommended"] = True
            existing["icon_score"] = track.get("icon_score", existing.get("icon_score", 0.0))
        elif track.get("recommended") and existing.get("recommended"):
            existing["icon_score"] = max(
                float(existing.get("icon_score") or 0.0),
                float(track.get("icon_score") or 0.0),
            )

    return list(unique.values())


def _build_unique_summary(tracks: list[dict]) -> dict[str, list[dict]]:
    unique_tracks = _dedupe_tracks_for_export(tracks)
    return {
        "all_tracks": unique_tracks,
        "recommended_tracks": [item for item in unique_tracks if item.get("recommended")],
        "organic_tracks": [item for item in unique_tracks if not item.get("recommended")],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "error": None,
        "results": [],
        "summary": {
            "all_tracks": [],
            "recommended_tracks": [],
            "organic_tracks": []
        }
    }

    if request.method == "POST":
        uploaded_files = request.files.getlist("images")
        
        if not uploaded_files or uploaded_files[0].filename == "":
            context["error"] = "Please choose at least one image."
            return render_template("index.html", **context)

        for uploaded in uploaded_files:
            if not allowed_file(uploaded.filename):
                continue

            filename = secure_filename(uploaded.filename)
            unique_name = f"{uuid4().hex}_{filename}"
            save_path = UPLOAD_DIR / unique_name
            uploaded.save(save_path)

            try:
                ocr_output = ocr_service.read_text(save_path)
                all_lines = ocr_output["lines"]
                all_candidates = extract_music_candidates(all_lines, save_path)
                
                # 为每首歌打上来源标签，方便导出表格
                for item in all_candidates:
                    item['source_file'] = filename
                
                rec_tracks = [item for item in all_candidates if item.get("recommended")]
                org_tracks = [item for item in all_candidates if not item.get("recommended")]

                # 记录单张图片结果
                context["results"].append({
                    "original_filename": filename,
                    "image_url": f"/uploads/{unique_name}",
                    "all_candidates": all_candidates,
                    "recommended_candidates": rec_tracks,
                    "raw_text": ocr_output["raw_text"],
                })

                # 更新全量汇总数据
                context["summary"]["all_tracks"].extend(all_candidates)
                context["summary"]["recommended_tracks"].extend(rec_tracks)
                context["summary"]["organic_tracks"].extend(org_tracks)

            except Exception as exc:
                context["error"] = f"OCR Error: {exc}"

        context["summary"] = _build_unique_summary(context["summary"]["all_tracks"])

    return render_template("index.html", **context)

@app.route("/uploads/<path:filename>")
def uploads(filename: str):
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
