from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from PIL import Image


TINY_NOISE_TOKENS = {
    "e", "il", "ll", "ii", "li", "l1", "1l", "i1", "1i", "日", "目", "口", "川", "心",
    "护", "m", "iii", "i", "111", "11", "1", "l", "|", "||", "|||", "ili", "ill",
    "lli", "n", "u", "w", "v", "y", "in", "im", "小", "山", "州",
}

OCR_ICON_NOISE_CHARS = "川日目口小山州护心"
OCR_BADGE_PREFIX_TOKENS = {"e", "y"}
ASCII_STROKE_NOISE_CHARS = "ilI|!1nmuwv"
DECORATIVE_NOISE_CHARS = ".-=…"
PRESERVED_PREFIX_CHARS = "&'$¥€£-"


class OCRService:
    """Wrapper around PaddleOCR with a stable app-facing API."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is not None:
            return self._ocr

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency `paddleocr`. Install requirements first."
            ) from exc

        init_attempts = [
            {
                "lang": self.lang,
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "PP-OCRv5_mobile_rec",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            },
            {
                "lang": self.lang,
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            },
            {"use_angle_cls": True, "lang": self.lang},
            {"lang": self.lang},
        ]

        last_error = None
        for kwargs in init_attempts:
            try:
                self._ocr = PaddleOCR(**kwargs)
                return self._ocr
            except Exception as exc:  # pragma: no cover
                last_error = exc

        raise RuntimeError(f"Could not initialize PaddleOCR: {last_error}")

    def read_text(self, image_path: str | Path) -> dict[str, Any]:
        image_path = str(image_path)
        ocr = self._get_ocr()

        result = None
        try:
            result = ocr.ocr(image_path, cls=True)
        except TypeError:
            result = ocr.ocr(image_path)
        except Exception:
            result = None

        lines = _flatten_paddle_result(result)

        if not lines and hasattr(ocr, "predict"):
            predict_attempts = [
                lambda: ocr.predict(image_path),
                lambda: ocr.predict([image_path]),
            ]
            for attempt in predict_attempts:
                try:
                    predict_result = attempt()
                    predict_lines = _flatten_paddle_result(predict_result)
                    if predict_lines:
                        result = predict_result
                        lines = predict_lines
                        break
                except Exception:
                    continue

        return {
            "lines": lines,
            "raw_text": "\n".join(item["text"] for item in lines),
            "raw_result": result,
        }


def _flatten_paddle_result(result: Any) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    seen: set[tuple[str, float | None, float | None]] = set()

    def append_line(text: Any, confidence: Any = None, bbox: Any = None) -> None:
        if text is None:
            return
        text_str = str(text).strip()
        if not text_str:
            return
        meta = _bbox_meta(bbox)
        key = (text_str, meta.get("x"), meta.get("y"))
        if key in seen:
            return
        seen.add(key)
        flattened.append(
            {
                "text": text_str,
                "confidence": round(float(confidence), 4) if isinstance(confidence, (int, float)) else None,
                "bbox": bbox,
                **meta,
            }
        )

    def try_extract_page_like(page_like: Any) -> bool:
        texts = None
        polys = None
        scores = None

        if isinstance(page_like, dict):
            texts = page_like.get("rec_texts", None)
            if texts is None:
                texts = page_like.get("texts", None)
            if texts is None:
                texts = page_like.get("text", None)

            polys = page_like.get("det_polys", None)
            if polys is None:
                polys = page_like.get("dt_polys", None)
            if polys is None:
                polys = page_like.get("polys", None)
            if polys is None:
                polys = page_like.get("boxes", None)

            scores = page_like.get("rec_scores", None)
            if scores is None:
                scores = page_like.get("scores", None)
            if scores is None:
                scores = page_like.get("score", None)
        else:
            for attr in ("rec_texts", "texts", "text"):
                if hasattr(page_like, attr):
                    candidate = getattr(page_like, attr)
                    if candidate is not None:
                        texts = candidate
                        break
            for attr in ("det_polys", "dt_polys", "polys", "boxes"):
                if hasattr(page_like, attr):
                    candidate = getattr(page_like, attr)
                    if candidate is not None:
                        polys = candidate
                        break
            for attr in ("rec_scores", "scores", "score"):
                if hasattr(page_like, attr):
                    candidate = getattr(page_like, attr)
                    if candidate is not None:
                        scores = candidate
                        break

        matched = False

        if isinstance(texts, list):
            for idx, text in enumerate(texts):
                bbox = polys[idx] if isinstance(polys, list) and idx < len(polys) else None
                confidence = scores[idx] if isinstance(scores, list) and idx < len(scores) else None
                append_line(text, confidence, bbox)
                matched = True
            return matched

        if texts is not None:
            append_line(texts, scores, polys)
            return True

        return matched

    def walk(node: Any) -> None:
        if node is None:
            return

        if try_extract_page_like(node):
            return

        if isinstance(node, dict):
            for value in node.values():
                walk(value)
            return

        if hasattr(node, "to_dict"):
            try:
                walk(node.to_dict())
                return
            except Exception:
                pass

        if hasattr(node, "__dict__"):
            try:
                node_dict = vars(node)
                if node_dict:
                    walk(node_dict)
                    return
            except Exception:
                pass

        if isinstance(node, (list, tuple)):
            if len(node) >= 2:
                first = node[0]
                second = node[1]
                if (
                    isinstance(first, (list, tuple))
                    and len(first) >= 4
                    and isinstance(second, (list, tuple))
                    and len(second) >= 1
                ):
                    text = second[0]
                    confidence = second[1] if len(second) > 1 else None
                    append_line(text, confidence, first)
                    return

            for item in node:
                walk(item)
            return

    walk(result)
    return flattened


def _bbox_meta(bbox: Any) -> dict[str, float | int | None]:
    if bbox is None:
        return {
            "x": None,
            "y": None,
            "w": None,
            "h": None,
            "x2": None,
            "y2": None,
            "cx": None,
            "cy": None,
        }

    if hasattr(bbox, "tolist"):
        try:
            bbox = bbox.tolist()
        except Exception:
            return {
                "x": None,
                "y": None,
                "w": None,
                "h": None,
                "x2": None,
                "y2": None,
                "cx": None,
                "cy": None,
            }

    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return {
            "x": None,
            "y": None,
            "w": None,
            "h": None,
            "x2": None,
            "y2": None,
            "cx": None,
            "cy": None,
        }

    xs = [float(pt[0]) for pt in bbox]
    ys = [float(pt[1]) for pt in bbox]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return {
        "x": x1,
        "y": y1,
        "w": x2 - x1,
        "h": y2 - y1,
        "x2": x2,
        "y2": y2,
        "cx": (x1 + x2) / 2,
        "cy": (y1 + y2) / 2,
    }


def _cluster_lines_by_row(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not lines:
        return []

    ordered = sorted(lines, key=lambda x: ((x.get("cy") or x.get("y") or 0), (x.get("x") or 0)))
    rows: list[list[dict[str, Any]]] = []

    for item in ordered:
        item_cy = item.get("cy") if item.get("cy") is not None else (item.get("y") or 0)
        item_h = item.get("h") or 0

        if not rows:
            rows.append([item])
            continue

        last_row = rows[-1]
        row_cys = [r.get("cy") if r.get("cy") is not None else (r.get("y") or 0) for r in last_row]
        row_hs = [r.get("h") or 0 for r in last_row]
        row_cy = sum(row_cys) / len(row_cys)
        
        row_h = min(row_hs) if row_hs else 0
        threshold = max(6, min(item_h, row_h) * 0.4)

        if abs(item_cy - row_cy) <= threshold:
            last_row.append(item)
        else:
            rows.append([item])

    return [sorted(row, key=lambda x: (x.get("x") or 0)) for row in rows]


def _row_text(row: list[dict[str, Any]]) -> str:
    merged = merge_same_row_fragments(sorted(row, key=lambda x: (x.get("x") or 0)))
    parts = []
    for item in merged:
        text = clean_ocr_text(item.get("text", ""))
        if text:
            parts.append(text)
    return normalize_text(" ".join(parts))


def _row_anchor(row: list[dict[str, Any]]) -> dict[str, Any]:
    merged = merge_same_row_fragments(sorted(row, key=lambda x: (x.get("x") or 0)))
    if not merged:
        return row[0]

    preferred: list[dict[str, Any]] = []
    
    for item in merged:
        text = clean_ocr_text(item.get("text", ""))
        compact_text = compact_content_text(text)
        if not text:
            continue
        if compact_text in TINY_NOISE_TOKENS:
            continue
        if len(compact_text) <= 2:
            continue
        preferred.append(item)

    anchor_source = preferred[0] if preferred else max(merged, key=lambda it: (it.get("w") or 0))
    source_group = preferred if preferred else merged

    xs = [it.get("x") for it in source_group if it.get("x") is not None]
    ys = [it.get("y") for it in source_group if it.get("y") is not None]
    x2s = [it.get("x2") for it in source_group if it.get("x2") is not None]
    y2s = [it.get("y2") for it in source_group if it.get("y2") is not None]

    anchor = dict(anchor_source)
    if xs:
        anchor["x"] = min(xs)
    if ys:
        anchor["y"] = min(ys)
    if x2s:
        anchor["x2"] = max(x2s)
    if y2s:
        anchor["y2"] = max(y2s)
    if anchor.get("x") is not None and anchor.get("x2") is not None:
        anchor["w"] = anchor["x2"] - anchor["x"]
        anchor["cx"] = (anchor["x"] + anchor["x2"]) / 2
    if anchor.get("y") is not None and anchor.get("y2") is not None:
        anchor["h"] = anchor["y2"] - anchor["y"]
        anchor["cy"] = (anchor["y"] + anchor["y2"]) / 2
    return anchor


def extract_music_candidates(
    lines: list[dict[str, Any]], image_path: str | Path | None = None
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in lines:
        text = clean_ocr_text(item.get("text", ""))
        if not text:
            continue
        normalized.append({**item, "text": text})

    normalized.sort(key=lambda x: ((x.get("y") or 0), (x.get("x") or 0)))
    xs = [l["x"] for l in normalized if l.get("x") is not None]
    if xs:
        median_x = sorted(xs)[len(xs) // 2]
        normalized = [l for l in normalized if (l.get("x") or 0) > median_x - 35]

    image = None
    if image_path:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = None

    rows = _cluster_lines_by_row(normalized)
    row_blocks: list[dict[str, Any]] = []
    for row in rows:
        text = _row_text(row)
        if not text:
            continue
        anchor = _row_anchor(row)
        row_blocks.append(
            {
                "text": text,
                "anchor": anchor,
                "x": anchor.get("x"),
                "y": anchor.get("y"),
                "y2": anchor.get("y2"),
                "cy": anchor.get("cy"),
                "h": anchor.get("h"),
                "w": anchor.get("w"),
            }
        )

    content_start_y = estimate_track_content_start_y(row_blocks)

    ui_tokens_exact = {
        "queue", "edit", "playing", "liked songs", "smart shuffle", "repeat", "timer",
        "队列", "编辑", "正在播放", "已点赞的歌曲", "循环播放", "定时器",
        "recents", "music", "podcasts",
    }

    candidate_xs = [
        rb["x"]
        for rb in row_blocks
        if rb.get("x") is not None
        and rb.get("text")
        and len(compact_content_text(rb["text"])) > 2
        and rb["text"].lower().strip() not in ui_tokens_exact
    ]
    text_column_x = sorted(candidate_xs)[len(candidate_xs) // 2] if candidate_xs else 0.0

    filtered_rows: list[dict[str, Any]] = []
    for rb in row_blocks:
        text = clean_ocr_text(rb["text"])
        low = text.lower().strip()
        compact = compact_content_text(low)
        x = rb.get("x") or 0.0
        w = rb.get("w") or 0.0
        h = rb.get("h") or 0.0

        if not text:
            continue
        if content_start_y is not None and content_start_y < 900 and (rb.get("y") or 0.0) < content_start_y - 90:
            continue
        if low in ui_tokens_exact:
            continue
        numeric_title_candidate = is_numeric_title_text(text) and compact not in TINY_NOISE_TOKENS

        if compact in TINY_NOISE_TOKENS:
            continue
        if is_noise(text) and not numeric_title_candidate:
            continue

        in_text_column = (
            text_column_x <= 0
            or x >= text_column_x - 28
            or (x + w) >= text_column_x - 6
        )
        if not in_text_column:
            continue

        looks_like_cover_art = (
            text_column_x > 0
            and x < text_column_x - 42
            and len(text.split()) <= 2
            and text.upper() == text
            and w < 95
            and h < 24
        )
        if looks_like_cover_art:
            continue

        filtered_rows.append({**rb, "text": text})

    pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    i = 0
    while i < len(filtered_rows) - 1:
        row1 = filtered_rows[i]
        title = clean_title_text(row1["text"])
        numeric_title = is_numeric_title_text(title)
        short_punctuated_title = is_short_punctuated_title_text(title)
        context_required_title = numeric_title or short_punctuated_title
        
        if not title:
            i += 1
            continue

        if not context_required_title and (is_noise(title) or not looks_like_title(title)):
            i += 1
            continue

        matched = False

        for j in range(1, min(6, len(filtered_rows) - i)):
            row2 = filtered_rows[i + j]
            artist = clean_artist_text(row2["text"])

            if not artist or is_noise(artist) or not looks_like_artist(artist):
                continue

            if (
                j == 1
                and i + j + 1 < len(filtered_rows)
                and rows_start_track_pair(row2, filtered_rows[i + j + 1], text_column_x)
            ):
                break

            score_title_anchor = {**row1["anchor"], "text": title}
            score_artist_anchor = {**row2["anchor"], "text": artist}
            
            score = pair_score(score_title_anchor, score_artist_anchor)
            
            if score <= 0:
                continue

            if context_required_title and not short_title_has_valid_context(
                score_title_anchor,
                score_artist_anchor,
                text_column_x,
            ):
                continue

            if score > 0:
                key = (title.lower(), artist.lower())
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    recommended = False
                    icon_score = 0.0
                    if image is not None:
                        recommended, icon_score = detect_recommendation_icon(
                            image,
                            row1["anchor"],
                            row2["anchor"],
                        )

                    pairs.append({
                        "title": title,
                        "artist": artist,
                        "source": "spotify-strict-two-line-layout",
                        "recommended": recommended,
                        "icon_score": round(icon_score, 3),
                        "title_y": row1["anchor"].get("y"),
                        "artist_y": row2["anchor"].get("y"),
                    })
                matched = True
                i = i + j + 1  
                break

        if not matched:
            i += 1

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in sorted(
        pairs,
        key=lambda x: (
            not x["recommended"],
            x.get("artist_y") if x.get("artist_y") is not None else math.inf,
        ),
    ):
        key = (item["title"].lower(), item["artist"].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped[:30]


def pair_score(title_line: dict[str, Any], artist_line: dict[str, Any]) -> float:
    tx = title_line.get("x") or 0
    ax = artist_line.get("x") or 0
    
    ty = title_line.get("y") or 0
    ay = artist_line.get("y") or 0
    th = max((title_line.get("h") or 18), 8)
    ah = max((artist_line.get("h") or 16), 8)

    vertical_gap = ay - ty
    if vertical_gap < -max(th, ah) * 0.5:
        return -999.0

    avg_h = (th + ah) / 2.0
    
    if vertical_gap < 0:
        vertical_gap = 0.1

    normalized_v_gap = vertical_gap / avg_h

    if normalized_v_gap > 3.5:
        return -999.0

    horizontal_gap = abs(tx - ax)
    normalized_h_gap = horizontal_gap / avg_h

    score = 100.0
    
    if normalized_v_gap > 2.0:
        score -= (normalized_v_gap - 2.0) * 30
        
    score -= normalized_h_gap * 4

    title = clean_title_text(title_line["text"])
    artist = clean_artist_text(artist_line["text"])
    title_uses_context = (
        is_numeric_title_text(title)
        or is_short_punctuated_title_text(title)
        or is_short_cjk_title_text(title)
    )
    title_clean = compact_content_text(title)
    artist_clean = compact_content_text(artist)
    artist_uses_context = is_short_cjk_text(artist) or is_short_stylized_latin_artist_text(artist)

    if (len(title_clean) <= 2 and not title_uses_context) or (len(artist_clean) <= 2 and not artist_uses_context):
        score -= 120
    if (not title_uses_context and re.fullmatch(r"[ilI|!1]+", title_clean)) or re.fullmatch(r"[ilI|!1]+", artist_clean):
        score -= 160

    if len(title.split()) > 15:
        score -= 20
    if len(artist.split()) > 14:
        score -= 20
    if any(noise in artist.lower() for noise in ["next from", "liked songs", "recently played", "queue", "playing"]):
        score -= 100
        
    if re.search(r"\b(and|feat|featuring)\b", artist.lower()) and "," not in artist.lower():
        score -= 20
    if re.search(r"\.\.\.|…", title):
        score += 8

    return score


def estimate_track_content_start_y(row_blocks: list[dict[str, Any]]) -> float | None:
    candidates: list[float] = []

    for current, following in zip(row_blocks, row_blocks[1:]):
        title = clean_title_text(current.get("text", ""))
        artist = clean_artist_text(following.get("text", ""))

        if not title or not artist:
            continue
        if is_noise(title) or is_noise(artist):
            continue
        if not looks_like_title(title) or not looks_like_artist(artist):
            continue

        cy = current.get("cy")
        next_cy = following.get("cy")
        if cy is None or next_cy is None:
            continue
        if next_cy <= cy:
            continue

        y = current.get("y")
        x = current.get("x")
        next_x = following.get("x")
        if y is None or x is None or next_x is None:
            continue

        if abs(x - next_x) <= 70:
            candidates.append(float(y))

    return min(candidates) if candidates else None


def rows_start_track_pair(
    title_row: dict[str, Any],
    artist_row: dict[str, Any],
    text_column_x: float,
) -> bool:
    title = clean_title_text(title_row.get("text", ""))
    artist = clean_artist_text(artist_row.get("text", ""))

    if not title or not artist:
        return False
    if is_noise(title) or is_noise(artist):
        return False
    if not looks_like_title(title) or not looks_like_artist(artist):
        return False

    title_anchor = {**title_row["anchor"], "text": title}
    artist_anchor = {**artist_row["anchor"], "text": artist}
    if pair_score(title_anchor, artist_anchor) <= 0:
        return False

    if (
        (is_numeric_title_text(title) or is_short_punctuated_title_text(title))
        and not short_title_has_valid_context(title_anchor, artist_anchor, text_column_x)
    ):
        return False

    return has_title_artist_row_shape(title_anchor, artist_anchor)


def has_title_artist_row_shape(title_line: dict[str, Any], artist_line: dict[str, Any]) -> bool:
    tx = title_line.get("x")
    ax = artist_line.get("x")
    ty = title_line.get("y")
    ay = artist_line.get("y")
    th = title_line.get("h") or 0.0
    ah = artist_line.get("h") or 0.0

    if tx is None or ax is None or ty is None or ay is None:
        return False
    if abs(tx - ax) > 70:
        return False
    if ay <= ty:
        return False
    if th <= 0 or ah <= 0:
        return False
    return th >= ah * 1.08 or th >= ah + 4


def is_numeric_title_text(text: str) -> bool:
    return re.fullmatch(r"\d+", clean_ocr_text(text)) is not None


def is_short_punctuated_title_text(text: str) -> bool:
    return re.fullmatch(r"[A-Za-z]\.", clean_ocr_text(text)) is not None


def is_short_cjk_title_text(text: str) -> bool:
    return is_short_cjk_text(text)


def is_short_cjk_text(text: str) -> bool:
    cleaned = clean_ocr_text(text)
    content = compact_content_text(cleaned)
    return 1 < len(content) <= 2 and any(is_cjk_char(char) for char in content)


def is_short_stylized_latin_artist_text(text: str) -> bool:
    cleaned = clean_ocr_text(text)
    compact = compact_content_text(cleaned)
    if not 2 <= len(compact) <= 3:
        return False
    if compact in TINY_NOISE_TOKENS:
        return False
    if sum(1 for char in cleaned if char.isalpha()) < 2:
        return False
    return re.fullmatch(r"[A-Za-z0-9]+(?:[._'&-][A-Za-z0-9]+)+", cleaned) is not None


def is_cjk_char(char: str) -> bool:
    return (
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
    )


def compact_content_text(text: str) -> str:
    return "".join(char.lower() for char in clean_ocr_text(text) if char.isalnum())


def has_letter(text: str) -> bool:
    return any(char.isalpha() for char in clean_ocr_text(text))


def looks_like_ui_time(text: str) -> bool:
    low = clean_ocr_text(text).lower()
    return re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*/\s*\d{1,2}:\d{2}(?::\d{2})?)?", low) is not None


def short_title_has_valid_context(
    title_line: dict[str, Any],
    artist_line: dict[str, Any],
    text_column_x: float,
) -> bool:
    title = clean_title_text(title_line.get("text", ""))
    artist = clean_artist_text(artist_line.get("text", ""))

    if not (is_numeric_title_text(title) or is_short_punctuated_title_text(title)):
        return False
    if not has_letter(artist):
        return False

    tx = title_line.get("x")
    ax = artist_line.get("x")
    ty = title_line.get("y")
    ay = artist_line.get("y")
    tw = title_line.get("w") or 0.0
    th = max((title_line.get("h") or 18), 8)
    ah = max((artist_line.get("h") or 16), 8)

    if tx is None or ax is None or ty is None or ay is None:
        return False

    if text_column_x > 0 and tx < text_column_x - 28 and tx + tw < text_column_x - 6:
        return False

    avg_h = (th + ah) / 2.0
    vertical_gap = ay - ty
    if vertical_gap <= max(2.0, avg_h * 0.25):
        return False
    if vertical_gap / avg_h > 2.8:
        return False

    horizontal_gap = ax - tx
    if horizontal_gap < -max(18.0, avg_h * 1.25):
        return False
    if horizontal_gap > max(120.0, avg_h * 2.4):
        return False

    return True


def detect_recommendation_icon(
    image: Image.Image,
    title_line: dict[str, Any],
    artist_line: dict[str, Any],
) -> tuple[bool, float]:
    x = artist_line.get("x")
    y = artist_line.get("y")
    h = artist_line.get("h")
    tx = title_line.get("x")  # 获取歌名的起始位置
    
    artist_cy = artist_line.get("cy") if artist_line.get("cy") is not None else (y + (h / 2 if h is not None else 0))

    if x is None or y is None or h is None or tx is None:
        return False, 0.0

    row_top = int(max(0, y - max(15, h * 0.6)))
    ay2 = artist_line.get("y2")
    row_bottom = int(min(image.height, (ay2 if ay2 is not None else (y + h)) + max(15, h * 0.6)))

    anchor_left = min(x, tx)
    left = int(max(0, anchor_left - max(8, h * 0.35)))
    right = int(min(image.width, x + max(20, h * 0.8)))

    if right <= left or row_bottom <= row_top:
        return False, 0.0

    crop = image.crop((left, row_top, right, row_bottom))
    width, height = crop.size
    pixels = list(crop.getdata())
    if not pixels or width <= 0 or height <= 0:
        return False, 0.0

    mask = [[False for _ in range(width)] for _ in range(height)]
    
    for yy in range(height):
        for xx in range(width):
            r, g, b = pixels[yy * width + xx]
            if g > 100 and g > r + 30 and g > b + 25:
                mask[yy][xx] = True

    visited = [[False for _ in range(width)] for _ in range(height)]
    best_score = 0.0

    for yy in range(height):
        for xx in range(width):
            if not mask[yy][xx] or visited[yy][xx]:
                continue

            stack = [(xx, yy)]
            visited[yy][xx] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cx, cy = stack.pop()
                coords.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] and not visited[ny][nx]:
                        visited[ny][nx] = True
                        stack.append((nx, ny))

            area = len(coords)
            if area < 15:  
                continue

            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            bw = x2 - x1 + 1
            bh = y2 - y1 + 1
            box_area = bw * bh
            fill_ratio = area / box_area if box_area else 0.0
            aspect = bw / bh if bh else 99.0

            if bw > max(50, int(width * 0.9)):
                continue
            if bh > max(50, int(height * 0.95)):
                continue
            if aspect > 2.5 or aspect < 0.4:
                continue

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            left_bias = 1.0 - min(center_x / max(width, 1), 1.0)
            target_y = max(0.0, min(float(height), float(artist_cy - row_top)))
            vertical_bias = 1.0 - min(abs(center_y - target_y) / max(height, 1), 1.0)

            score = (
                min(area / 60.0, 1.0) * 1.5
                + fill_ratio * 1.5
                + left_bias * 0.8
                + vertical_bias * 1.2
            )

            if score > best_score:
                best_score = score

    detected = best_score >= 2.0
    return detected, round(best_score, 3)


def is_noise(text: str) -> bool:
    low = text.lower().strip()
    skip_terms = {
        "search", "playlist", "playlists", "liked songs", "albums", "artists", 
        "podcasts", "shows", "download", "shuffle", "spotify", "home", "library", 
        "queue", "now playing", "next from: liked songs", "recently played", 
        "next from", "recommended songs", "your queue",
        "队列", "编辑", "正在播放", "已点赞的歌曲", "循环播放", "定时器",
    }
    if low in skip_terms:
        return True
    if looks_like_ui_time(low):
        return True
    if re.fullmatch(r"\d+", low):
        return True
    if len(low) <= 1:
        return True
    if low.startswith("next from"):
        return True
        
    compact = compact_content_text(low)
    if compact in TINY_NOISE_TOKENS:
        return True
    if re.fullmatch(r"[il1|!I]+", compact):
        return True
    return False


def looks_like_title(text: str) -> bool:
    low = text.lower().strip()
    compact = compact_content_text(low)
    if is_noise(text):
        return False
    if len(text.split()) > 24:
        return False
    if len(compact) <= 1:
        return False
    if re.fullmatch(r"[il1|!]+", compact):
        return False
    return True


def looks_like_artist(text: str) -> bool:
    low = text.lower().strip()
    compact = compact_content_text(low)
    if is_noise(text):
        return False
    if len(text.split()) > 24:
        return False
    if len(compact) <= 1:
        return False
    if re.fullmatch(r"[il1|!]+", compact):
        return False
    if any(token in low for token in ["feat", "featuring", "album", "playlist", "queue"]):
        return False
    return True


def clean_ocr_text(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+&", " &", text)
    return text.strip()


def clean_title_text(text: str) -> str:
    text = strip_ui_noise_affixes(text, strip_explicit_badge=False)
    return text.strip(" ,-=")


def clean_artist_text(text: str) -> str:
    text = strip_ui_noise_affixes(text, strip_explicit_badge=True)
    text = strip_leading_badge_token_noise(text)
    text = re.sub(r"\s+,\s*", ", ", text)
    return text.strip(" ,-=")


def strip_ui_noise_affixes(text: str, *, strip_explicit_badge: bool) -> str:
    text = clean_ocr_text(text)
    if strip_explicit_badge:
        text = strip_explicit_badge_prefix(text)

    for _ in range(6):
        previous = text
        text = strip_leading_icon_noise(text)
        text = strip_leading_ascii_noise_token(text)
        text = strip_trailing_icon_noise(text)
        text = strip_trailing_ascii_noise_token(text)
        text = clean_ocr_text(text)
        if text == previous:
            break

    return text


def strip_explicit_badge_prefix(text: str) -> str:
    text = re.sub(r"^E\s+(?=\S)", "", text)
    return re.sub(r"^E(?=[A-Z][a-z])", "", text)


def strip_leading_badge_token_noise(text: str) -> str:
    match = re.match(r"^([A-Za-z])\s+(.+)$", text)
    if not match:
        return text

    token, rest = match.groups()
    if token.lower() not in OCR_BADGE_PREFIX_TOKENS:
        return text
    if not looks_like_badge_prefixed_artist(rest):
        return text
    return rest


def looks_like_badge_prefixed_artist(text: str) -> bool:
    compact = compact_content_text(text)
    if len(compact) <= 2:
        return False
    if not has_letter(text):
        return False
    if "," in text:
        return True
    return bool(re.match(r"^[A-Z][A-Za-z'._-]{2,}\s+[A-Z][A-Za-z'._-]{2,}", text))


def strip_leading_icon_noise(text: str) -> str:
    preserved = re.escape(PRESERVED_PREFIX_CHARS)
    text = re.sub(rf"^[^\w{preserved}]{{1,3}}\s+(?=\w)", "", text)
    text = re.sub(rf"^[^\w{preserved}]{{1,2}}(?=[A-Z][a-z])", "", text)
    text = re.sub(rf"^[{OCR_ICON_NOISE_CHARS}]\s+(?=\w)", "", text)
    return re.sub(rf"^[{OCR_ICON_NOISE_CHARS}](?=[A-Za-z0-9_])", "", text)


def strip_trailing_icon_noise(text: str) -> str:
    text = re.sub(rf"\s+[{OCR_ICON_NOISE_CHARS}]$", "", text)
    return re.sub(rf"(?<=[A-Za-z0-9_])[{OCR_ICON_NOISE_CHARS}]$", "", text)


def strip_leading_ascii_noise_token(text: str) -> str:
    noise_chars = re.escape(ASCII_STROKE_NOISE_CHARS + DECORATIVE_NOISE_CHARS)
    match = re.match(rf"^([{noise_chars}]+)\s+(?=\w)", text)
    if match and is_stripable_ascii_noise_token(match.group(1)):
        return text[match.end():]
    return text


def strip_trailing_ascii_noise_token(text: str) -> str:
    noise_chars = re.escape(ASCII_STROKE_NOISE_CHARS + DECORATIVE_NOISE_CHARS)
    match = re.search(rf"\s+([{noise_chars}]+)$", text)
    if match and is_stripable_ascii_noise_token(match.group(1)):
        return text[:match.start()]
    return text


def is_stripable_ascii_noise_token(token: str) -> bool:
    if len(token) > 1:
        return True
    return not token.isalpha()


def merge_same_row_fragments(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not lines:
        return []

    merged: list[dict[str, Any]] = []
    current = dict(lines[0])

    def compact_fragment(value: str) -> str:
        return compact_content_text(value)

    def should_merge(left_item: dict[str, Any], right_item: dict[str, Any]) -> bool:
        left_text = left_item.get("text", "")
        right_text = right_item.get("text", "")
        left_compact = compact_fragment(left_text)
        right_compact = compact_fragment(right_text)

        if not left_text or not right_text:
            return False
        
        if right_compact in TINY_NOISE_TOKENS or left_compact in TINY_NOISE_TOKENS:
            return False
        return True

    for item in lines[1:]:
        current_cy = current.get("cy") if current.get("cy") is not None else current.get("y", 0)
        item_cy = item.get("cy") if item.get("cy") is not None else item.get("y", 0)
        current_h = current.get("h") or 0
        item_h = item.get("h") or 0
        current_x2 = current.get("x2") if current.get("x2") is not None else (current.get("x") or 0)
        item_x = item.get("x") or 0
        gap = item_x - current_x2

        same_row = abs(item_cy - current_cy) <= max(8, max(current_h, item_h) * 0.45)
        close_horizontally = -6 <= gap <= max(26, max(current_h, item_h) * 1.4)

        if same_row and close_horizontally and should_merge(current, item):
            left_text = current.get("text", "")
            right_text = item.get("text", "")
            if left_text.endswith(("-", "/")):
                combined_text = f"{left_text}{right_text}"
            elif right_text.startswith((",", ".", ";", ":")):
                combined_text = f"{left_text}{right_text}"
            else:
                combined_text = f"{left_text} {right_text}"

            current["text"] = normalize_text(combined_text)
            current["x"] = min(v for v in [current.get("x"), item.get("x")] if v is not None)
            current["y"] = min(v for v in [current.get("y"), item.get("y")] if v is not None)
            current["x2"] = max(v for v in [current.get("x2"), item.get("x2"), current.get("x"), item.get("x")] if v is not None)
            current["y2"] = max(v for v in [current.get("y2"), item.get("y2"), current.get("y"), item.get("y")] if v is not None)
            current["w"] = current["x2"] - current["x"] if current.get("x2") is not None and current.get("x") is not None else current.get("w")
            current["h"] = current["y2"] - current["y"] if current.get("y2") is not None and current.get("y") is not None else current.get("h")
            current["cx"] = (current["x"] + current["x2"]) / 2 if current.get("x") is not None and current.get("x2") is not None else current.get("cx")
            current["cy"] = (current["y"] + current["y2"]) / 2 if current.get("y") is not None and current.get("y2") is not None else current.get("cy")
            continue

        merged.append(current)
        current = dict(item)

    merged.append(current)
    return merged


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
