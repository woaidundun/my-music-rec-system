import unittest

from PIL import Image, ImageDraw

from ocr_service import detect_recommendation_icon


def line_anchor(text, x, y, w=120, h=50):
    return {
        "text": text,
        "x": float(x),
        "y": float(y),
        "w": float(w),
        "h": float(h),
        "x2": float(x + w),
        "y2": float(y + h),
        "cx": float(x + w / 2),
        "cy": float(y + h / 2),
    }


class RecommendationIconDetectionTests(unittest.TestCase):
    def test_album_art_edge_is_not_recommendation_icon(self):
        image = Image.new("RGB", (360, 280), (18, 18, 18))
        draw = ImageDraw.Draw(image)
        draw.rectangle((176, 150, 183, 220), fill=(0, 190, 120))

        detected, score = detect_recommendation_icon(
            image,
            line_anchor("Love Song", 217, 100, 220, 61),
            line_anchor("方大同", 217, 174, 120, 51),
        )

        self.assertFalse(detected)
        self.assertEqual(0.0, score)

    def test_green_icon_near_text_column_is_detected(self):
        image = Image.new("RGB", (360, 280), (18, 18, 18))
        draw = ImageDraw.Draw(image)
        draw.rectangle((220, 186, 238, 204), fill=(30, 215, 95))

        detected, score = detect_recommendation_icon(
            image,
            line_anchor("Stitches", 218, 100, 180, 50),
            line_anchor("Shawn Mendes", 265, 172, 240, 43),
        )

        self.assertTrue(detected)
        self.assertGreaterEqual(score, 2.0)


if __name__ == "__main__":
    unittest.main()
