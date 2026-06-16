import unittest

from ocr_service import extract_music_candidates


def ocr_line(text, x, y, w=None, h=16):
    if w is None:
        w = max(8, len(text) * 8)
    return {
        "text": text,
        "confidence": 0.99,
        "bbox": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        "x": float(x),
        "y": float(y),
        "w": float(w),
        "h": float(h),
        "x2": float(x + w),
        "y2": float(y + h),
        "cx": float(x + w / 2),
        "cy": float(y + h / 2),
    }


class NumericTitleExtractionTests(unittest.TestCase):
    def test_charli_xcx_numeric_title_is_recognized(self):
        tracks = extract_music_candidates(
            [
                ocr_line("360", 120, 100, 28),
                ocr_line("Charli XCX", 120, 122, 76, 14),
            ]
        )

        self.assertIn(
            ("360", "Charli XCX"),
            {(track["title"], track["artist"]) for track in tracks},
        )

    def test_other_numeric_titles_work_in_track_context(self):
        cases = [
            ("1999", "Prince"),
            ("7", "Catfish and the Bottlemen"),
            ("505", "Arctic Monkeys"),
        ]

        for title, artist in cases:
            with self.subTest(title=title):
                tracks = extract_music_candidates(
                    [
                        ocr_line(title, 120, 100, max(8, len(title) * 10)),
                        ocr_line(artist, 120, 122, len(artist) * 8, 14),
                    ]
                )
                self.assertIn(
                    (title, artist),
                    {(track["title"], track["artist"]) for track in tracks},
                )

    def test_regular_english_title_still_works(self):
        tracks = extract_music_candidates(
            [
                ocr_line("Sweet Disposition", 120, 100, 128),
                ocr_line("The Temper Trap", 120, 122, 112, 14),
            ]
        )

        self.assertEqual(
            [("Sweet Disposition", "The Temper Trap")],
            [(track["title"], track["artist"]) for track in tracks],
        )

    def test_ui_numbers_and_progress_are_not_tracks(self):
        tracks = extract_music_candidates(
            [
                ocr_line("0:32", 120, 50, 34, 12),
                ocr_line("Dreams", 120, 82, 48),
                ocr_line("Fleetwood Mac", 120, 104, 104, 14),
                ocr_line("1", 42, 150, 8, 14),
                ocr_line("Songbird", 120, 150, 64),
                ocr_line("Oasis", 120, 172, 40, 14),
                ocr_line("75", 300, 218, 20, 12),
            ]
        )

        pairs = [(track["title"], track["artist"]) for track in tracks]
        self.assertIn(("Dreams", "Fleetwood Mac"), pairs)
        self.assertIn(("Songbird", "Oasis"), pairs)
        self.assertNotIn(("0:32", "Dreams"), pairs)
        self.assertFalse(any(title in {"1", "75"} for title, _artist in pairs))

    def test_numeric_title_with_symbol_artist_and_explicit_badge(self):
        tracks = extract_music_candidates(
            [
                ocr_line("530", 226, 1128, 93, 53),
                ocr_line("E", 269, 1208, 42, 34),
                ocr_line("¥$, Kanye West, Ty Dolla $ign", 304, 1200, 508, 51),
                ocr_line("wrong faces.", 226, 1320, 275, 51),
                ocr_line("Brent Faiyaz", 260, 1389, 219, 46),
            ]
        )

        pairs = [(track["title"], track["artist"]) for track in tracks]
        self.assertIn(("530", "¥$, Kanye West, Ty Dolla $ign"), pairs)
        self.assertIn(("wrong faces.", "Brent Faiyaz"), pairs)
        self.assertNotIn(("E ¥$, Kanye West, Ty Dolla $ign", "wrong faces."), pairs)

    def test_short_punctuated_title_is_recognized_in_track_context(self):
        tracks = extract_music_candidates(
            [
                ocr_line("K.", 240, 1905, 52, 50),
                ocr_line("Cigarettes After Sex", 265, 1966, 346, 47),
                ocr_line("Fall in Love with You.", 240, 2091, 431, 51),
                ocr_line("Montell Fish", 264, 2152, 212, 44),
            ]
        )

        pairs = [(track["title"], track["artist"]) for track in tracks]
        self.assertIn(("K.", "Cigarettes After Sex"), pairs)
        self.assertIn(("Fall in Love with You.", "Montell Fish"), pairs)
        self.assertNotIn(("Cigarettes After Sex", "Fall in Love with You."), pairs)

    def test_unpunctuated_single_letter_badge_is_not_a_title(self):
        tracks = extract_music_candidates(
            [
                ocr_line("E", 240, 100, 24, 24),
                ocr_line("Cigarettes After Sex", 265, 160, 346, 47),
            ]
        )

        pairs = [(track["title"], track["artist"]) for track in tracks]
        self.assertNotIn(("E", "Cigarettes After Sex"), pairs)
        self.assertEqual([], pairs)


if __name__ == "__main__":
    unittest.main()
