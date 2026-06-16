import unittest

from ocr_service import clean_artist_text, clean_title_text, extract_music_candidates


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


class MultilingualTrackExtractionTests(unittest.TestCase):
    def test_cjk_artist_is_valid_artist_text(self):
        tracks = extract_music_candidates(
            [
                ocr_line("Say a lil something", 240, 100, 320, 50),
                ocr_line("🎊 萧敬腾", 265, 160, 130, 44),
            ]
        )

        self.assertEqual(
            [("Say a lil something", "萧敬腾")],
            [(track["title"], track["artist"]) for track in tracks],
        )

    def test_mixed_latin_and_cjk_artist_is_valid(self):
        tracks = extract_music_candidates(
            [
                ocr_line("Three Kingdom Love", 240, 100, 360, 50),
                ocr_line("TANK 吕建忠", 265, 160, 210, 44),
            ]
        )

        self.assertEqual(
            [("Three Kingdom Love", "TANK 吕建忠")],
            [(track["title"], track["artist"]) for track in tracks],
        )

    def test_icon_noise_prefixes_do_not_pollute_track_text(self):
        tracks = extract_music_candidates(
            [
                ocr_line("川", 165, 100, 22, 30),
                ocr_line("Higher (feat. Lukas Graham)", 220, 100, 430, 50),
                ocr_line("护", 165, 160, 22, 30),
                ocr_line("Pink Cafe, Brandon Beal, Lukas Graham", 220, 160, 560, 44),
            ]
        )

        self.assertEqual(
            [("Higher (feat. Lukas Graham)", "Pink Cafe, Brandon Beal, Lukas Graham")],
            [(track["title"], track["artist"]) for track in tracks],
        )

    def test_ui_icon_noise_affixes_are_cleaned_generically(self):
        cases = [
            (clean_artist_text, "护 中文歌手", "中文歌手"),
            (clean_artist_text, "🎊 Generic Artist", "Generic Artist"),
            (clean_artist_text, "目Mixed_Name123", "Mixed_Name123"),
            (clean_artist_text, "心 artist-name", "artist-name"),
            (clean_artist_text, "Y Generic Artist, Guest Artist", "Generic Artist, Guest Artist"),
            (clean_title_text, "川 Generic Title", "Generic Title"),
            (clean_title_text, "||| Generic Title", "Generic Title"),
        ]

        for cleaner, raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(expected, cleaner(raw))

    def test_ui_icon_noise_cleaning_preserves_real_content_prefixes(self):
        self.assertEqual("I Believe I Can Fly", clean_title_text("I Believe I Can Fly"))
        self.assertEqual("心跳", clean_title_text("心跳"))
        self.assertEqual("山丘", clean_title_text("山丘"))
        self.assertEqual("Y La Bamba", clean_artist_text("Y La Bamba"))
        self.assertEqual("A Great Big World", clean_artist_text("A Great Big World"))
        self.assertEqual("¥$, Example Artist", clean_artist_text("¥$, Example Artist"))

    def test_icon_noise_prefix_before_latin_artist_does_not_pollute_pair(self):
        tracks = extract_music_candidates(
            [
                ocr_line("LATATA", 220, 100, 180, 50),
                ocr_line("心 i-dle", 220, 160, 120, 44),
            ]
        )

        self.assertEqual(
            [("LATATA", "i-dle")],
            [(track["title"], track["artist"]) for track in tracks],
        )

    def test_badge_letter_prefix_before_artist_does_not_pollute_pair(self):
        tracks = extract_music_candidates(
            [
                ocr_line("Call Me When You Break Up (with Gr...", 220, 100, 580, 50),
                ocr_line("Y Generic Artist, Guest Artist", 220, 160, 420, 44),
            ]
        )

        self.assertEqual(
            [("Call Me When You Break Up (with Gr...", "Generic Artist, Guest Artist")],
            [(track["title"], track["artist"]) for track in tracks],
        )

    def test_two_character_cjk_titles_do_not_shift_pairs(self):
        tracks = extract_music_candidates(
            [
                ocr_line("Call Me Maybe", 220, 851, 311, 54),
                ocr_line("Carly Rae Jepsen", 220, 926, 294, 43),
                ocr_line("夜曲", 216, 1034, 101, 58),
                ocr_line("-", 1199, 1083, 29, 20),
                ocr_line("3ap Ghin", 83, 1112, 62, 28),
                ocr_line("周傑倫", 264, 1110, 120, 47),
                ocr_line("5", 110, 1143, 12, 8),
                ocr_line("和你", 215, 1220, 104, 58),
                ocr_line("余佳運", 216, 1294, 123, 51),
                ocr_line("春風吹", 216, 1404, 149, 58),
                ocr_line("方大同", 219, 1483, 116, 44),
                ocr_line("愛錯", 216, 1592, 102, 58),
                ocr_line("王力宏", 262, 1668, 122, 48),
                ocr_line("瘦子", 215, 1779, 101, 58),
                ocr_line("丁世光", 220, 1854, 119, 48),
                ocr_line("Count on Me", 220, 1970, 271, 48),
                ocr_line("Bruno Mars", 219, 2041, 199, 44),
            ]
        )

        pairs = [(track["title"], track["artist"]) for track in tracks]
        self.assertIn(("夜曲", "周傑倫"), pairs)
        self.assertIn(("愛錯", "王力宏"), pairs)
        self.assertIn(("瘦子", "丁世光"), pairs)
        self.assertIn(("Count on Me", "Bruno Mars"), pairs)
        self.assertNotIn(("余佳運", "春風吹"), pairs)
        self.assertNotIn(("丁世光", "Count on Me"), pairs)

    def test_two_character_cjk_artists_do_not_shift_pairs(self):
        tracks = extract_music_candidates(
            [
                ocr_line("Too Good At Goodbyes", 219, 1030, 495, 57),
                ocr_line("Sam Smith", 218, 1106, 190, 42),
                ocr_line("Stupid Pop Song", 216, 1214, 359, 64),
                ocr_line("陶喆", 217, 1289, 85, 50),
                ocr_line("一點點", 219, 1402, 146, 55),
                ocr_line("陶喆", 217, 1475, 84, 50),
                ocr_line("Gimme! Gimme! Gimme! (A Man After...", 218, 1588, 825, 55),
                ocr_line("ABBA", 217, 1664, 107, 43),
                ocr_line("和你", 215, 1773, 104, 57),
                ocr_line("余佳運", 217, 1849, 122, 48),
                ocr_line("Love Song", 217, 1959, 227, 61),
                ocr_line("方大同", 217, 2033, 122, 51),
            ]
        )

        pairs = [(track["title"], track["artist"]) for track in tracks]
        self.assertIn(("Stupid Pop Song", "陶喆"), pairs)
        self.assertIn(("一點點", "陶喆"), pairs)
        self.assertIn(("Love Song", "方大同"), pairs)
        self.assertNotIn(("Stupid Pop Song", "一點點"), pairs)
        self.assertNotIn(("陶喆", "Gimme! Gimme! Gimme! (A Man After..."), pairs)

    def test_short_stylized_latin_artist_does_not_shift_pairs(self):
        tracks = extract_music_candidates(
            [
                ocr_line("Always Wrong", 220, 1000, 300, 58),
                ocr_line("E Q-X", 220, 1075, 85, 44),
                ocr_line("S2", 135, 1145, 32, 24),
                ocr_line("La rencontre - Bande origin...", 220, 1184, 520, 58),
                ocr_line("0", 135, 1254, 16, 24),
                ocr_line("Francis Lai", 220, 1260, 210, 44),
                ocr_line("One Dance", 220, 1368, 230, 58),
                ocr_line("Drake", 220, 1440, 110, 44),
            ]
        )

        pairs = [(track["title"], track["artist"]) for track in tracks]
        self.assertIn(("Always Wrong", "Q-X"), pairs)
        self.assertIn(("La rencontre - Bande origin...", "Francis Lai"), pairs)
        self.assertIn(("One Dance", "Drake"), pairs)
        self.assertNotIn(("Always Wrong", "La rencontre - Bande origin..."), pairs)
        self.assertNotIn(("Francis Lai", "One Dance"), pairs)

    def test_orphan_artist_row_does_not_steal_next_title(self):
        tracks = extract_music_candidates(
            [
                ocr_line("Current Song", 220, 900, 300, 58),
                ocr_line("Current Artist", 220, 975, 240, 44),
                ocr_line("SingleName", 220, 1080, 180, 44),
                ocr_line("Long Title - Bande ori...", 220, 1184, 520, 58),
                ocr_line("Composer Name, Guest Name", 220, 1260, 360, 44),
                ocr_line("Next Track", 220, 1368, 230, 58),
                ocr_line("Next Artist", 220, 1440, 210, 44),
            ]
        )

        pairs = [(track["title"], track["artist"]) for track in tracks]
        self.assertIn(("Current Song", "Current Artist"), pairs)
        self.assertIn(("Long Title - Bande ori...", "Composer Name, Guest Name"), pairs)
        self.assertIn(("Next Track", "Next Artist"), pairs)
        self.assertNotIn(("SingleName", "Long Title - Bande ori..."), pairs)
        self.assertNotIn(("Composer Name, Guest Name", "Next Track"), pairs)

    def test_status_bar_and_recents_header_are_not_tracks(self):
        tracks = extract_music_candidates(
            [
                ocr_line("11:49", 170, 63, 143, 58),
                ocr_line("T", 1002, 85, 39, 29),
                ocr_line("96", 1070, 67, 82, 51),
                ocr_line("Recents", 551, 203, 179, 51),
                ocr_line("Music", 67, 347, 115, 51),
                ocr_line("Podcasts", 251, 350, 165, 46),
                ocr_line("Out of Time", 264, 532, 252, 52),
                ocr_line("The Weeknd", 263, 604, 224, 44),
                ocr_line("Chicago Freestyle (feat. Giveon)", 263, 717, 677, 57),
                ocr_line("Drake, GIVEON", 262, 786, 317, 51),
            ]
        )

        pairs = [(track["title"], track["artist"]) for track in tracks]
        self.assertNotIn(("T 96", "Recents"), pairs)
        self.assertIn(("Out of Time", "The Weeknd"), pairs)
        self.assertIn(("Chicago Freestyle (feat. Giveon)", "Drake, GIVEON"), pairs)


if __name__ == "__main__":
    unittest.main()
