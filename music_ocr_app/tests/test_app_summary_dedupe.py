import unittest

from app import _build_unique_summary


class SummaryDedupeTests(unittest.TestCase):
    def test_duplicate_current_track_is_kept_once(self):
        summary = _build_unique_summary(
            [
                {
                    "title": "PREACHER MAN",
                    "artist": "Ye, Kanye West",
                    "recommended": False,
                    "icon_score": 0.0,
                    "source_file": "queue_1.png",
                },
                {
                    "title": "preacher man",
                    "artist": " Ye,  Kanye West ",
                    "recommended": False,
                    "icon_score": 0.0,
                    "source_file": "queue_2.png",
                },
            ]
        )

        self.assertEqual(1, len(summary["all_tracks"]))
        self.assertEqual(0, len(summary["recommended_tracks"]))
        self.assertEqual(1, len(summary["organic_tracks"]))
        self.assertEqual("queue_1.png", summary["all_tracks"][0]["source_file"])

    def test_duplicate_track_is_promoted_if_any_copy_is_recommended(self):
        summary = _build_unique_summary(
            [
                {
                    "title": "Treat You Better",
                    "artist": "Shawn Mendes",
                    "recommended": False,
                    "icon_score": 0.0,
                    "source_file": "queue_1.png",
                },
                {
                    "title": "Treat You Better",
                    "artist": "Shawn Mendes",
                    "recommended": True,
                    "icon_score": 4.1,
                    "source_file": "queue_2.png",
                },
            ]
        )

        self.assertEqual(1, len(summary["all_tracks"]))
        self.assertEqual(1, len(summary["recommended_tracks"]))
        self.assertEqual(0, len(summary["organic_tracks"]))
        self.assertTrue(summary["all_tracks"][0]["recommended"])
        self.assertEqual(4.1, summary["all_tracks"][0]["icon_score"])


if __name__ == "__main__":
    unittest.main()
