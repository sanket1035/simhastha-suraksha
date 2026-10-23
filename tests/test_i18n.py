import unittest

from src.i18n import ADVICE, LANGS, LEVEL_FULL, LEVEL_SHORT, TEXT, zone_name

LEVELS = ["Low", "Medium", "High", "Critical"]


class TestI18n(unittest.TestCase):
    def test_every_language_has_every_string(self):
        for code in LANGS.values():
            self.assertEqual(set(TEXT[code]), set(TEXT["en"]), code)
            self.assertEqual(set(TEXT[code]["views"]), set(TEXT["en"]["views"]), code)
            for table in (ADVICE, LEVEL_FULL, LEVEL_SHORT):
                self.assertEqual(set(table[code]), set(LEVELS), code)

    def test_zone_names(self):
        self.assertEqual(zone_name("Ramkund", "mr"), "रामकुंड")
        self.assertEqual(zone_name("Ramkund", "en"), "Ramkund")


if __name__ == "__main__":
    unittest.main()
