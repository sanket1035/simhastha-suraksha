import unittest

from src.risk import (RiskSmoother, density_from_count, level_for_density, load_thresholds, risk_level)


class RiskTests(unittest.TestCase):
    def test_worked_example_600_people_200m2_is_high(self):
        self.assertAlmostEqual(density_from_count(600, 200), 3.0)
        self.assertEqual(risk_level(600, 200), "High")

    def test_boundaries_on_100_m2(self):
        cases = [(99, "Low"), (100, "Medium"), (249, "Medium"), (250, "High"), (399, "High"), (400, "Critical")]
        for count, expected in cases:
            self.assertEqual(risk_level(count, 100), expected, "count=%d" % count)

    def test_zero_and_negative_counts_are_low(self):
        self.assertEqual(risk_level(0, 50), "Low")
        self.assertEqual(risk_level(-5, 50), "Low")

    def test_invalid_area_raises(self):
        for bad in (0, -10, None):
            with self.assertRaises(ValueError):
                density_from_count(10, bad)

    def test_default_thresholds_load_and_are_ordered(self):
        t = load_thresholds()
        self.assertLess(t["medium"], t["high"])
        self.assertLess(t["high"], t["critical"])

    def test_bad_thresholds_rejected(self):
        import tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "t.yaml")
        with open(path, "w") as f:
            f.write("thresholds:\n  medium: 3\n  high: 2\n  critical: 4\n")
        with self.assertRaises(ValueError):
            load_thresholds(path)

    def test_custom_thresholds_used(self):
        t = {"medium": 0.5, "high": 1.0, "critical": 2.0}
        self.assertEqual(level_for_density(1.5, t), "High")


class SmootherTests(unittest.TestCase):
    def test_first_update_sets_level(self):
        s = RiskSmoother(window=1)
        self.assertEqual(s.update(50, 100), "Low")

    def test_single_spike_does_not_change_level(self):
        s = RiskSmoother(window=1, confirm_up=2)
        s.update(50, 100)
        self.assertEqual(s.update(500, 100), "Low")
        self.assertEqual(s.update(50, 100), "Low")

    def test_escalates_after_confirmation(self):
        s = RiskSmoother(window=1, confirm_up=2)
        s.update(50, 100)
        s.update(500, 100)
        self.assertEqual(s.update(500, 100), "Critical")

    def test_deescalates_slower_than_it_escalates(self):
        s = RiskSmoother(window=1, confirm_up=2, confirm_down=4)
        s.update(500, 100)
        levels = [s.update(50, 100) for _ in range(4)]
        self.assertEqual(levels[:3], ["Critical"] * 3)
        self.assertEqual(levels[3], "Low")


if __name__ == "__main__":
    unittest.main()
