"""Risk logic: turn a people count and a zone area into a risk level.

The model only counts people. The safety decision lives here, driven by
density (people per square metre) and thresholds from config/risk_thresholds.yaml.
"""
from collections import deque
from pathlib import Path

LEVELS = ["Low", "Medium", "High", "Critical"]
LEVEL_COLORS = {"Low": "#1D9E75", "Medium": "#FAC775", "High": "#EF9F27", "Critical": "#E24B4A"}

DEFAULT_THRESHOLDS = {"medium": 1.0, "high": 2.5, "critical": 4.0}
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "risk_thresholds.yaml"


def _validate(thresholds):
    m, h, c = thresholds["medium"], thresholds["high"], thresholds["critical"]
    if not (0 < m < h < c):
        raise ValueError("Thresholds must satisfy 0 < medium < high < critical, got %r" % (thresholds,))


def load_thresholds(path=None):
    """Load thresholds from YAML; fall back to defaults if the file or PyYAML is missing."""
    thresholds = dict(DEFAULT_THRESHOLDS)
    try:
        import yaml
        with open(Path(path) if path else _CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        thresholds.update({k: float(v) for k, v in data.get("thresholds", {}).items()})
    except (FileNotFoundError, ImportError):
        pass
    _validate(thresholds)
    return thresholds


def density_from_count(count, area_m2):
    """People per square metre. Negative counts are treated as zero."""
    if area_m2 is None or area_m2 <= 0:
        raise ValueError("area_m2 must be greater than zero")
    return max(float(count), 0.0) / float(area_m2)


def level_for_density(density, thresholds=None):
    t = thresholds or load_thresholds()
    if density >= t["critical"]:
        return "Critical"
    if density >= t["high"]:
        return "High"
    if density >= t["medium"]:
        return "Medium"
    return "Low"


def risk_level(count, area_m2, thresholds=None):
    """Return one of Low / Medium / High / Critical."""
    return level_for_density(density_from_count(count, area_m2), thresholds)


class RiskSmoother:
    """Stops alerts from flickering on video.

    Averages density over the last `window` frames, escalates after
    `confirm_up` consecutive frames at a higher level, and de-escalates only
    after `confirm_down` frames (slower, because a false 'all clear' is worse
    than a false alarm).
    """

    def __init__(self, window=5, confirm_up=2, confirm_down=4, thresholds=None):
        self.thresholds = thresholds or load_thresholds()
        self.densities = deque(maxlen=window)
        self.confirm_up = confirm_up
        self.confirm_down = confirm_down
        self.level = None
        self._pending = None
        self._pending_n = 0

    def update(self, count, area_m2):
        self.densities.append(density_from_count(count, area_m2))
        avg = sum(self.densities) / len(self.densities)
        candidate = level_for_density(avg, self.thresholds)
        if self.level is None:
            self.level = candidate
            return self.level
        if candidate == self.level:
            self._pending, self._pending_n = None, 0
            return self.level
        if candidate == self._pending:
            self._pending_n += 1
        else:
            self._pending, self._pending_n = candidate, 1
        going_up = LEVELS.index(candidate) > LEVELS.index(self.level)
        needed = self.confirm_up if going_up else self.confirm_down
        if self._pending_n >= needed:
            self.level = candidate
            self._pending, self._pending_n = None, 0
        return self.level
