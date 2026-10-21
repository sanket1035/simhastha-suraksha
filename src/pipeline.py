"""Frames in, risk history and alerts out. No UI code, so it is easy to test."""
from .predict import predict
from .risk import RiskSmoother, density_from_count, load_thresholds


def analyze_frames(frames, area_m2, thresholds=None, predict_fn=predict, smoother=None):
    """Run the full flow on (time_s, frame_rgb) pairs for ONE zone.

    Returns a dict with:
      history          list of {time_s, count, density, level} per frame
      alerts           list of {time_s, from, to} whenever the smoothed level changes
      last_frame       last RGB frame (or None)
      last_density_map last predicted density map (or None)
    """
    thresholds = thresholds or load_thresholds()
    smoother = smoother or RiskSmoother(thresholds=thresholds)
    history, alerts = [], []
    last_frame = last_map = None
    prev_level = None
    for t, frame in frames:
        count, dens_map = predict_fn(frame)
        level = smoother.update(count, area_m2)
        if prev_level is not None and level != prev_level:
            alerts.append({"time_s": round(float(t), 1), "from": prev_level, "to": level})
        prev_level = level
        history.append({"time_s": round(float(t), 1), "count": float(count),
                        "density": density_from_count(count, area_m2), "level": level})
        last_frame, last_map = frame, dens_map
    return {"history": history, "alerts": alerts, "last_frame": last_frame, "last_density_map": last_map}
