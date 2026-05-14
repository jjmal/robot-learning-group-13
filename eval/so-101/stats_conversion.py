import json
import numpy as np

with open('../../model/checkpoints/stats.json') as f:
    lerobot_stats = json.load(f)

def convert_stats(s):
    mean = np.array(s["mean"])
    std = np.array(s["std"])
    min_val = np.array(s["min"])
    max_val = np.array(s["max"])
    
    # approximate clamp_min/clamp_max from percentiles
    p2 = np.array(s["q01"])   # closest to p2
    p98 = np.array(s["q99"])  # closest to p98
    
    clamp_min = mean + 1.5 * (p2 - mean)
    clamp_max = mean + 1.5 * (p98 - mean)
    
    clamped_min = np.maximum(min_val, clamp_min)
    clamped_max = np.minimum(max_val, clamp_max)
    
    # approximate mean_clamped and std_clamped
    mean_clamped = np.clip(mean, clamp_min, clamp_max)
    std_clamped = std  # approximation

    # percentiles — mimic-video needs 0-100 in 0.1 steps = 1001 values
    # approximate with linear interpolation between known quantiles
    percentiles = np.array([
        np.interp(
            np.linspace(0, 100, 1001),
            [0, 1, 10, 50, 90, 99, 100],
            [min_val[i], p2[i], np.array(s["q10"])[i], 
             np.array(s["q50"])[i], np.array(s["q90"])[i], p98[i], max_val[i]]
        )
        for i in range(len(mean))
    ]).T.tolist()  # shape [1001, D]

    return {
        "min": min_val.tolist(),
        "max": max_val.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "percentiles": percentiles,
        "clamp_min": clamp_min.tolist(),
        "clamp_max": clamp_max.tolist(),
        "mean_clamped": mean_clamped.tolist(),
        "std_clamped": std_clamped.tolist(),
    }

mimic_stats = {
    "action/actions": convert_stats(lerobot_stats["action"]),
    "obs/observation_state": convert_stats(lerobot_stats["observation.state"]),
}

with open('../../model/checkpoints/stats_mimic.json', 'w') as f:
    json.dump(mimic_stats, f)

print("Done.")