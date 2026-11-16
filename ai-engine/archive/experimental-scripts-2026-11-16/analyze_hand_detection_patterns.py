
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("artifacts/features/INCLUDE")

rows = []

for file in ROOT.rglob("*.npz"):
    with np.load(file) as data:
        left = data["left_hand_detected"].astype(bool)
        right = data["right_hand_detected"].astype(bool)

        rows.append({
            "file": str(file),
            "frames": len(left),
            "left_rate": left.mean() * 100,
            "right_rate": right.mean() * 100,
            "left_missing": (~left).sum(),
            "right_missing": (~right).sum(),
            "left_completely_missing": not left.any(),
            "right_completely_missing": not right.any(),
        })

df = pd.DataFrame(rows)

print("\nHAND DETECTION PATTERN ANALYSIS")
print("=" * 55)

for hand in ["left", "right"]:
    rate = df[f"{hand}_rate"]

    print(f"\n{hand.upper()} HAND")
    print("Completely missing videos:", 
          df[f"{hand}_completely_missing"].sum())
    print("Videos with detection below 50%:", (rate < 50).sum())
    print("Videos with detection below 80%:", (rate < 80).sum())
    print("Videos with 100% detection:", (rate == 100).sum())
    print("Median detection rate:", round(rate.median(), 2), "%")

output = Path("data/manifests/validation/hand_detection_patterns.csv")
df.to_csv(output, index=False)

print("\nReport saved to:", output)
