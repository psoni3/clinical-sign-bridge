
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("artifacts/features/INCLUDE")


def get_missing_runs(mask):
    """Return lengths of consecutive missing-detection runs."""
    missing = ~mask.astype(bool)
    padded = np.concatenate(([False], missing, [False]))
    changes = np.diff(padded.astype(int))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    return (ends - starts).tolist()


rows = []

for file in ROOT.rglob("*.npz"):
    with np.load(file) as data:
        left = data["left_hand_detected"].astype(bool)
        right = data["right_hand_detected"].astype(bool)

        left_runs = get_missing_runs(left)
        right_runs = get_missing_runs(right)

        rows.append({
            "file": str(file),
            "label": str(data["label"].item()),
            "left_detection": left.mean() * 100,
            "right_detection": right.mean() * 100,
            "left_longest_gap": max(left_runs, default=0),
            "right_longest_gap": max(right_runs, default=0),
            "left_gap_count": len(left_runs),
            "right_gap_count": len(right_runs),
        })

df = pd.DataFrame(rows)

print("\nHAND DETECTION GAP ANALYSIS")
print("=" * 60)

for hand in ["left", "right"]:
    print(f"\n{hand.upper()} HAND")
    print("Videos with longest missing gap > 5 frames:",
          (df[f"{hand}_longest_gap"] > 5).sum())
    print("Videos with longest missing gap > 15 frames:",
          (df[f"{hand}_longest_gap"] > 15).sum())
    print("Median longest missing gap:",
          df[f"{hand}_longest_gap"].median(), "frames")

    print("\n10 videos with worst detection:")
    worst = df.nsmallest(10, f"{hand}_detection")

    for _, row in worst.iterrows():
        print(
            f"{row['label']} | "
            f"Detection: {row[f'{hand}_detection']:.1f}% | "
            f"Longest gap: {row[f'{hand}_longest_gap']} frames | "
            f"{row['file']}"
        )

output = Path(
    "data/manifests/validation/hand_detection_gaps.csv"
)
df.to_csv(output, index=False)

print("\nDetailed report saved to:", output)
