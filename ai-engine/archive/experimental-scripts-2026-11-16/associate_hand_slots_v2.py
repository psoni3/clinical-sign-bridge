
import numpy as np
from pathlib import Path

SOURCE = Path(
    "artifacts/mediapipe/standalone_experiment/MVI_8817.npz"
)

OUTPUT = Path(
    "artifacts/mediapipe/standalone_experiment/"
    "MVI_8817_associated_v2.npz"
)

MAX_GAP = 5
MAX_COST = 0.30
MIN_MARGIN = 0.005

data = np.load(SOURCE)

landmarks = data["landmarks"]
detected = data["hand_detected"]
handedness = data["handedness"]
confidence = data["confidence"]

T = len(detected)

tracked_landmarks = np.zeros((T, 2, 21, 3), dtype=np.float32)
tracked_detected = np.zeros((T, 2), dtype=bool)
track_ids = np.zeros((T, 2), dtype=np.int8)
association_cost = np.full((T, 2), np.nan, dtype=np.float32)
ambiguous = np.zeros(T, dtype=bool)

# Last known observation for each tracked identity.
last_landmarks = [None, None]
last_frame = [-1, -1]

def distance(a, b):
    return float(np.linalg.norm(a - b, axis=1).mean())

for frame in range(T):

    slots = np.where(detected[frame])[0].tolist()

    if not slots:
        continue

    # Initialize identities from the first frame containing two hands.
    if all(item is None for item in last_landmarks):

        if len(slots) != 2:
            continue

        for track, slot in enumerate(slots):
            tracked_landmarks[frame, track] = landmarks[frame, slot]
            tracked_detected[frame, track] = True
            track_ids[frame, track] = track + 1

            last_landmarks[track] = landmarks[frame, slot].copy()
            last_frame[track] = frame

        continue

    active_tracks = [
        i for i in range(2)
        if last_landmarks[i] is not None
        and frame - last_frame[i] <= MAX_GAP
    ]

    # Expire stale identities.
    for track in range(2):
        if last_landmarks[track] is not None:
            if frame - last_frame[track] > MAX_GAP:
                last_landmarks[track] = None
                last_frame[track] = -1

    active_tracks = [
        i for i in range(2) if last_landmarks[i] is not None
    ]

    if len(active_tracks) == 2 and len(slots) == 2:

        t0, t1 = active_tracks
        s0, s1 = slots

        same = (
            distance(last_landmarks[t0], landmarks[frame, s0])
            + distance(last_landmarks[t1], landmarks[frame, s1])
        )

        swapped = (
            distance(last_landmarks[t0], landmarks[frame, s1])
            + distance(last_landmarks[t1], landmarks[frame, s0])
        )

        best = min(same, swapped)
        margin = abs(same - swapped)

        if best > MAX_COST or margin < MIN_MARGIN:
            ambiguous[frame] = True
            continue

        assignment = (
            [(t0, s0), (t1, s1)]
            if same < swapped
            else [(t0, s1), (t1, s0)]
        )

    elif len(active_tracks) == 1 and len(slots) == 1:

        track = active_tracks[0]
        slot = slots[0]

        cost = distance(
            last_landmarks[track],
            landmarks[frame, slot]
        )

        if cost > MAX_COST:
            ambiguous[frame] = True
            continue

        assignment = [(track, slot)]

    else:
        # Avoid guessing when the number of observations and
        # available identities do not provide a clear assignment.
        ambiguous[frame] = True
        continue

    for track, slot in assignment:

        tracked_landmarks[frame, track] = landmarks[frame, slot]
        tracked_detected[frame, track] = True
        track_ids[frame, track] = track + 1

        association_cost[frame, track] = distance(
            last_landmarks[track],
            landmarks[frame, slot]
        )

        last_landmarks[track] = landmarks[frame, slot].copy()
        last_frame[track] = frame

np.savez_compressed(
    OUTPUT,
    tracked_landmarks=tracked_landmarks,
    tracked_detected=tracked_detected,
    track_ids=track_ids,
    association_cost=association_cost,
    ambiguous=ambiguous,
    original_handedness=handedness,
    original_confidence=confidence
)

print("\nASSOCIATION V2 COMPLETE")
print("=" * 55)
print("Output:", OUTPUT)
print("Total frames:", T)
print("Frames with tracked hands:",
      int(tracked_detected.any(axis=1).sum()))
print("Frames with two tracked hands:",
      int((tracked_detected.sum(axis=1) == 2).sum()))
print("Ambiguous frames:", int(ambiguous.sum()))
print("Source file unchanged.")
