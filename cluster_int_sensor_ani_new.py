import os
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from multiprocessing import Pool
from decord import VideoReader, cpu, gpu
import subprocess

# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Overlay sensor data on video")

parser.add_argument("--session_id", type=int, required=True, help="Session ID (e.g., 386)")
parser.add_argument("--video_file", type=str, required=True, help="Path to video file")

# Optional overrides (you can extend later if needed)
parser.add_argument("--metadata_csv", type=str, default="./markers/session1_complete_final_markers.csv")
parser.add_argument("--green_dir", type=str, default="./sensors/D1/Green")
parser.add_argument("--blue_dir", type=str, default="./sensors/D1/Blue")
parser.add_argument("--out_video", type=str, default=None)

args = parser.parse_args()

session_id = args.session_id
video_file = args.video_file
metadata_csv = args.metadata_csv
green_dir = Path(args.green_dir)
blue_dir  = Path(args.blue_dir)

OUT_VIDEO = args.out_video if args.out_video is not None else f"session_{session_id}_stable.mkv"
# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

# session_id = 383
# metadata_csv = "./markers/session1_complete_final_markers.csv"
# video_file = "383_all_cams_anonymized.mp4"
# green_dir = Path("./sensors/D1/Green")
# blue_dir  = Path("./sensors/D1/Blue")

# <<< SET YOUR SENSOR COLUMNS HERE >>>
X_COL = "accelerometerAccelerationX(G)"
Y_COL = "accelerometerAccelerationY(G)"
Z_COL = "accelerometerAccelerationZ(G)"

WINDOW_SEC = 5.0
SENSOR_LAG_SEC = 1.5

CHUNK_SIZE = 1024
BATCH_PER_WORKER = 256

# OUT_VIDEO = f"session_{session_id}_stable.mp4"

# ------------------------------------------------------------
# CPU COUNT
# ------------------------------------------------------------
N_WORKERS = int(
    os.environ.get("SLURM_CPUS_PER_TASK",
    os.environ.get("OMP_NUM_THREADS",
    os.cpu_count()))
)
print(f"[CPU] Using {N_WORKERS} workers")

cv2.setNumThreads(0)

# ------------------------------------------------------------
# VIDEO READER
# ------------------------------------------------------------
def get_video_reader(path):
    if torch.cuda.is_available():
        try:
            return VideoReader(path, ctx=gpu(0))
        except Exception:
            pass
    return VideoReader(path, ctx=cpu(0))

vr = get_video_reader(video_file)
fps = vr.get_avg_fps()
frame_count = len(vr)
print(f"[VIDEO] FPS={fps:.2f} | Frames={frame_count}")

# ------------------------------------------------------------
# SESSION METADATA
# ------------------------------------------------------------
session_df = pd.read_csv(metadata_csv, names=["id", "event", "time"], header=0)

def get_bounds(df, sid):
    s = df[df.event == f"{sid}_start"]["time"]
    e = df[df.event == f"{sid}_end"]["time"]
    if len(s) == 0 or len(e) == 0:
        raise RuntimeError("Missing session markers")
    return float(s.iloc[0]), float(e.iloc[0])

sess_start, sess_end = get_bounds(session_df, session_id)

video_start = pd.to_datetime("2023-11-27T12:15:46Z", utc=True)
session_start = video_start + pd.to_timedelta(sess_start, unit="s")
session_end   = video_start + pd.to_timedelta(sess_end, unit="s")

# ------------------------------------------------------------
# LOAD SENSOR DATA
# ------------------------------------------------------------
def load_sensor(dirpath):
    df = pd.concat(pd.read_csv(f) for f in sorted(dirpath.glob("*.csv")))
    df["loggingTime"] = pd.to_datetime(df["loggingTime(txt)"], utc=True)
    return df.sort_values("loggingTime")

def slice_sensor(df):
    df = df[
        (df.loggingTime >= session_start + pd.to_timedelta(SENSOR_LAG_SEC, "s")) &
        (df.loggingTime <= session_end   + pd.to_timedelta(SENSOR_LAG_SEC, "s"))
    ].copy()
    ts = df.loggingTime.astype("int64") / 1e9

    vals = np.stack([
        df[X_COL].values,
        df[Y_COL].values,
        df[Z_COL].values
    ], axis=1)  # shape: (N, 3)

    return ts.values, vals

blue_t, blue_v   = slice_sensor(load_sensor(blue_dir))
green_t, green_v = slice_sensor(load_sensor(green_dir))

# ------------------------------------------------------------
# TIME BASE (VIDEO-DRIVEN)
# ------------------------------------------------------------
frame_idxs = np.arange(frame_count, dtype=np.int32)
t0 = min(blue_t.min(), green_t.min())

# ------------------------------------------------------------
# FAST WINDOW SLICE
# ------------------------------------------------------------
def window_slice(times, vals, t, window):
    left  = np.searchsorted(times, t - window, side="left")
    right = np.searchsorted(times, t + window, side="right")
    return times[left:right] - t, vals[left:right]

# ------------------------------------------------------------
# DRAWING
# ------------------------------------------------------------
AXIS_COLORS = [
    (255, 0, 0),   # X = Red   (BGR)
    (0, 255, 0),   # Y = Green
    (0, 0, 255),   # Z = Blue
]

def draw_signal_panel(frame, rel_t, vals3, panel="left"):
    h, w, _ = frame.shape
    panel_w = w // 2

    x0 = 0 if panel == "left" else panel_w
    x_center = x0 + panel_w // 2

    base_y = h - 100
    sx, sy = 80, 40

    # Center time line (t = 0)
    cv2.line(frame, (x_center, base_y - 140), (x_center, h), (255, 255, 255), 2)

    if len(rel_t) == 0:
        return

    # Draw each axis
    for axis in range(3):
        color = AXIS_COLORS[axis]
        vals = vals3[:, axis]

        pts = []
        for t, v in zip(rel_t, vals):
            x = int(x_center + t * sx)
            y = int(base_y - v * sy)
            pts.append((x, y))

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], color, 2)

        # Mark current point (closest to t=0)
        idx0 = np.argmin(np.abs(rel_t))
        cx, cy = pts[idx0]
        cv2.circle(frame, (cx, cy), 4, color, -1)

    # Panel border
    cv2.rectangle(frame, (x0, base_y - 140), (x0 + panel_w, h), (80, 80, 80), 1)

# ------------------------------------------------------------
# WORKER
# ------------------------------------------------------------
def process_batch(args):
    frames_batch, global_start = args
    out = []

    for j, f in enumerate(frames_batch):
        i = global_start + j
        f = f.copy()

        # current video time
        t = t0 + i / fps

        rb, vb = window_slice(blue_t, blue_v, t, WINDOW_SEC)
        rg, vg = window_slice(green_t, green_v, t, WINDOW_SEC)

        # Left = Green watch, Right = Blue watch
        draw_signal_panel(f, rb, vb, panel="right")
        draw_signal_panel(f, rg, vg, panel="left")

        # Labels
        start_str = session_start.strftime("%Y-%m-%d %H:%M:%S UTC")
        cv2.putText(f, f"Green Watch (X=R, Y=G, Z=B) | Start: {start_str}", (20, f.shape[0] - 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(f, f"Blue Watch (X=R, Y=G, Z=B) | Start: {start_str}", (f.shape[1]//2 + 20, f.shape[0] - 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.append(f)

    return out

# ------------------------------------------------------------
# FFMPEG SETUP
# ------------------------------------------------------------
H, W, _ = vr[0].asnumpy().shape

ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo",
    "-pix_fmt", "rgb24",
    "-s", f"{W}x{H}",
    "-r", str(fps),
    "-i", "-",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-threads", str(N_WORKERS),
    "-r", str(fps),
    OUT_VIDEO
]

proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=10**8)

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
pool = Pool(processes=N_WORKERS, maxtasksperchild=100)
print("[STREAM] Processing in chunks...")

for start in range(0, len(frame_idxs), CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, len(frame_idxs))
    idx_chunk = frame_idxs[start:end]
    frames = vr.get_batch(idx_chunk).asnumpy()

    tasks = []
    for i in range(0, len(frames), BATCH_PER_WORKER):
        tasks.append((frames[i:i+BATCH_PER_WORKER], start + i))

    for batch in pool.imap_unordered(process_batch, tasks, chunksize=1):
        for f in batch:
            proc.stdin.write(f.tobytes())

    del frames, tasks
    print(f"[STREAM] {start} → {end}")

pool.close()
pool.join()

proc.stdin.close()
proc.wait()

print("✔ Finished.")
