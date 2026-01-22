import os
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from multiprocessing import Pool
from decord import VideoReader, cpu, gpu
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
session_id = 385
metadata_csv = "./markers/session1_complete_final_markers.csv"
# video_file = "/ds/videos/nurse_2.0/videos_sync/1080p/session_1/383_all_cams.mp4"
video_file = '385_all_cams_anonymized.mkv'

green_dir = Path("./sensors/D1/Green")
blue_dir  = Path("./sensors/D1/Blue")

Z_COL = "accelerometerAccelerationZ(G)"
WINDOW_SEC = 5.0
SENSOR_LAG_SEC = 1.5

CHUNK_SIZE = 256          # frames per streaming chunk
BATCH_PER_WORKER = 16     # frames per task
OUT_VIDEO = f"session_{session_id}_stable.mkv"

# ------------------------------------------------------------
# CPU COUNT (RESPECT SCHEDULER)
# ------------------------------------------------------------
N_WORKERS = int(
    os.environ.get("SLURM_CPUS_PER_TASK",
    os.environ.get("OMP_NUM_THREADS",
    os.cpu_count()))
)

print(f"[CPU] Using {N_WORKERS} workers")

# ------------------------------------------------------------
# VIDEO READER (SAFE FALLBACK)
# ------------------------------------------------------------
def get_video_reader(path):
    try:
        if torch.cuda.is_available():
            try:
                return VideoReader(path, ctx=gpu(0)), "gpu"
            except Exception:
                pass
        return VideoReader(path, ctx=cpu(0)), "cpu"
    except Exception as e:
        print("⚠ Decord not available, falling back to OpenCV")
        import cv2
        cap = cv2.VideoCapture(path)
        return cap, "opencv"

video_reader, backend = get_video_reader(video_file)

if backend in ["gpu", "cpu"]:
    fps = video_reader.get_avg_fps()
    frame_count = len(video_reader)
else:
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))


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
    return ts.values, df[Z_COL].values

blue_t, blue_v   = slice_sensor(load_sensor(blue_dir))
green_t, green_v = slice_sensor(load_sensor(green_dir))

Y_MIN = min(blue_v.min(), green_v.min())
Y_MAX = max(blue_v.max(), green_v.max())

# ------------------------------------------------------------
# MASTER TIMELINE
# ------------------------------------------------------------
master_t = np.sort(np.unique(np.concatenate([blue_t, green_t])))
t0 = master_t[0]

frame_idxs = np.clip(((master_t - t0) * fps).astype(np.int32), 0, frame_count - 1)

# ------------------------------------------------------------
# PRECOMPUTE SENSOR WINDOWS (ONCE)
# ------------------------------------------------------------
def build_windows(times, vals):
    out = []
    for t in master_t:
        m = np.abs(times - t) <= WINDOW_SEC
        out.append((times[m] - t, vals[m]))
    return out

blue_win  = build_windows(blue_t, blue_v)
green_win = build_windows(green_t, green_v)

# ------------------------------------------------------------
# FIGURE SETUP
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax_video = fig.add_subplot(2, 1, 1)
ax_plot  = fig.add_subplot(2, 1, 2)

# First frame
if backend in ["gpu", "cpu"]:
    frame0 = video_reader[0].asnumpy()
else:
    ret, frame0 = video_reader.read()
    frame0 = frame0[..., ::-1]

img_disp = ax_video.imshow(frame0)
ax_video.axis("off")

# lines = [ax_plot.plot([], [], label=c)[0] for c in cols]
# vline = ax_plot.axvline(0, color="k", linestyle="--", label="Now")
line_b, = ax_plot.plot([], [], color="blue", label="Blue Z")
line_g, = ax_plot.plot([], [], color="green", label="Green Z")
vline = ax_plot.axvline(0, color="k", linestyle="--")

ax_plot.set_xlim(-WINDOW_SEC, WINDOW_SEC)
ax_plot.set_ylim(Y_MIN,Y_MAX)
ax_plot.legend()

# # ============================================================
# # PARALLEL WORKER
# # ============================================================
# def process_batch(args):
#     frames_batch, global_start = args
#     out = []

#     H, W, _ = frames_batch[0].shape
#     plot_h = int(H * 0.35)

#     for j, frame in enumerate(frames_batch):
#         i = global_start + j
#         rb, vb = blue_win[i]
#         rg, vg = green_win[i]

#         plot_img = render_sensor_plot(rb, vb, rg, vg, W, plot_h)
#         plot_img = cv2.resize(plot_img, (W, plot_h))

#         combined = np.vstack([frame, plot_img])
#         out.append(combined)

#     return out

# # ------------------------------------------------------------
# # STREAMING ENCODE SETUP (CPU SAFE)
# # ------------------------------------------------------------
# H, W, _ = vr[0].asnumpy().shape

# # CPU-safe encoding: libx264 (veryfast)
# ffmpeg_cmd = [
#     "ffmpeg", "-y",
#     "-f", "rawvideo",
#     "-pix_fmt", "rgb24",
#     "-s", f"{W}x{H}",
#     "-r", str(fps),
#     "-i", "-",
#     "-c:v", "libx264",
#     "-preset", "veryfast",
#     "-crf", "18",
#     "-pix_fmt", "yuv420p",
#     OUT_VIDEO
# ]

# proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# # ============================================================
# # MAIN STREAM LOOP
# # ============================================================
# pool = Pool(N_WORKERS)
# print("[STREAM] Processing...")

# for start in range(0, len(frame_idxs), CHUNK_SIZE):
#     end = min(start + CHUNK_SIZE, len(frame_idxs))
#     idx_chunk = frame_idxs[start:end]
#     frames = vr.get_batch(idx_chunk).asnumpy()

#     tasks = []
#     for i in range(0, len(frames), BATCH_PER_WORKER):
#         tasks.append((frames[i:i+BATCH_PER_WORKER], start + i))

#     results = pool.map(process_batch, tasks)

#     for batch in results:
#         for f in batch:
#             proc.stdin.write(f.tobytes())

#     del frames, results
#     print(f"[STREAM] {start} → {end}")

# pool.close()
# pool.join()
# proc.stdin.close()
# proc.wait()

# print("✔ Finished. CPU fully utilized.")

# ------------------------------------------------------------
# UPDATE
# ------------------------------------------------------------
def update(i):
    idx = frame_idxs[i]
    if backend in ["gpu", "cpu"]:
        frame = video_reader[idx].asnumpy()
    else:
        video_reader.set(1, idx)
        _, frame = video_reader.read()
        frame = frame[..., ::-1]
    img_disp.set_array(frame)


    rb, vb = blue_win[i]
    rg, vg = green_win[i]
    line_b.set_data(rb, vb)
    line_g.set_data(rg, vg)


    return img_disp, line_b, line_g, vline


ani = FuncAnimation(fig, update, frames=len(master_t), interval=1000/fps)
ani.save(OUT_VIDEO, writer="ffmpeg", fps=fps)

print("✔ Done. Video generated successfully.")
