"""Quick latency benchmark for the Piccolo pipeline."""
import sys, time
sys.stdout.reconfigure(line_buffering=True)

from src.config import load_config
from src.camera import CameraCapture
from src.stereo_processor import StereoProcessor
from src.display import StereoDisplay
import pygame
import numpy as np

cfg = load_config("config.yaml")
cfg.display.fullscreen = False

print("Opening cameras...", flush=True)
cam_l = CameraCapture(0, 1920, 1080, backend="opencv", name="cam-L").start()
cam_r = CameraCapture(1, 1920, 1080, backend="opencv", name="cam-R").start()
time.sleep(1)  # let cameras warm up

proc = StereoProcessor(cfg.stereo, 960, 1080)
disp = StereoDisplay(cfg.display)
disp.open()

FRAMES = 120
WARMUP = 20
times = []
t_cam = []
t_proc = []
t_disp = []

print(f"Benchmarking {FRAMES} frames ({WARMUP} warmup)...", flush=True)
for i in range(FRAMES):
    t0 = time.perf_counter()

    fl = cam_l.read_no_copy()
    fr = cam_r.read_no_copy()
    t1 = time.perf_counter()

    if fl is None or fr is None:
        disp.tick()
        for ev in pygame.event.get():
            pass
        continue

    eye_l, eye_r, sbs = proc.process_pair(fl, fr)
    t2 = time.perf_counter()

    disp.show(sbs)
    t3 = time.perf_counter()

    disp.tick()
    for ev in pygame.event.get():
        pass

    dt = (time.perf_counter() - t0) * 1000
    times.append(dt)
    t_cam.append((t1 - t0) * 1000)
    t_proc.append((t2 - t1) * 1000)
    t_disp.append((t3 - t2) * 1000)

cam_l.stop()
cam_r.stop()
disp.close()

if times:
    # Skip warmup frames
    times = times[WARMUP:]
    t_cam = t_cam[WARMUP:]
    t_proc = t_proc[WARMUP:]
    t_disp = t_disp[WARMUP:]

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    fps = 1000 / avg if avg > 0 else 0

    print(f"\n{'='*50}", flush=True)
    print(f"RESULTS ({len(times)} frames, warmup skipped)", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  Total loop:  avg={avg:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms", flush=True)
    print(f"  Effective FPS: {fps:.0f}", flush=True)
    print(f"  Camera read: avg={sum(t_cam)/len(t_cam):.2f}ms", flush=True)
    print(f"  Stereo proc: avg={sum(t_proc)/len(t_proc):.2f}ms", flush=True)
    print(f"  Display:     avg={sum(t_disp)/len(t_disp):.2f}ms", flush=True)
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    print(f"  P50={p50:.1f}ms  P95={p95:.1f}ms", flush=True)
else:
    print("No frames captured!", flush=True)
