#!/usr/bin/env python3
"""Slice dmon.log by the bench windows in phases.log and report per-window
hardware-counter stats (SM util %, memory-bandwidth util %, power, HBM used).

H20 reference: fp16/bf16 dense ~148 TFLOPS, fp8 ~296 TFLOPS, HBM3 ~4.0 TB/s.
dmon 'sm' = % of time at least one kernel was on an SM; 'mem' = % of time the
memory controller was busy (DRAM active) -- both time-based, 1Hz samples.
"""
import re
from datetime import datetime

samples = []  # (dt, pwr, sm, mem, fb)
for line in open("dmon.log"):
    if line.startswith("#"):
        continue
    f = line.split()
    if len(f) < 12:
        continue
    dt = datetime.strptime(f[0] + " " + f[1], "%Y%m%d %H:%M:%S")
    samples.append((dt, int(f[3]), int(f[6]), int(f[7]), int(f[12])))

marks = {}
order = []
for line in open("phases.log"):
    name, d, t = line.split()
    marks[name] = datetime.strptime(d + " " + t, "%Y-%m-%d %H:%M:%S")
    order.append(name)

windows = []
for name, ts in marks.items():
    if name.endswith("_START") and name.replace("_START", "_END") in marks:
        base = name[:-6]
        windows.append((base, ts, marks[base + "_END"]))

print(f"{'window':<16}{'dur_s':>6}{'sm%avg':>8}{'sm%max':>8}{'mem%avg':>9}{'mem%max':>9}{'pwr_avg':>9}{'fb_GB':>7}")
for base, t0, t1 in windows:
    win = [s for s in samples if t0 <= s[0] <= t1]
    if not win:
        continue
    n = len(win)
    sm_a = sum(s[2] for s in win) / n
    mem_a = sum(s[3] for s in win) / n
    print(f"{base:<16}{(t1-t0).total_seconds():>6.0f}{sm_a:>8.1f}{max(s[2] for s in win):>8}"
          f"{mem_a:>9.1f}{max(s[3] for s in win):>9}"
          f"{sum(s[1] for s in win)/n:>9.0f}{max(s[4] for s in win)/1024:>7.1f}")
