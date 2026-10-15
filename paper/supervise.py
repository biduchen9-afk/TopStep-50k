"""Keep the forward-test harness running: restart it when it exits, with backoff, log its output and keep rolling backups of the state and bar files (local copies only; never commit them).
   python -m paper.supervise [--log paper_harness.log] [--backup-dir paper_backups] [--backup-hours 6] -- --source ticks --symbol NQ.F --dataset futures --warm-from-lse 60 --fresh --port 8765
Everything after `--` is passed to `python -m paper.run`. The first launch uses the arguments as given; later launches drop --fresh and --warm-from-lse when the state file already exists, so a restart
resumes the journal instead of starting a new clock. The LSE key is read from the environment (or --key-file, which is passed through); it is never printed or logged."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Callable, List, Optional


def restart_args(args: List[str], state_path: str) -> List[str]:
    """Arguments for a relaunch: keep the clock (no --fresh, no warm-up) when a state file exists."""
    if not os.path.exists(state_path):
        return list(args)
    out, skip = [], False
    for x in args:
        if skip:
            skip = False
            continue
        if x == "--fresh":
            continue
        if x == "--warm-from-lse":
            skip = True
            continue
        if x.startswith("--warm-from-lse="):
            continue
        out.append(x)
    return out


def state_path_of(args: List[str]) -> str:
    for i, x in enumerate(args):
        if x == "--state" and i + 1 < len(args):
            return args[i + 1]
        if x.startswith("--state="):
            return x.split("=", 1)[1]
    return "paper_state.pkl"


def backup(state: str, backup_dir: str, keep: int = 14, now: Optional[float] = None) -> List[str]:
    os.makedirs(backup_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M", time.localtime(now or time.time()))
    bars = os.path.splitext(state)[0] + "_bars.csv"
    made = []
    for src in (state, bars):
        if os.path.exists(src):
            dst = os.path.join(backup_dir, f"{os.path.basename(src)}.{stamp}")
            shutil.copy2(src, dst); made.append(dst)
    for base in {os.path.basename(state), os.path.basename(bars)}:
        old = sorted(f for f in os.listdir(backup_dir) if f.startswith(base + "."))
        for f in old[:-keep]:
            os.remove(os.path.join(backup_dir, f))
    return made


def supervise(args: List[str], log: str = "paper_harness.log", backup_dir: Optional[str] = "paper_backups", backup_hours: float = 6.0, max_restarts: Optional[int] = None,
              sleep: Callable[[float], None] = time.sleep, python: str = sys.executable, module: str = "paper.run", poll: float = 5.0) -> int:
    state = state_path_of(args); fails = 0; launches = 0; last_backup = time.time()
    while True:
        cmd = [python, "-m", module] + (list(args) if launches == 0 else restart_args(args, state))
        launches += 1
        t0 = time.time()
        with open(log, "a") as lf:
            lf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} launch {launches}: {' '.join(cmd[1:])}\n"); lf.flush()
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            try:
                while p.poll() is None:
                    time.sleep(poll)
                    if backup_dir and time.time() - last_backup >= backup_hours * 3600:
                        backup(state, backup_dir); last_backup = time.time()
            except KeyboardInterrupt:
                p.terminate()
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    p.kill()
                if backup_dir:
                    backup(state, backup_dir)
                return 130
            lf.write(f"=== exited with code {p.returncode} after {time.time() - t0:.0f} s\n")
        if backup_dir:
            backup(state, backup_dir)
        fails = 0 if time.time() - t0 > 600 else fails + 1
        if max_restarts is not None and launches > max_restarts:
            return p.returncode or 0
        sleep(min(5.0 * 2 ** min(fails, 6), 300.0))


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" in argv:
        i = argv.index("--"); mine, theirs = argv[:i], argv[i + 1:]
    else:
        mine, theirs = argv, []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="paper_harness.log")
    ap.add_argument("--backup-dir", default="paper_backups")
    ap.add_argument("--backup-hours", type=float, default=6.0)
    a = ap.parse_args(mine)
    if not theirs:
        sys.exit("give the harness arguments after --, e.g.  -- --source ticks --symbol NQ.F --dataset futures --warm-from-lse 60 --fresh")
    return supervise(theirs, a.log, a.backup_dir, a.backup_hours)


if __name__ == "__main__":
    sys.exit(main())
