#!/usr/bin/env python3
"""wslock.py - WarpSpeed GPU lock runner (mechanics layer under infra/gpu_run).

Owns everything that must behave identically on darwin (mock tests) and linux
(GPU node): advisory file locks via fcntl.flock(2) (kernel releases them on
process death - no stale-lock cleanup ever needed), child process-group
lifecycle with timeout kill, device export, and GPU-minutes accounting.

Subcommands:
  run    acquire device lock(s) per --mode, run a command, release, propagate exit
  probe  report free/held for each device lock (used by acceptance tests)

Exit codes for `run`:
  child's exit code      normal completion
  124                    child exceeded --timeout (group SIGTERM then SIGKILL)
  125                    lock not acquired within --lock-wait
  128+N                  child killed by signal N (shell convention)

Lock files live in --lock-dir as gpu_<i>; they are MACHINE-GLOBAL (default
/tmp/warpspeed/locks) so concurrent searches on different projects can never
double-book a device. Lock files are never unlinked.
"""

import argparse
import errno
import fcntl
import json
import os
import random
import signal
import subprocess
import sys
import time


class LockTimeout(Exception):
    pass


def _warn(msg):
    sys.stderr.write("wslock: %s\n" % msg)
    sys.stderr.flush()


def lock_path(lock_dir, device):
    return os.path.join(lock_dir, "gpu_%d" % device)


def ensure_lock_dir(lock_dir):
    try:
        os.makedirs(lock_dir, exist_ok=True)
    except OSError as e:
        raise SystemExit("wslock: cannot create lock dir %s: %s" % (lock_dir, e))
    # Best-effort: world-writable + sticky so any user on a shared node can
    # create/flock lock files (flock needs only an open fd, but O_CREAT does).
    try:
        os.chmod(lock_dir, 0o1777)
    except OSError:
        pass


def open_lock(lock_dir, device):
    p = lock_path(lock_dir, device)
    fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        os.chmod(p, 0o666)
    except OSError:
        pass
    return fd


def try_lock_nb(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as e:
        if e.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            return False
        raise


def lock_blocking(fd, remaining):
    """Blocking flock with a deadline. Kernel wake order on contended flocks is
    near-FIFO; strict mutual exclusion is what we guarantee (documented
    deviation from the spec's literal 'FIFO')."""
    if remaining <= 0:
        raise LockTimeout()

    def _alarm(_sig, _frm):
        raise LockTimeout()

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def write_holder(fd, info):
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, (json.dumps(info) + "\n").encode())
    except OSError:
        pass


def acquire(args):
    """Returns (devices_held, fds). fds stay open for the lock's lifetime."""
    deadline = time.monotonic() + args.lock_wait
    if args.mode == "pool":
        # Caller (gpu_run) passes devices pre-shuffled. Try each nonblocking;
        # if all busy, poll the whole set until one frees up or deadline.
        while True:
            for dev in args.devices:
                fd = open_lock(args.lock_dir, dev)
                if try_lock_nb(fd):
                    return [dev], [fd]
                os.close(fd)
            if time.monotonic() >= deadline:
                raise LockTimeout()
            time.sleep(0.5 + random.random() * 0.5)
    elif args.mode == "exclusive":
        dev = args.devices[0]
        fd = open_lock(args.lock_dir, dev)
        lock_blocking(fd, deadline - time.monotonic())
        return [dev], [fd]
    elif args.mode == "all":
        # Ascending order = deadlock-safe against every other acquirer.
        devs = sorted(args.devices)
        fds = []
        try:
            for dev in devs:
                fd = open_lock(args.lock_dir, dev)
                lock_blocking(fd, deadline - time.monotonic())
                fds.append(fd)
        except Exception:
            for fd in fds:
                os.close(fd)
            raise
        return devs, fds
    else:
        raise SystemExit("wslock: unknown mode %r" % args.mode)


def run_hook(label, cmd):
    """Run a pre/post hook via the shell. Returns exit code."""
    if not cmd:
        return 0
    rc = subprocess.call(cmd, shell=True)
    if rc != 0:
        _warn("%s hook exited %d: %s" % (label, rc, cmd))
    return rc


def append_accounting(record):
    state = os.environ.get("WARPSPEED_STATE", "")
    if not state:
        return
    try:
        log_dir = os.path.join(state, "logs")
        os.makedirs(log_dir, exist_ok=True)
        line = (json.dumps(record) + "\n").encode()
        fd = os.open(os.path.join(log_dir, "gpu_minutes.jsonl"),
                     os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except OSError as e:
        _warn("accounting write failed: %s" % e)


def cmd_run(args):
    ensure_lock_dir(args.lock_dir)
    try:
        devices, fds = acquire(args)
    except LockTimeout:
        _warn("lock not acquired within %ss (mode=%s devices=%s)"
              % (args.lock_wait, args.mode, args.devices))
        return 125

    dev_str = ",".join(str(d) for d in devices)
    for fd in fds:
        write_holder(fd, {"pid": os.getpid(), "role": args.role,
                          "label": args.label, "since": time.time(),
                          "devices": dev_str})
    # Contract: callers parse this exact line from stderr to record device_id.
    sys.stderr.write("GPU_RUN_DEVICE=%s\n" % dev_str)
    sys.stderr.flush()

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = dev_str
    env["GPU_RUN_DEVICE"] = dev_str
    env["WARPSPEED_GPU_ROLE"] = args.role

    t0 = time.monotonic()
    rc = 130
    timed_out = False
    try:
        pre_rc = run_hook("pre", args.pre)
        if pre_rc != 0 and args.strict_hooks:
            return pre_rc
        proc = subprocess.Popen(args.cmd, env=env, start_new_session=True)

        pending = {"sig": None}

        def _forward(sig, _frm):
            pending["sig"] = sig
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass

        old_term = signal.signal(signal.SIGTERM, _forward)
        old_int = signal.signal(signal.SIGINT, _forward)
        try:
            try:
                rc = proc.wait(timeout=args.timeout if args.timeout else None)
            except subprocess.TimeoutExpired:
                timed_out = True
                _warn("timeout after %ss; killing process group %d"
                      % (args.timeout, proc.pid))
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    proc.wait()
                rc = 124
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
        if pending["sig"] is not None and not timed_out:
            rc = 128 + pending["sig"]
        elif rc < 0:  # child died from a signal
            rc = 128 - rc
        return rc
    finally:
        run_hook("post", args.post)
        wall = time.monotonic() - t0
        append_accounting({
            "ts_end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "role": args.role, "devices": dev_str, "label": args.label,
            "exp_id": os.environ.get("WARPSPEED_EXP_ID", ""),
            "wall_s": round(wall, 3), "exit": rc, "timeout": timed_out,
        })
        for fd in fds:
            try:
                os.close(fd)  # closing the fd releases the flock
            except OSError:
                pass


def cmd_probe(args):
    ensure_lock_dir(args.lock_dir)
    out = {}
    for dev in args.devices:
        fd = open_lock(args.lock_dir, dev)
        holder = None
        try:
            if try_lock_nb(fd):
                fcntl.flock(fd, fcntl.LOCK_UN)
                state = "free"
            else:
                state = "held"
                try:
                    with open(lock_path(args.lock_dir, dev)) as f:
                        holder = json.loads(f.read() or "null")
                except (OSError, ValueError):
                    holder = None
        finally:
            os.close(fd)
        out[str(dev)] = {"state": state, "holder": holder}
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


def parse_devices(s):
    # order-preserving dedupe: a repeated device would deadlock this process
    # against itself (flock on a second fd of an already-held lock file blocks)
    out = []
    for x in s.split(","):
        if x != "" and int(x) not in out:
            out.append(int(x))
    return out


def main(argv):
    ap = argparse.ArgumentParser(prog="wslock.py")
    sub = ap.add_subparsers(dest="sub")

    rp = sub.add_parser("run")
    rp.add_argument("--lock-dir", required=True)
    rp.add_argument("--devices", required=True, type=parse_devices)
    rp.add_argument("--mode", required=True, choices=["pool", "exclusive", "all"])
    rp.add_argument("--lock-wait", type=float, default=3600.0)
    rp.add_argument("--timeout", type=float, default=None)
    rp.add_argument("--role", default="pool")
    rp.add_argument("--label", default="")
    rp.add_argument("--pre", default="")
    rp.add_argument("--post", default="")
    rp.add_argument("--strict-hooks", action="store_true")
    rp.add_argument("cmd", nargs=argparse.REMAINDER)

    pp = sub.add_parser("probe")
    pp.add_argument("--lock-dir", required=True)
    pp.add_argument("--devices", required=True, type=parse_devices)

    args = ap.parse_args(argv)
    if args.sub == "run":
        if args.cmd and args.cmd[0] == "--":
            args.cmd = args.cmd[1:]
        if not args.cmd:
            ap.error("run: missing command after --")
        return cmd_run(args)
    if args.sub == "probe":
        return cmd_probe(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
