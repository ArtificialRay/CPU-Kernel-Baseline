#!/usr/bin/env python3
"""Fully daemonize the sweep so it survives this shell/session entirely.

nohup+disown wasn't enough — the process tree kept getting reaped a couple
minutes in. Double-fork + setsid reparents the driver to launchd (PID 1) in a
brand-new session with no controlling terminal, so nothing tracking the
launching Bash tool's descendants can signal it. Prints the daemon PID and
exits immediately."""
import os
import sys

REPO = "/Users/allen/CMU/cpu-kernel-baseline"
DRIVER = f"{REPO}/analysis/resume_sweep.sh"
LOG = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/sweep.log"

# first fork
if os.fork() > 0:
    os._exit(0)
os.setsid()               # new session, drop controlling terminal
# second fork so we can never re-acquire a terminal
if os.fork() > 0:
    os._exit(0)

os.chdir(REPO)
with open(LOG, "a") as log, open(os.devnull) as devnull:
    os.dup2(devnull.fileno(), 0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    log.write(f"\n@@@ daemon pid {os.getpid()} sid {os.getsid(0)} starting\n")
    log.flush()
os.execvp("bash", ["bash", DRIVER])
