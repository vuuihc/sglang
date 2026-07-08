#!/bin/bash
# Stop the PD cluster started by launch_pd.sh.
cd "$(dirname "$0")"
for p in lb decode prefill; do
  if [ -f "logs/$p.pid" ]; then
    kill "$(cat logs/$p.pid)" 2>/dev/null && echo "killed $p ($(cat logs/$p.pid))"
    rm -f "logs/$p.pid"
  fi
done
# belt-and-suspenders: kill any stray launch_server / router
pkill -f "sglang.launch_server" 2>/dev/null
pkill -f "sglang_router.launch_router" 2>/dev/null
echo "teardown done"
