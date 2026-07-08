# PR #30233 e2e repro (AutoDL 4090, PD disaggregation)

Reproduces: PD disaggregation garbage output when an over-long input request
(`input_len >= max_req_input_len`, auto-truncate off) is aborted on prefill but
still enters the KV-transfer pipeline. Fix streams the abort back to the client
before enqueue.

## Layout (on remote: `/root/autodl-tmp/pr30233/repro`)
- `env.sh` — env workarounds (nvrtc lib path, inductor import stub) + PYTHONPATH
  to the patched tree + model/HF-mirror. Source before anything.
- `_sitefix/sitecustomize.py` — the inductor import stub (auto-loaded via PYTHONPATH).
- `launch_pd.sh` — start prefill (GPU0) + decode (GPU1) + mini-LB. Logs → `logs/`.
- `repro.py` — send short + over-long + short requests through the LB, print verdict.
- `capture_metrics.sh <label>` — snapshot prefill/decode `/metrics` + `/get_server_info`.
- `teardown.sh` — stop the cluster.

## Run (once machine is in 2-GPU mode)
```bash
cd /root/autodl-tmp/pr30233/repro
bash launch_pd.sh                 # wait until "cluster up"
python repro.py --lb http://127.0.0.1:8000 --ctx 4096
bash capture_metrics.sh after-overlong
bash teardown.sh
```

## Expected
- FIXED tree: over-long request → HTTP 400 with "Input length ... exceeds ..."
  within a second or two; cluster healthy for the follow-up request.
- Buggy tree (revert the fix): over-long request → garbage completion or hang;
  decode ran on uninitialized KV.

## To compare against the buggy behavior
Point PYTHONPATH at the installed (unpatched) sglang, or `git stash` the fix in
the patched tree, relaunch, and re-run `repro.py`.

## Notes
- Transfer backend: mooncake (installed). Needs GPUs attached to import
  (`libcuda.so.1` is a stub in no-card mode).
- The env workarounds are machine-specific (torch 2.11 / sm89 sgl_kernel), NOT
  part of the PR.
