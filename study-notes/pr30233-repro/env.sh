# Source this before any sglang command on the AutoDL 4090 box.
# It bakes in the two environment workarounds this machine needs (both are
# machine/driver issues, NOT related to the PR):
#   1. libnvrtc.so.13 lives under nvidia/cu13/lib but isn't on the loader path,
#      so sgl_kernel fails to import.
#   2. torch 2.11's _inductor has a duplicate-template assert on import; we stub
#      torch._inductor.compile_fx (only get_patched_config_dict is needed at
#      import time by sglang's deep_gemm module).
# And it points PYTHONPATH at the patched sglang tree.

export PY=/root/miniconda3/bin/python
export REPRO=/root/autodl-tmp/pr30233
export PYTHONPATH=$REPRO/python

# miniconda bin on PATH so flashinfer's JIT can find the `ninja` executable.
export PATH=/root/miniconda3/bin:$PATH

# (1) nvrtc lib path
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

# (1b) mooncake engine needs GLIBCXX_3.4.30 which conda's libstdc++ lacks;
# preload the system libstdc++ (6.0.30) which has it.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:${LD_PRELOAD:-}

# (2) inductor import stub, installed via a sitecustomize on PYTHONPATH
export PYTHONPATH=$REPRO/repro/_sitefix:$PYTHONPATH

# (3) box has sgl-kernel 0.4.3, patched tree (main) wants 0.4.4. The fix is pure
# scheduler logic (no kernel changes), so skipping the version guard is safe here.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# model + HF mirror
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct
