# Auto-imported by Python when this dir is on PYTHONPATH.
# Env workaround only (torch 2.11 inductor duplicate-template import bug on this
# sm89 box). Must run before sglang import. NOT part of the PR.
#
# sglang's deep_gemm module does `from torch._inductor.compile_fx import
# get_patched_config_dict` at import time, which drags in the broken inductor
# import chain. We only need that one name to exist; the server runs without
# torch.compile so nothing actually calls into inductor.
import sys
import types

if "torch._inductor.compile_fx" not in sys.modules:
    _m = types.ModuleType("torch._inductor.compile_fx")
    _m.get_patched_config_dict = lambda *a, **k: {}
    sys.modules["torch._inductor.compile_fx"] = _m
