"""Bootstrap to run the load-back timing unit test on a host where the
installed flashinfer is older than main expects.

Shims the few attrs main's import chain reaches before importing sglang, then
hands off to unittest. Throwaway script for benchmarking on rented GPUs.
"""

import sys
import unittest

# --- shim flashinfer surface that main's quantization init touches ---
import flashinfer as _fi  # noqa: E402

for _missing, _fallback in [
    ("mm_mxfp8", "bmm_mxfp8"),
]:
    if not hasattr(_fi, _missing) and hasattr(_fi, _fallback):
        setattr(_fi, _missing, getattr(_fi, _fallback))

# --- prepend our PR's source tree to the import path ---
sys.path.insert(0, "/root/sglang_pr/python")

if __name__ == "__main__":
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_hicache_load_back_timing",
        "/root/sglang_pr/test/registered/unit/mem_cache/test_hicache_load_back_timing.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(module)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
