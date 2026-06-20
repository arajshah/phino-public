"""Auto-imported shim so DINO-WM runs under DDP without touching prejepa.py.

Lightning's DDP launcher re-executes ``prejepa.py`` directly in each worker
subprocess, bypassing the ``CPUOffloadCallback`` shim that ``train_dinowm.py``
injects (that wrapper only covers the single, in-process case). ``prejepa.py``
hardcodes ``spt.callbacks.CPUOffloadCallback`` (an optional memory optimization)
which is absent from the released ``stable_pretraining`` (0.1.7).

Python auto-imports ``sitecustomize`` at interpreter startup if it is found on
``sys.path``. By putting this directory on ``PYTHONPATH`` for the training
command, EVERY process — rank 0 and all DDP subprocesses — applies the no-op
shim before ``prejepa.py`` builds its callbacks. prejepa.py stays unmodified.

This file shadows the system ``sitecustomize`` (because PYTHONPATH is searched
first), so we explicitly chain-load the original to preserve its behavior.
"""

# 1) Apply the CPUOffloadCallback shim (best effort; harmless if already present).
try:
    import lightning as pl
    import stable_pretraining as spt

    if not hasattr(spt.callbacks, "CPUOffloadCallback"):
        spt.callbacks.CPUOffloadCallback = type(
            "CPUOffloadCallback", (pl.Callback,), {}
        )
except Exception:
    pass

# 2) Chain-load the system sitecustomize we shadowed, so we don't disable it.
try:
    import importlib.util as _u
    import os as _os

    for _p in ("/usr/lib/python3.12/sitecustomize.py",):
        if _os.path.exists(_p):
            _spec = _u.spec_from_file_location("_system_sitecustomize", _p)
            _mod = _u.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            break
except Exception:
    pass
