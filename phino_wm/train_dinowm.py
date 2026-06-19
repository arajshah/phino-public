#!/usr/bin/env python
"""Non-invasive launcher for stable-worldmodel's DINO-WM (prejepa.py).

The vendored ``scripts/train/prejepa.py`` hardcodes
``spt.callbacks.CPUOffloadCallback`` (an optional memory optimization) which is
absent from the released ``stable-pretraining`` (0.1.7). Rather than edit their
source, we inject a no-op shim into the already-imported ``stable_pretraining``
module, then run prejepa.py unmodified via runpy. All CLI args are forwarded to
Hydra untouched.

Usage:
    python train_dinowm.py --config-name phino/pusht_baseline trainer.devices=1 ...
"""

import runpy

import lightning as pl
import stable_pretraining as spt

# Provide the optional callback if the installed stable_pretraining lacks it.
if not hasattr(spt.callbacks, "CPUOffloadCallback"):

    class _NoOpCPUOffloadCallback(pl.Callback):
        """Stand-in for the unreleased CPUOffloadCallback (no-op)."""

    spt.callbacks.CPUOffloadCallback = _NoOpCPUOffloadCallback

PREJEPA = "/home/colligo/phino/stable-worldmodel/scripts/train/prejepa.py"

if __name__ == "__main__":
    # run_name='__main__' triggers prejepa.py's `if __name__ == '__main__': run()`,
    # so its @hydra.main parses sys.argv exactly as a direct invocation would.
    runpy.run_path(PREJEPA, run_name="__main__")
