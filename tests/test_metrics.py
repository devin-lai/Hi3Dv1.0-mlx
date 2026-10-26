"""The reported memory budget must include the short-lived conditioner process."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hi3d_mac.metrics import Profiler


def test_child_physical_footprint_is_counted():
    profiler = Profiler(interval=0.01)
    try:
        with profiler.phase("child"):
            child = subprocess.Popen([sys.executable, "-c",
                "import time; x=bytearray(128*1024**2); time.sleep(0.7)"])
            assert child.wait(timeout=10) == 0
    finally:
        profiler.close()
    summary = profiler.summary()
    phase = summary["phases"][0]
    assert phase["tree_peak_gb"] > phase["peak_gb"] + 0.10
    assert summary["peak_process_tree_footprint_gb"] >= phase["tree_peak_gb"]
