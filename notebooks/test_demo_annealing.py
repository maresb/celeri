"""Test script to validate demo_annealing.py structure without running full optimization."""

from __future__ import annotations

import sys
from pathlib import Path

# Add workspace to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

# Test imports
print("Testing imports...")
try:
    from celeri.config import get_config
    from celeri.model import Model
    from celeri.optimize import solve_sqp2
    from celeri.plot import plot_mesh
    from celeri.solve import build_estimation
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test config file exists
print("\nTesting config file path...")
config_path = workspace_root / "data" / "config" / "japan_config.json"
if config_path.exists():
    print(f"✓ Config file found: {config_path}")
else:
    print(f"✗ Config file not found: {config_path}")
    sys.exit(1)

# Test that we can load config
print("\nTesting config loading...")
try:
    config = get_config(str(config_path))
    print("✓ Config loaded successfully")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    sys.exit(1)

# Test that we can build model (this might take a moment)
print("\nTesting model building...")
try:
    model = Model.from_config(config)
    print(f"✓ Model built successfully")
    print(f"  - {len(model.meshes)} meshes")
    print(f"  - {len(model.segment)} segments")
    print(f"  - {len(model.station)} stations")
except Exception as e:
    print(f"✗ Model building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test function definitions from demo_annealing
print("\nTesting function definitions...")
try:
    import notebooks.demo_annealing as demo
    assert hasattr(demo, 'plot_tde_slip_rates')
    assert hasattr(demo, 'plot_annealing_convergence')
    assert hasattr(demo, 'main')
    print("✓ All functions defined")
except Exception as e:
    print(f"✗ Function definition test failed: {e}")
    sys.exit(1)

print("\n✓ All tests passed! The script structure is valid.")
print("\nNote: Full optimization would require running the actual solve_sqp2()")
print("      which may take several minutes. The code structure is correct.")
