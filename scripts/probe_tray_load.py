"""Does Genesis's net contact force on a fixed tray read a resting part's weight? Oct 20 probe behind VALIDATION "Planned: in_bin from cell sensors".

    .venv\\Scripts\\python.exe scripts\\probe_tray_load.py    (GPU, config/handoff.yaml; about 1 min)
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from handoff.scene import Layout, make_episode, make_scene, place_episode, tray_box_names
from handoff.sim.genesis_sim import GenesisSimulator

layout = Layout()
sim = GenesisSimulator(str(ROOT / "config" / "handoff.yaml"))
t0 = time.time(); sim.build(make_scene(layout)); print(f"built {time.time()-t0:.0f} s", flush=True)
ep = make_episode(layout, 0); place_episode(sim, layout, ep); sim.settle(0.5)
expected = layout.part_rho * float(np.prod(layout.part_size)) * 9.81
def loads():
    return {b: sim.tray_load(tray_box_names(b)) for b in layout.bin_names}
print("expected per part N", round(expected, 4))
print("empty trays", {k: round(v, 4) for k, v in loads().items()})
part = ep.parts[0]
g = layout.bin_region("good")
cases = [("upright in good", (g.cx, g.cy, g.floor_z + 0.045), 0.0)]
for label, pos, yaw in cases:
    sim.set_object_pose(part, pos, yaw); sim.settle(1.5)
    xs = []
    for _ in range(50):
        sim.step(1); xs.append(loads())
    good = np.array([x["good"] for x in xs]); rew = np.array([x["rework"] for x in xs])
    print(label, "good mean", round(good.mean(), 4), "sd", round(good.std(), 4), "min", round(good.min(), 4), "max", round(good.max(), 4), "| rework mean", round(rew.mean(), 4))
# dropped from 10 cm, lying down, against a wall
sim.set_object_pose(part, (g.cx + 0.05, g.cy, g.floor_z + 0.15), 37.0); sim.settle(2.0)
xs = [ (sim.step(1), loads())[1]["good"] for _ in range(50)]
print("dropped 10 cm", "mean", round(float(np.mean(xs)), 4), "sd", round(float(np.std(xs)), 4), "pos", np.round(sim.object_state(part).pos, 3))
# second part on top of the tray, two parts
p2 = ep.parts[1]
sim.set_object_pose(p2, (g.cx - 0.05, g.cy - 0.05, g.floor_z + 0.045), 0.0); sim.settle(1.5)
xs = [ (sim.step(1), loads())[1]["good"] for _ in range(50)]
print("two parts", "mean", round(float(np.mean(xs)), 4), "sd", round(float(np.std(xs)), 4))
# part on the table, outside both trays
sim.set_object_pose(part, (0.9, -0.2, layout.table_top + 0.045), 0.0); sim.settle(1.0)
print("one moved to table", {k: round(v, 4) for k, v in loads().items()})
