"""Does shade on the parchment drive the false alarms on clean single capsules? (Nov 6, sel02)

For every clean leak_capsule picture of a block cut, the median brightness of its four corners (bare parchment) against
d3's probability of "leaker", as a correlation and by brightness band.

    .\\run.ps1 python scripts/capsule_shade_check.py output/capsules/sel02/blocks/n1
"""
import json, sys
import numpy as np
from PIL import Image
D = sys.argv[1] if len(sys.argv) > 1 else "output/capsules/sel02/blocks/n1"
rows = [json.loads(l) for l in open(D + "/decisions.jsonl")]
arms = {a["index_global"]: a for a in (json.loads(l) for l in open(D + "/arms.jsonl")) if a["arm"] == "d3-27b-vision" and not a.get("error")}
xs, ps, fa = [], [], []
for i, r in enumerate(rows):
    if r["truth"] != "good" or i not in arms:
        continue
    im = np.asarray(Image.open(D + "/" + r["frame"]).convert("L"), dtype=float)
    h, w = im.shape
    corners = np.concatenate([im[:h // 5, :w // 5].ravel(), im[:h // 5, -w // 5:].ravel(), im[-h // 5:, :w // 5].ravel(), im[-h // 5:, -w // 5:].ravel()])
    b = float(np.median(corners))
    p = arms[i]["probs"]["leaker"]
    xs.append(b); ps.append(p); fa.append(p > 0.5)
xs, ps, fa = np.array(xs), np.array(ps), np.array(fa)
print("clean single capsules:", len(xs), " corr(paper brightness, d3 leaker prob) = %.2f" % np.corrcoef(xs, ps)[0, 1])
for lo, hi in ((0, 150), (150, 180), (180, 200), (200, 220), (220, 256)):
    m = (xs >= lo) & (xs < hi)
    if m.sum():
        print(f"  paper brightness {lo:3d}-{hi:3d}: n {m.sum():4d}  false alarm {fa[m].mean():.2f}  mean leaker prob {ps[m].mean():.2f}")
