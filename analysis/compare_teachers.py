"""Agreement between two teachers' logits on the same requests: top answer and Atelier verdict, per use."""
import json, sys
from collections import defaultdict
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student")); sys.path.insert(0, str(ROOT / "data"))
from format import fixed_kind, option_labels
from readers import verdict
from atelier_uses import use_of

a = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
b = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
st = defaultdict(lambda: defaultdict(float))
diffs = []
for x, y in zip(a, b, strict=True):
    req = x["request"]
    for key, q in req["questions"].items():
        pa = (torch.tensor(x["logits"][key]) / x["temperature"]).softmax(-1)
        pb = (torch.tensor(y["logits"][key]) / y["temperature"]).softmax(-1)
        kind, labels = fixed_kind(key, q) or "open", option_labels(q)
        va, vb = verdict(kind, labels, pa.tolist()), verdict(kind, labels, pb.tolist())
        for name in (use_of(req), "ALL"):
            s = st[name]; s["n"] += 1; s["top1"] += float(pa.argmax() == pb.argmax())
            s["verdict"] += float(va == vb); s["dp"] += float((pa - pb).abs().max())
            s["sureA"] += va is not None; s["sureB"] += vb is not None
            s["conflict"] += float(va is not None and vb is not None and va != vb)
        if va != vb:
            diffs.append((use_of(req), json.dumps(req["state"], ensure_ascii=False)[:90], va, vb))
print(f"{'use':<12} {'n':>4} {'top1':>6} {'verdict':>8} {'|dp|':>6} {'sure A/B':>10} {'conflict':>8}")
for k in sorted(st, key=lambda k: (k == "ALL", k)):
    s = st[k]; n = s["n"]
    print(f"{k:<12} {int(n):>4} {s['top1']/n:>6.3f} {s['verdict']/n:>8.3f} {s['dp']/n:>6.3f} {s['sureA']/n:>4.2f}/{s['sureB']/n:<4.2f} {s['conflict']/n:>8.3f}")
print("\nverdicts that differ (A, B):")
for d in diffs[:25]: print("  ", d)
