"""One report of what a model version can and can't do, measured the same way every time.

    python analysis/capabilities.py --model runs/nano-final --name v0.1
    python analysis/capabilities.py --compare          # all saved reports side by side

For the shipped ONNX file (int8 + 4-bit embeddings, the runtime's guard on) it measures:
- Atelier's own eval (tests/decisionsEval.test.ts) through runtime/node, per decision: right with the model, right
  when sure, unsure;
- held-out agreement with the teacher per decision and per language (the trainer's 5% split, eval cases left out);
- probes it never trained on: 20 languages, open questions (new kinds of questions about new text), held-out
  question families written for the browser, many tab groups.
Writes analysis/results/capabilities-<name>.json and docs/capabilities/<name>.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
import sys
import time
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
from format import encode, fixed_kind, option_labels, state_text  # noqa: E402
from readers import verdict  # noqa: E402
from train import read_jsonl, without_eval, eval_question_keys, question_key  # noqa: E402

RESULTS = ROOT / "analysis" / "results"
DOCS = ROOT / "docs" / "capabilities"
POOLS = [ROOT / "data" / "teacher" / f"pool-v{i}.flash.jsonl" for i in range(1, 7)]
OPEN_POOLS = [ROOT / "data" / "teacher" / "pool-open.flash.jsonl"]
PROBES = {
    "languages": ROOT / "data" / "teacher" / "probe-lang.flash.jsonl",
    "open_unseen": ROOT / "data" / "teacher" / "probe-open.flash.jsonl",
    "open_browser_families": ROOT / "data" / "teacher" / "open-heldout.flash.jsonl",
    "open_browser_families_2": ROOT / "data" / "teacher" / "gen3-heldout.flash.jsonl",
    "open_browser_families_3": ROOT / "data" / "teacher" / "gen4-heldout.flash.jsonl",
    "tabs_many_groups": ROOT / "data" / "teacher" / "probe-tabs.flash.jsonl",
}
GUARD = {"min_latin": 0.7, "max_bytes": 0.1, "min_letters": 8}


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


class Model:
    def __init__(self, folder: Path, file: str) -> None:
        self.tok = AutoTokenizer.from_pretrained(str(folder))
        self.temps = json.loads((folder / "calibration.json").read_text()) if (folder / "calibration.json").exists() else {}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(folder / file), options, providers=["CPUExecutionProvider"])
        self.byte_ids = {i for t, i in self.tok.get_vocab().items() if t.startswith("<0x") and t.endswith(">")}

    def unreadable(self, state) -> bool:
        text = state_text(state)
        letters = [c for c in text if unicodedata.category(c).startswith("L")]
        if len(letters) >= GUARD["min_letters"] and sum("LATIN" in unicodedata.name(c, "") for c in letters) / len(letters) < GUARD["min_latin"]:
            return True
        ids = self.tok.encode(text, add_special_tokens=False)
        return bool(ids) and sum(i in self.byte_ids for i in ids) / len(ids) > GUARD["max_bytes"]

    def probs(self, state, key: str, q: dict) -> tuple[np.ndarray, str]:
        e = encode(self.tok, state, key, q, 512)
        ids = np.array([e["input_ids"]], dtype=np.int64)
        scores = self.session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
        return softmax(scores[e["markers"]] / float(self.temps.get(e["kind"], 1.0))), e["kind"]


class MiniModel:
    """atelier-mini through llama.cpp (student/serve_mini.py): no language guard, its own calibrated temperatures."""

    def __init__(self, model_dir: Path, gguf: Path) -> None:
        from serve_mini import Mini

        calibration = model_dir / "calibration.json"
        self.mini = Mini(model_dir / "merged", gguf, 6, json.loads(calibration.read_text()) if calibration.exists() else None)
        # Raw scores per question survive the run (analysis/cascade.py and later reports reuse them).
        self.cache_file = model_dir / f"scores-{gguf.stem}.json"
        self.scores: dict[str, list[float]] = json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}
        self.cache: dict[str, tuple[np.ndarray, str]] = {}

    def unreadable(self, state) -> bool:
        return False

    def probs(self, state, key: str, q: dict) -> tuple[np.ndarray, str]:
        h = hashlib.md5(json.dumps([state, key, q], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if h not in self.cache:
            if h not in self.scores:
                enc = self.mini.encoder.plan({"state": state, "questions": {key: q}}, None, "separate")[0]
                self.scores[h] = (self.mini.readout @ self.mini.hidden(enc.input_ids.tolist())).tolist()[: len(option_labels(q))]
            kind = fixed_kind(key, q) or "open"
            self.cache[h] = (softmax(np.array(self.scores[h]) / float(self.mini.temperatures.get(kind, 1.0))), kind)
        return self.cache[h]

    def close(self) -> None:
        self.cache_file.write_text(json.dumps(self.scores))
        self.mini.close()


def score_rows(model: Model, rows: list[dict], group, guard: bool = True) -> dict:
    """Agreement with the teacher's verdict per group: agree, conflict (both sure, differ), overreach (student sure,
    teacher not), underreach, declined (guard), n."""
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        declined = guard and model.unreadable(r["request"]["state"])
        for key, q in r["request"]["questions"].items():
            kind, labels = fixed_kind(key, q) or "open", option_labels(q)
            v_t = verdict(kind, labels, softmax(np.array(r["logits"][key]) / r["temperature"]).tolist())
            v_s = None
            if not declined:
                p, _ = model.probs(r["request"]["state"], key, q)
                v_s = verdict(kind, labels, p.tolist())
            for name in (group(r), "ALL"):
                s = out[name]
                s["n"] += 1
                s["agree"] += v_s == v_t
                s["conflict"] += v_s is not None and v_t is not None and v_s != v_t
                s["overreach"] += v_s is not None and v_t is None
                s["underreach"] += v_s is None and v_t is not None and not declined
                s["declined"] += declined
                s["decided"] += v_s is not None
                s["decided_right"] += v_s is not None and v_s == v_t
    result = {}
    for g, s in out.items():
        row = {k: (round(v / s["n"], 4) if k != "n" else int(v)) for k, v in s.items() if k not in ("decided", "decided_right")}
        # Coverage: how often it decides; precision: when it decides, how often the teacher decides the same.
        row["coverage"] = round(s["decided"] / s["n"], 4)
        row["precision"] = round(s["decided_right"] / s["decided"], 4) if s["decided"] else None
        result[g] = row
    return result


def held_out(paths: list[Path]) -> list[dict]:
    rows = without_eval([r for p in paths if p.exists() for r in read_jsonl(p)])
    held = [r for r in rows if int(hashlib.md5(json.dumps(r["request"], sort_keys=True).encode()).hexdigest(), 16) % 20 == 0]
    banned = eval_question_keys()
    return [r for r in held if not any(question_key(q) in banned for q in r["request"]["questions"].values())]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def atelier_eval(folder: Path, file: str, name: str) -> dict:
    """Atelier's own eval through runtime/node (guard on, open questions declined, as shipped)."""
    port = free_port()
    server = subprocess.Popen(["node", "--no-warnings", "--experimental-strip-types", "src/server.ts", str(folder.resolve()), str(port), "4", file, "decline"],
                              cwd=ROOT / "runtime" / "node", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"runtime/node did not start for {folder}")
        out = RESULTS / f"atelier-eval-{name}.txt"
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "tools" / "atelier_eval.ps1"), "-Url",
                        f"http://127.0.0.1:{port}", "-Model", "atelier-nano", "-Out", str(out)], capture_output=True, timeout=1200)
    finally:
        server.terminate()
    return parse_atelier_eval(out)


def parse_atelier_eval(out: Path) -> dict:
    text = re.sub(r"\s*\n\s*", " ", out.read_text(encoding="utf-8", errors="replace"))
    result = {}
    for m in re.finditer(r"atelier-[\w-]+ \(custom\) (\w+): rule alone (\d+)/(\d+), with the model (\d+)/\d+; (\d+)/(\d+) right when sure.*?unsure on (\d+)", text):
        use, rule, n, model_right, sure_right, sure, unsure = m.groups()
        result[use] = {"n": int(n), "rule": int(rule), "model": int(model_right), "sure_right": int(sure_right), "sure": int(sure), "unsure": int(unsure)}
    m = re.search(r"Tests\s+(?:(\d+) failed \| )?(\d+) passed", text)
    result["_tests"] = {"failed": int(m.group(1) or 0), "passed": int(m.group(2))} if m else {}
    return result


def run(model_dir: Path, name: str, file: str, gguf: Path | None = None, eval_file: Path | None = None) -> dict:
    if gguf:
        model, size = MiniModel(model_dir, gguf), gguf.stat().st_size
    else:
        folder = model_dir / "onnx" if (model_dir / "onnx").is_dir() else model_dir
        model, size = Model(folder, file), (folder / file).stat().st_size
    report: dict = {"name": name, "model": str(model_dir), "file": gguf.name if gguf else file, "mb": round(size / 1e6, 1),
                    "date": time.strftime("%Y-%m-%d %H:%M"), "guard": not gguf}
    report["atelier_eval"] = parse_atelier_eval(eval_file) if eval_file else atelier_eval(folder, file, name)
    held = held_out(POOLS)
    report["held_out_by_decision"] = score_rows(model, held, lambda r: r["meta"]["use"])
    report["held_out_by_language"] = score_rows(model, held, lambda r: r["meta"].get("lang", "?"))
    open_held = held_out(OPEN_POOLS)
    if open_held:
        report["held_out_open_by_source"] = score_rows(model, open_held, lambda r: r["meta"]["source"])
    for probe, path in PROBES.items():
        if path.exists():
            group = (lambda r: r["meta"].get("lang", "?")) if probe == "languages" else (lambda r: r["meta"].get("family") or r["meta"]["source"])
            report[f"probe_{probe}"] = score_rows(model, read_jsonl(path), group)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"capabilities-{name}.json").write_text(json.dumps(report, indent=1))
    write_markdown(report)
    if gguf:
        model.close()
    return report


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_markdown(r: dict) -> None:
    lines = [f"# Capabilities: {r['name']}", "", f"`{r['model']}` ({r['file']}, {r['mb']} MB), measured {r['date']}.", "",
             "## Atelier's own eval", "", "| Decision | Rules alone | With the model | Right when sure | Unsure |", "|---|---|---|---|---|"]
    for use, e in r["atelier_eval"].items():
        if use.startswith("_"):
            continue
        lines.append(f"| {use} | {e['rule']}/{e['n']} | {e['model']}/{e['n']} | {e['sure_right']}/{e['sure']} | {e['unsure']} |")
    t = r["atelier_eval"].get("_tests", {})
    lines += ["", f"Tests: {t.get('passed', '?')} passed, {t.get('failed', '?')} failed.", ""]

    def table(title: str, data: dict, key_name: str) -> None:
        lines.extend([f"## {title}", "", f"| {key_name} | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |",
                      "|---|---|---|---|---|---|---|---|"])
        for k, s in sorted(data.items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
            precision = pct(s["precision"]) if s.get("precision") is not None else "-"
            lines.append(f"| {k} | {s['n']} | {pct(s['agree'])} | {pct(s.get('coverage', 0))} | {precision} | {pct(s['conflict'])} | {pct(s['overreach'])} | {pct(s['declined'])} |")
        lines.append("")

    table("Held out, per decision", r["held_out_by_decision"], "Decision")
    table("Held out, per language", r["held_out_by_language"], "Language")
    if "held_out_open_by_source" in r:
        table("Held out, open questions per source", r["held_out_open_by_source"], "Source")
    for key, title in (("probe_languages", "Probe: 20 languages (never seen)"), ("probe_open_unseen", "Probe: new kinds of questions about new text"),
                       ("probe_open_browser_families", "Probe: held-out browser question families"), ("probe_open_browser_families_2", "Probe: held-out browser question families, second set"), ("probe_open_browser_families_3", "Probe: held-out browser question families, third set (gen4)"), ("probe_tabs_many_groups", "Probe: 18-28 tab groups")):
        if key in r:
            data = r[key]
            if len(data) > 25:  # many families: show the overall and the 10 worst
                worst = sorted((k for k in data if k != "ALL"), key=lambda k: -data[k]["conflict"])[:10]
                data = {k: data[k] for k in worst + ["ALL"]}
                title += " (the 10 with most contradictions, and all)"
            table(title, data, "Group")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / f"{r['name']}.md").write_text("\n".join(lines), encoding="utf-8")


def compare() -> None:
    reports = sorted((json.loads(p.read_text()) for p in RESULTS.glob("capabilities-*.json")), key=lambda r: r["date"])
    if not reports:
        return
    lines = ["# Capabilities across versions", "", "| | " + " | ".join(r["name"] for r in reports) + " |", "|---|" + "---|" * len(reports)]
    lines.append("| Size (MB) | " + " | ".join(str(r["mb"]) for r in reports) + " |")
    uses = sorted({u for r in reports for u in r["atelier_eval"] if not u.startswith("_")})
    for use in uses:
        cells = []
        for r in reports:
            e = r["atelier_eval"].get(use)
            cells.append(f"{e['model']}/{e['n']} ({e['sure_right']}/{e['sure']} sure)" if e else "")
        lines.append(f"| Atelier eval: {use} | " + " | ".join(cells) + " |")
    for key, label in (("held_out_by_decision", "Held out"), ("probe_open_unseen", "New kinds of questions"),
                       ("probe_open_browser_families", "Held-out browser families"), ("probe_open_browser_families_2", "Held-out browser families (2nd set)"), ("probe_open_browser_families_3", "Held-out browser families (3rd set)"), ("probe_languages", "20 languages (nano guarded, mini not)")):
        cells = [(f"{pct(r[key]['ALL']['agree'])} agree, decides {pct(r[key]['ALL'].get('coverage', 0))}, right when deciding "
                  f"{pct(r[key]['ALL']['precision']) if r[key]['ALL'].get('precision') is not None else '-'}") if key in r else "" for r in reports]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--file", default="model.emb4.int8.onnx")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--gguf", type=Path, help="atelier-mini: score through llama.cpp with this GGUF")
    parser.add_argument("--atelier-eval-file", type=Path, help="parse this Atelier eval output instead of running runtime/node")
    args = parser.parse_args()
    if args.model:
        report = run(args.model, args.name or args.model.name, args.file, args.gguf, args.atelier_eval_file)
        print(json.dumps({k: v for k, v in report.items() if k in ("name", "mb", "atelier_eval")}, indent=1))
    if args.compare or not args.model:
        compare()


if __name__ == "__main__":
    main()
