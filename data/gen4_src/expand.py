# Expands compact sources (src_N.txt) into data/gen4/world_N.jsonl
import json, sys, glob, os, re, collections
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(os.path.dirname(here), "gen4")
LANGS = set("ja zh ko ru uk ar he fa hi bn ta th vi id tr el pl it pt sw en".split())
existing = set(open(os.path.join(os.path.dirname(here), "existing_families.txt"), encoding="utf-8").read().split())

def crit(typ, s):
    if s is None: return None
    parts = [p.strip() for p in s.split(";;") if p.strip()]
    if typ == "score": return parts
    d = {}
    for p in parts:
        k, _, v = p.partition("=")
        d[k.strip()] = v.strip() or None
    return d

def parse(path):
    fams = []; fam = None; item = None
    for raw in open(path, encoding="utf-8").read().split("\n"):
        line = raw.rstrip()
        if line.startswith("@@ "):
            name, typ = [x.strip() for x in line[3:].split("|")]
            fam = {"name": name, "type": typ, "q": None, "c": None, "items": []}; fams.append(fam); item = None
        elif line.startswith("Q: ") and item is None:
            fam["q"] = line[3:].strip()
        elif line.startswith("C: ") and item is None:
            fam["c"] = line[3:].strip()
        elif line.startswith("## "):
            head, _, lq = line[3:].partition("|")
            lang, _, ans = head.strip().partition("/")
            item = {"lang": lang.strip(), "ans": ans.strip(), "lq": lq.strip() or None, "lc": None, "lines": []}
            fam["items"].append(item)
        elif line.startswith("LC: ") and item is not None and not item["lines"]:
            item["lc"] = line[4:].strip()
        elif item is not None:
            item["lines"].append(line)
    return fams

def state_of(lines):
    while lines and not lines[-1].strip(): lines = lines[:-1]
    while lines and not lines[0].strip(): lines = lines[1:]
    ne = [l for l in lines if l.strip()]
    if ne and all(re.match(r"^~[a-z_]+: ", l) for l in ne):
        d = {}
        for l in ne:
            k, _, v = l[1:].partition(": ")
            d[k] = v.strip()
        return d
    return "\n".join(lines)

langs = collections.Counter(); qlocal = 0; total = 0; seen = set(); problems = []
for path in sorted(glob.glob(os.path.join(here, "src_*.txt")), key=lambda p: int(re.findall(r"(\d+)\.txt", p)[0])):
    n = re.findall(r"(\d+)\.txt", path)[0]
    rows = []
    for fam in parse(path):
        nm, typ = fam["name"], fam["type"]
        if nm in existing or nm in seen: problems.append(f"dup family {nm}")
        seen.add(nm)
        if typ not in ("choice", "noul", "score"): problems.append(f"bad type {nm}")
        if len(fam["items"]) != 8: problems.append(f"{nm}: {len(fam['items'])} items")
        base = crit(typ, fam["c"])
        bal = collections.Counter(i["ans"] for i in fam["items"])
        if typ == "noul" and (bal.get("y", 0) < 3 or bal.get("n", 0) < 3): problems.append(f"{nm}: balance {dict(bal)}")
        if typ == "choice":
            bad = [a for a in bal if a not in base]
            if bad: problems.append(f"{nm}: unknown answer keys {bad}")
            if max(bal.values()) > 4: problems.append(f"{nm}: balance {dict(bal)}")
        for it in fam["items"]:
            if it["lang"] not in LANGS: problems.append(f"{nm}: lang {it['lang']}")
            st = state_of(it["lines"])
            if not st: problems.append(f"{nm}: empty state {it['lang']}")
            q = {"type": typ, "instructions": it["lq"] or fam["q"]}
            if it["lq"]:
                qlocal += 1
                if it["lc"]: q["criteria"] = crit(typ, it["lc"])
                elif typ == "choice": q["criteria"] = {k: None for k in base}
                elif typ == "score": q["criteria"] = base
            elif base is not None:
                q["criteria"] = base
            rows.append({"family": nm, "lang": it["lang"], "state": st, "question": q})
            langs[it["lang"]] += 1; total += 1
    assert len(rows) <= 120, (path, len(rows))
    with open(os.path.join(out, f"world_{n}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"world_{n}.jsonl: {len(rows)} lines")
print("families:", len(seen), "total:", total, "local-language questions:", qlocal)
print("langs:", ", ".join(f"{k} {v}" for k, v in langs.most_common()))
print("PROBLEMS:" if problems else "no problems", *problems, sep="\n  ")
