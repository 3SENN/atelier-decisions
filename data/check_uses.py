# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Checks data/atelier_uses.py against requests Atelier really sent (a teacher server log) and the eval fixtures.

    python data/check_uses.py data/teacher/eval-flash.jsonl ../atelier/tests/fixtures/decisions-eval.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atelier_uses as U  # noqa: E402

log, fixtures = Path(sys.argv[1]), Path(sys.argv[2])
with log.open(encoding="utf-8") as f:  # not splitlines(): JSON strings may hold U+2028
    sent = [json.loads(line)["request"] for line in f if line.strip()]
cases = json.loads(fixtures.read_text(encoding="utf-8"))
skill_list = cases["skills"]["skills"]

built = []
built += [U.needs_you(c["reply"]) for c in cases["needsYou"]]
built += [U.dictation(c["spoken"], c["field"]) for c in cases["dictation"]]
built += [U.agent_click(c["name"], c["role"], c["page"], c["url"]) for c in cases["agentClicks"]]
built += [U.search_or_ask(c["text"]) for c in cases["searchOrAsk"]]
built += [U.voice_target(c["spoken"], c["field"], c["value"]) for c in cases["voiceTarget"]]
built += [U.effort(c["message"], c["previous"]) for c in cases["effort"]]
built += [U.memory(c["transcript"]) for c in cases["memory"]]
built += [U.tabs(c["title"], c["url"], c["groups"]) for c in cases["tabs"]]
built += [U.skills(c["message"], skill_list) for c in cases["skills"]["messages"]]
lab = cases["labeling"]
built += [U.label_item(i["text"], lab["questions"], lab["context"]) for i in lab["items"]]

canon = lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False)  # noqa: E731
sent_set = {canon(r) for r in sent}
missing = [r for r in built if canon(r) not in sent_set]
print(f"{len(built)} requests built from the fixtures; {len(built) - len(missing)} identical to what Atelier sent")
for r in missing[:5]:
    use = U.use_of(r)
    near = [s for s in sent if U.use_of(s) == use]
    print(f"  MISMATCH {use}: built {canon(r)[:300]}")
    if near:
        print(f"           sent  {canon(near[0])[:300]}")
sys.exit(1 if missing else 0)
