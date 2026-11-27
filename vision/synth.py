# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pictures drawn with their answers: buttons that are only an icon, documents, and pages in a state. No model
labels anything: each picture is of the kind it was drawn as.

Photos (vision/photos.py) teach what things look like. They can't teach what a browser needs to tell apart on a
screen, and nobody has labelled that at scale. So it is drawn, with a real browser engine (vision/render.mjs):

- control   a button with only an icon: 63 jobs (cart, search, close, ...) from nine open icon sets (vision/icons.py),
            in the sizes, shapes, colours and themes sites use, at 1x, 2x and 3x. The picture is the button's own
            box, as Atelier would cut it out of a page. And buttons whose icon is none of those jobs (a logo, an
            anchor): in training only ever a "no", and from the held-out sets the test of whether it names a button
            it should leave alone.
- document  22 kinds of paper and screen (a receipt, an invoice, a ticket, a letter, an identity card marked as a
            specimen, a menu, a calendar, a spreadsheet, a chat, a handwritten note, ...), with made-up content in
            English, Dutch, German, French or Spanish, as a flat scan or screenshot or as a photo taken at an
            angle on a table.
- page      a whole page in one of 8 states (ready, a cookie notice, a pop-up, a sign-in wall, an error, still
            loading, a robot check, a paywall) on six layouts of site, at desktop and phone sizes, with real photos
            (Open Images) where a site has pictures.

Held out (never trained on): 8 button jobs, 5 kinds of document, 2 page states; three icon sets, two page layouts,
and for every kind of document and page state one language; and a tenth of the rest.

    .venv/bin/python -I vision/synth.py render --family control document page      # draws and packs everything
    .venv/bin/python -I vision/synth.py sheet --out runs/vision/sheet                # a few of each, to look at

People, shops and companies are made up (step/synth.py's generator, .test addresses). Stored like the photos: a
document or page as the 256 x 256 square the tower reads, a button at its own size.
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

import icons as icon_sets  # noqa: E402
from questions import LANG_WEIGHTS, LANGS, Family, language  # noqa: E402
from synth_words import (AIRPORTS, BANKS, COMPANIES, DISHES, FILLER, GROCERIES, HELD_OUT_DOCUMENTS, HELD_OUT_PAGES, MONTHS, PAGE_OTHER, PAGES, SERVICES, SHOPS,  # noqa: E402
                         WEEKDAYS, documents, t)

CACHE = ROOT / "data" / "cache" / "vision"


def _step():
    spec = importlib.util.spec_from_file_location("atelier_step_synth", ROOT / "step" / "synth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STEP = _step()
esc = html_lib.escape

SANS = ["Helvetica Neue", "Arial", "Verdana", "Avenir Next", "Gill Sans", "Trebuchet MS", "Futura", "Optima", "system-ui", "Tahoma"]
SERIF = ["Georgia", "Times New Roman", "Palatino", "Baskerville", "Didot", "Hoefler Text", "Cochin", "Charter"]
MONO = ["Menlo", "Courier New", "Monaco", "Andale Mono", "Courier"]
HAND = ["Bradley Hand", "Marker Felt", "Noteworthy", "Chalkduster", "Snell Roundhand", "Apple Chancery", "Brush Script MT", "Comic Sans MS"]


def fam(rng: random.Random, *pools: list[str]) -> str:
    pool = [f for p in pools for f in p]
    generic = "monospace" if pools[0] is MONO else "cursive" if pools[0] is HAND else "serif" if pools[0] is SERIF else "sans-serif"
    return f'"{rng.choice(pool)}", {generic}'  # double quotes: it goes inside single-quoted style attributes


def hsl(h: float, s: float, light: float, a: float | None = None) -> str:
    return f"hsl({h % 360:.0f} {s:.0f}% {light:.0f}%" + (f" / {a})" if a is not None else ")")


class Ctx:
    """One picture's random choices: its language, a made-up person, an accent colour, its fonts."""

    def __init__(self, rng: random.Random, lang: str, photos: list[str] | None = None) -> None:
        self.rng, self.lang = rng, lang
        self.p = STEP.person(rng, lang)
        self.hue = rng.uniform(0, 360)
        self.acc = hsl(self.hue, rng.uniform(50, 85), rng.uniform(34, 52))
        self.sans, self.serif, self.mono = fam(rng, SANS), fam(rng, SERIF), fam(rng, MONO)
        self.photos = photos or []

    def t(self, key: str) -> str:
        return t(key, self.lang)

    def name(self) -> str:
        return f"{self.p['first']} {self.p['last']}"

    def address(self) -> list[str]:
        p = self.p
        street = f"{p['no']} {p['street']}" if self.lang in ("en", "fr") else f"{p['street']} {p['no']}"
        return [street, f"{p['postcode']} {p['city']}"]

    def words(self, n: int) -> str:
        return " ".join(self.rng.choice(FILLER[self.lang]) for _ in range(n))

    def sentence(self, lo: int = 6, hi: int = 14) -> str:
        s = self.words(self.rng.randint(lo, hi))
        return s[0].upper() + s[1:] + "."

    def para(self, sentences: int) -> str:
        return " ".join(self.sentence() for _ in range(sentences))

    def title(self, lo: int = 2, hi: int = 5) -> str:
        s = self.words(self.rng.randint(lo, hi))
        return s[0].upper() + s[1:]

    def money(self, value: float) -> str:
        whole, cents = divmod(round(value * 100), 100)
        if self.lang == "en":
            return f"£{whole}.{cents:02d}"
        return f"€ {whole},{cents:02d}" if self.lang == "nl" else f"{whole},{cents:02d} €"

    def date(self) -> str:
        d, m, y = self.rng.randint(1, 28), self.rng.randint(1, 12), self.rng.choice([2024, 2025, 2026])
        sep = {"nl": "-", "de": "."}.get(self.lang, "/")
        return f"{d:02d}{sep}{m:02d}{sep}{y}"

    def photo(self, style: str = "") -> str:
        """A real photo where a site would have one, or a block of colour when there are none to use."""
        if self.photos:
            return f"<img src='{self.rng.choice(self.photos)}' style='width:100%;height:100%;object-fit:cover;display:block;{style}'>"
        return f"<div style='width:100%;height:100%;background:linear-gradient({self.rng.randint(0, 360)}deg,{hsl(self.rng.uniform(0, 360), 45, 62)},{hsl(self.rng.uniform(0, 360), 50, 42)});{style}'></div>"


def bars(rng: random.Random, w: int, h: int, color: str = "#111") -> str:
    """A barcode's look: bars of random widths."""
    x, rects = 0.0, []
    while x < w:
        bw = rng.choice([1, 1, 2, 3]) * w / 120
        if rng.random() < 0.55:
            rects.append(f"<rect x='{x:.1f}' y='0' width='{bw:.1f}' height='{h}'/>")
        x += bw
    return f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' fill='{color}' style='display:block'>{''.join(rects)}</svg>"


def qr(rng: random.Random, size: int, n: int = 25, color: str = "#111") -> str:
    """A QR code's look: random modules and the three corner squares."""
    cells = []
    for y in range(n):
        for x in range(n):
            corner = (x < 7 and y < 7) or (x >= n - 7 and y < 7) or (x < 7 and y >= n - 7)
            if corner:
                cx, cy = (x if x < 7 else x - (n - 7)), (y if y < 7 else y - (n - 7))
                on = cx in (0, 6) or cy in (0, 6) or (2 <= cx <= 4 and 2 <= cy <= 4)
            else:
                on = rng.random() < 0.48
            if on:
                cells.append(f"<rect x='{x}' y='{y}' width='1.02' height='1.02'/>")
    return f"<svg width='{size}' height='{size}' viewBox='0 0 {n} {n}' fill='{color}' style='display:block'>{''.join(cells)}</svg>"


def table(rows: list[list[str]], widths: list[str], style: str = "", head: str = "", cell: str = "") -> str:
    out = [f"<table style='border-collapse:collapse;width:100%;{style}'>"]
    for i, row in enumerate(rows):
        css = head if i == 0 and head else cell
        out.append("<tr>" + "".join(f"<td style='width:{w};padding:5px 6px;{css}{'text-align:right;' if j and j == len(row) - 1 else ''}'>{v}</td>" for j, (v, w) in enumerate(zip(row, widths))) + "</tr>")
    return "".join(out) + "</table>"


# ── documents ────────────────────────────────────────────────────────────────────────────────────────────────────────
# Each returns (the sheet's html, its width in CSS pixels, what it is on: paper, screen or phone).

def doc_receipt(c: Ctx):
    rng = c.rng
    w = rng.choice([250, 270, 290, 310])
    items = [(g, rng.uniform(0.6, 14)) for g in rng.sample(GROCERIES[c.lang], rng.randint(3, 12))]
    total = sum(v for _, v in items)
    line = lambda a, b, s="": f"<div style='display:flex;justify-content:space-between;{s}'><span>{esc(a)}</span><span>{b}</span></div>"
    rule = f"<div style='border-top:1px {rng.choice(['dashed', 'solid', 'dotted'])} #333;margin:7px 0'></div>"
    body = "".join(line(g if rng.random() < 0.7 else f"{rng.randint(2, 4)} x {g}", c.money(v)) for g, v in items)
    head = f"<div style='text-align:center;font-weight:bold;font-size:{rng.randint(15, 20)}px'>{esc(rng.choice(SHOPS))}</div><div style='text-align:center'>{esc(' · '.join(c.address()))}</div>"
    tail = line(c.t("subtotal"), c.money(total / 1.21)) + line(f"{c.t('vat')} 21%", c.money(total - total / 1.21)) + line(c.t("total"), c.money(total), "font-weight:bold;font-size:1.25em")
    pay = line(rng.choice([c.t("card"), c.t("cash")]), c.money(total)) + f"<div style='margin-top:6px'>{c.date()} {rng.randint(8, 21)}:{rng.randint(0, 59):02d} #{rng.randint(1000, 99999)}</div>"
    code = f"<div style='margin:10px auto 4px;width:{w - 80}px'>{bars(rng, w - 80, 34)}</div>" if rng.random() < 0.6 else ""
    return (f"<div style='width:{w}px;padding:16px 14px;font:{rng.randint(10, 12)}px/1.5 {c.mono};color:#222;background:{rng.choice(['#fff', '#fdfdf8', '#f7f6f0'])}'>"
            f"{head}{rule}{body}{rule}{tail}{rule}{pay}{code}<div style='text-align:center;margin-top:8px'>{esc(c.t('thanks'))}</div></div>"), w + 28, "paper"


def doc_invoice(c: Ctx):
    rng = c.rng
    company = rng.choice(COMPANIES)
    lines = [(s, rng.randint(1, 12), rng.uniform(20, 480)) for s in rng.sample(SERVICES[c.lang], rng.randint(2, 7))]
    total = sum(q * v for _, q, v in lines)
    rows = [[c.t("description"), c.t("qty"), c.t("price"), c.t("amount")]] + [[esc(s), str(q), c.money(v), c.money(q * v)] for s, q, v in lines]
    logo = f"<div style='display:flex;align-items:center;gap:10px'><div style='width:38px;height:38px;border-radius:{rng.choice([0, 6, 19])}px;background:{c.acc}'></div><b style='font-size:18px'>{esc(company)}</b></div>"
    meta = f"<div style='text-align:right'><div style='font-size:26px;font-weight:bold;color:{c.acc};letter-spacing:1px'>{c.t('invoice')}</div><div>{c.t('invoice_no')} {rng.randint(2024001, 2026999)}</div><div>{c.t('date')}: {c.date()}</div><div>{c.t('due')}: {c.date()}</div></div>"
    to = f"<div style='margin:26px 0 18px'><div style='color:#777;font-size:11px;text-transform:uppercase'>{c.t('bill_to')}</div><b>{esc(c.name())}</b><br>{'<br>'.join(esc(a) for a in c.address())}</div>"
    tbl = table(rows, ["52%", "12%", "18%", "18%"], head=f"background:{c.acc};color:#fff;font-weight:bold;", cell="border-bottom:1px solid #ddd;")
    sums = f"<div style='margin:14px 0 0 auto;width:46%'>{table([[c.t('subtotal'), c.money(total)], [c.t('vat') + ' 21%', c.money(total * 0.21)], ['<b>' + c.t('total') + '</b>', '<b>' + c.money(total * 1.21) + '</b>']], ['55%', '45%'], cell='border-bottom:1px solid #eee;')}</div>"
    foot = f"<div style='margin-top:40px;color:#777;font-size:10px;border-top:1px solid #ddd;padding-top:8px'>{esc(company)} · IBAN NL{rng.randint(10, 99)} TEST 0{rng.randint(100000000, 999999999)} · {esc(c.sentence(4, 8))}</div>"
    return f"<div style='width:640px;min-height:860px;padding:44px 48px;font:13px/1.5 {c.sans};color:#222;background:#fff'><div style='display:flex;justify-content:space-between'>{logo}{meta}</div>{to}{tbl}{sums}{foot}</div>", 736, "paper"


def doc_boarding(c: Ctx):
    rng = c.rng
    a, b = rng.sample(AIRPORTS, 2)
    field = lambda k, v: f"<div><div style='font-size:9px;color:#777;text-transform:uppercase'>{k}</div><div style='font-weight:bold;font-size:15px'>{esc(v)}</div></div>"
    main = (f"<div style='flex:1;padding:14px 18px'><div style='display:flex;align-items:center;gap:16px;font-size:40px;font-weight:bold'>{a}<span style='font-size:22px;color:{c.acc}'>✈</span>{b}</div>"
            f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px'>{field(c.t('passenger'), c.name().upper())}{field(c.t('flight'), rng.choice('KLHBAF') + rng.choice('LAFYEU') + ' ' + str(rng.randint(100, 9999)))}"
            f"{field(c.t('gate'), rng.choice('ABCDEF') + str(rng.randint(1, 40)))}{field(c.t('seat'), str(rng.randint(1, 38)) + rng.choice('ABCDEF'))}{field(c.t('date'), c.date())}{field(c.t('boards'), f'{rng.randint(5, 22)}:{rng.randint(0, 5)}0')}</div>"
            f"<div style='margin-top:12px'>{bars(rng, 330, 42)}</div></div>")
    stub = f"<div style='width:150px;padding:14px;border-left:2px dashed #999'><div style='font-size:22px;font-weight:bold'>{a} → {b}</div><div style='margin:8px 0'>{field(c.t('seat'), str(rng.randint(1, 38)) + rng.choice('ABCDEF'))}</div>{qr(rng, 84)}</div>"
    return (f"<div style='width:660px;font:12px/1.4 {c.sans};color:#1b1b1b;background:#fff;border-radius:12px;overflow:hidden'><div style='background:{c.acc};color:#fff;padding:9px 18px;font-weight:bold;letter-spacing:2px;display:flex;justify-content:space-between'>"
            f"<span>{esc(rng.choice(STEP.BRANDS))} AIR</span><span>{c.t('boarding')}</span></div><div style='display:flex'>{main}{stub}</div></div>"), 660, "paper"


def doc_ticket(c: Ctx):
    rng = c.rng
    dark = rng.random() < 0.5
    bg, fg = (hsl(c.hue, 45, 18), "#fff") if dark else ("#fffdf5", "#222")
    train = rng.random() < 0.35
    if train:
        big = f"{esc(rng.choice(STEP.CITIES[c.lang]))} → {esc(rng.choice(STEP.CITIES[c.lang]))}"
        sub = f"{c.t('date')}: {c.date()} · {rng.randint(5, 22)}:{rng.randint(0, 59):02d} · {rng.choice(['1', '2'])}. {rng.choice(['class', 'klasse', 'Klasse', 'classe', 'clase'])}"
    else:
        big = esc(c.title(2, 4)).upper()
        sub = f"{c.date()} · {c.t('doors')} {rng.randint(17, 21)}:00 · {esc(rng.choice(STEP.CITIES[c.lang]))}"
    code = qr(rng, 96, color=fg) if rng.random() < 0.6 else bars(rng, 120, 70, color=fg)
    return (f"<div style='width:600px;display:flex;font:13px/1.4 {c.sans};color:{fg};background:{bg};border-radius:10px;overflow:hidden'><div style='flex:1;padding:20px 24px;border-right:2px dashed {fg}'>"
            f"<div style='font-size:10px;letter-spacing:3px;opacity:.75'>{c.t('admit')}</div><div style='font:bold {rng.randint(24, 34)}px {rng.choice([c.sans, c.serif])};margin:6px 0;color:{c.acc if not dark else '#fff'}'>{big}</div><div>{sub}</div>"
            f"<div style='margin-top:14px;display:flex;gap:26px'><span>{c.t('row')} <b>{rng.randint(1, 40)}</b></span><span>{c.t('seat')} <b>{rng.randint(1, 60)}</b></span><span><b>{c.money(rng.uniform(8, 140))}</b></span></div></div>"
            f"<div style='padding:20px;display:grid;place-items:center'>{code}<div style='font:10px {c.mono};margin-top:6px'>{rng.randint(10 ** 9, 10 ** 10)}</div></div></div>"), 600, "paper"


def doc_letter(c: Ctx):
    rng = c.rng
    sender = f"<div style='text-align:right;color:#444'><b>{esc(rng.choice(COMPANIES))}</b><br>{'<br>'.join(esc(a) for a in Ctx(rng, c.lang).address())}</div>"
    paras = "".join(f"<p style='margin:0 0 12px;text-align:justify'>{esc(c.para(rng.randint(2, 5)))}</p>" for _ in range(rng.randint(3, 5)))
    sign = f"<div style='font:28px {fam(rng, HAND)};margin:8px 0 2px;color:#1a2a6c'>{esc(c.p['first'])} {esc(c.p['last'])}</div>"
    return (f"<div style='width:600px;min-height:820px;padding:56px 60px;font:{rng.randint(12, 14)}px/1.6 {rng.choice([c.serif, c.sans])};color:#222;background:#fff'>{sender}"
            f"<div style='margin:30px 0 22px'>{esc(c.name())}<br>{'<br>'.join(esc(a) for a in c.address())}</div><div style='margin-bottom:18px'>{esc(c.p['city'])}, {c.date()}</div>"
            f"<div style='font-weight:bold;margin-bottom:14px'>{c.t('subject')}: {esc(c.title(3, 6))}</div><div style='margin-bottom:12px'>{c.t('dear')} {esc(c.name())},</div>{paras}<div style='margin-top:20px'>{c.t('regards')}</div>{sign}</div>"), 720, "paper"


def doc_idcard(c: Ctx):
    rng = c.rng
    field = lambda k, v: f"<div style='margin-bottom:7px'><div style='font-size:8px;color:#555;text-transform:uppercase'>{k}</div><div style='font-weight:bold;font-size:14px'>{esc(v)}</div></div>"
    face = (f"<svg width='120' height='150' viewBox='0 0 120 150' style='display:block;background:{hsl(c.hue, 20, 80)};border-radius:6px'><circle cx='60' cy='56' r='27' fill='{hsl(c.hue, 15, 55)}'/>"
            f"<path d='M12 150 Q12 96 60 96 Q108 96 108 150 Z' fill='{hsl(c.hue, 15, 55)}'/></svg>")
    mrz = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ<<<<0123456789") for _ in range(30))
    return (f"<div style='width:500px;height:315px;border-radius:16px;overflow:hidden;position:relative;font:12px/1.3 {c.sans};color:#1b1b1b;"
            f"background:linear-gradient({rng.randint(100, 160)}deg,{hsl(c.hue, 45, 88)},{hsl(c.hue + 50, 50, 80)})'><div style='padding:12px 18px;font-weight:bold;letter-spacing:2px;display:flex;justify-content:space-between;color:{hsl(c.hue, 55, 28)}'>"
            f"<span>{c.t('idcard')}</span><span>{esc(c.p['country']).upper()}</span></div><div style='display:flex;gap:18px;padding:4px 18px'>{face}<div>{field(c.t('surname'), c.p['last'].upper())}{field(c.t('given'), c.p['first'])}"
            f"{field(c.t('born'), c.p['dob'])}{field(c.t('valid'), c.date())}</div></div><div style='position:absolute;left:18px;bottom:10px;font:13px {c.mono};letter-spacing:2px'>{mrz}</div>"
            f"<div style='position:absolute;right:30px;top:120px;font-size:40px;font-weight:bold;color:rgba(200,0,0,.28);transform:rotate(-18deg);letter-spacing:4px'>{c.t('specimen')}</div></div>"), 500, "paper"


def doc_bizcard(c: Ctx):
    rng = c.rng
    dark = rng.random() < 0.4
    bg, fg = (hsl(c.hue, 30, 14), "#f3f3f3") if dark else (rng.choice(["#fff", "#faf7f0", "#f2f5f7"]), "#222")
    centre = rng.random() < 0.4
    logo = f"<div style='width:{rng.randint(28, 44)}px;aspect-ratio:1;background:{c.acc};border-radius:{rng.choice(['0', '50%', '8px'])};{'margin:0 auto 10px' if centre else ''}'></div>"
    lines = f"{esc(c.p['phone'])}<br>{esc(c.p['email'])}<br>{esc(c.address()[0])}, {esc(c.address()[1])}"
    return (f"<div style='width:480px;height:270px;padding:30px 34px;box-sizing:border-box;font:12px/1.6 {rng.choice([c.sans, c.serif])};color:{fg};background:{bg};text-align:{'center' if centre else 'left'};"
            f"display:flex;flex-direction:column;justify-content:space-between'><div>{logo}<div style='font-size:{rng.randint(20, 26)}px;font-weight:bold'>{esc(c.name())}</div>"
            f"<div style='color:{c.acc if not dark else '#bbb'};letter-spacing:1px;text-transform:uppercase;font-size:11px'>{esc(c.title(1, 3))} · {esc(rng.choice(COMPANIES))}</div></div><div>{lines}</div></div>"), 480, "paper"


def doc_menu(c: Ctx):
    rng = c.rng
    dishes = rng.sample(DISHES[c.lang], len(DISHES[c.lang]))
    dark = rng.random() < 0.3
    bg, fg = ("#1d1b19", "#f0e9dc") if dark else (rng.choice(["#fbf7ee", "#fff", "#f4efe4"]), "#2a2622")
    out = ""
    for key in ("starters", "mains", "desserts", "drinks")[: rng.randint(3, 4)]:
        rows = "".join(f"<div style='display:flex;align-items:baseline;gap:6px;margin:7px 0'><span>{esc(dishes.pop() if dishes else c.title(2, 3))}</span><span style='flex:1;border-bottom:1px dotted {fg};opacity:.5'></span><span>{c.money(rng.uniform(3, 28))}</span></div>"
                       + (f"<div style='font-size:11px;opacity:.65;margin-top:-4px'>{esc(c.words(rng.randint(4, 8)))}</div>" if rng.random() < 0.5 else "") for _ in range(rng.randint(2, 4)))
        out += f"<div style='font:bold 17px {c.serif};letter-spacing:3px;text-transform:uppercase;margin:22px 0 6px;text-align:center;color:{c.acc if not dark else '#d8b46a'}'>{c.t(key)}</div>{rows}"
    return (f"<div style='width:460px;padding:38px 44px;font:14px/1.5 {c.serif};color:{fg};background:{bg};border:{rng.choice(['0', '6px double ' + fg])}'><div style='font:{rng.randint(30, 40)}px {rng.choice([c.serif, fam(rng, HAND)])};text-align:center'>{esc(rng.choice(SHOPS))}</div>"
            f"<div style='text-align:center;opacity:.6;font-size:11px;letter-spacing:2px'>— {esc(c.p['city'])} —</div>{out}</div>"), 560, "paper"


def doc_calendar(c: Ctx):
    rng = c.rng
    first, days = rng.randint(0, 6), rng.choice([28, 30, 31])
    head = "".join(f"<div style='padding:6px;font-weight:bold;color:#666;font-size:11px;text-transform:uppercase'>{d}</div>" for d in WEEKDAYS[c.lang])
    cells = ""
    for i in range(42 if first + days > 35 else 35):
        day = i - first + 1
        chips = "".join(f"<div style='background:{hsl(c.hue + 70 * rng.randint(0, 4), 60, 52)};color:#fff;border-radius:3px;padding:1px 4px;font-size:9px;margin-top:2px;white-space:nowrap;overflow:hidden'>{esc(c.words(rng.randint(1, 2)))}</div>"
                        for _ in range(rng.choice([0, 0, 0, 1, 1, 2]))) if 1 <= day <= days else ""
        today = "background:%s;color:#fff;border-radius:50%%;width:20px;height:20px;display:inline-grid;place-items:center" % c.acc if day == rng.randint(1, 31) else ""
        cells += f"<div style='border-top:1px solid #e3e3e3;border-left:1px solid #e3e3e3;min-height:{rng.choice([62, 74])}px;padding:4px;font-size:11px'><span style='{today}'>{day if 1 <= day <= days else ''}</span>{chips}</div>"
    return (f"<div style='width:760px;font:13px {c.sans};color:#222;background:#fff'><div style='display:flex;justify-content:space-between;align-items:center;padding:14px 18px'><b style='font-size:22px'>{MONTHS[c.lang][rng.randint(0, 11)].capitalize()} {rng.choice([2025, 2026])}</b>"
            f"<span style='color:#666'>‹ &nbsp; ›</span></div><div style='display:grid;grid-template-columns:repeat(7,1fr)'>{head}{cells}</div></div>"), 760, "screen"


def doc_spreadsheet(c: Ctx):
    rng = c.rng
    cols, rows = rng.randint(6, 9), rng.randint(14, 22)
    letters = "".join(f"<div style='background:#f1f3f4;border:1px solid #d7d7d7;text-align:center;color:#555;padding:2px'>{chr(65 + i)}</div>" for i in range(cols))
    grid = ""
    for r in range(rows):
        grid += f"<div style='background:#f1f3f4;border:1px solid #d7d7d7;text-align:center;color:#555;padding:3px 0'>{r + 1}</div>"
        for col in range(cols):
            if r == 0:
                v, s = esc(c.title(1, 2)), "font-weight:bold;background:%s" % hsl(c.hue, 45, 90)
            elif col == 0:
                v, s = esc(c.words(rng.randint(1, 2))), ""
            else:
                v, s = (f"{rng.uniform(1, 9000):,.2f}" if rng.random() < 0.8 else ""), "text-align:right"
            sel = f"outline:2px solid {c.acc};outline-offset:-2px;" if (r, col) == (rng.randint(1, rows - 1), rng.randint(0, cols - 1)) else ""
            grid += f"<div style='border:1px solid #e4e4e4;padding:3px 6px;white-space:nowrap;overflow:hidden;{s};{sel}'>{v}</div>"
    tools = "".join(f"<span style='display:inline-block;width:{rng.choice([18, 18, 46, 70])}px;height:18px;background:#dadce0;border-radius:3px;margin-right:6px'></span>" for _ in range(rng.randint(9, 14)))
    return (f"<div style='width:860px;font:11px {c.sans};color:#222;background:#fff'><div style='padding:8px 10px;background:{rng.choice(['#f8f9fa', hsl(140, 35, 30), '#fff'])};border-bottom:1px solid #ddd'>{tools}</div>"
            f"<div style='padding:4px 10px;border-bottom:1px solid #ddd;color:#666;font-family:{c.mono}'>fx &nbsp; =SUM(B2:B{rows})</div><div style='display:grid;grid-template-columns:38px repeat({cols},1fr)'><div style='background:#f1f3f4;border:1px solid #d7d7d7'></div>{letters}{grid}</div></div>"), 860, "screen"


def doc_chart(c: Ctx):
    rng = c.rng
    kind = rng.choice(["bars", "bars", "line", "pie"])
    w, h = 660, 340
    n = rng.randint(4, 10)
    values = [rng.uniform(15, 100) for _ in range(n)]
    colors = [hsl(c.hue + 47 * i, 60, 52) for i in range(n)]
    axis = f"<line x1='40' y1='10' x2='40' y2='{h - 30}' stroke='#888'/><line x1='40' y1='{h - 30}' x2='{w - 10}' y2='{h - 30}' stroke='#888'/>" + "".join(
        f"<line x1='40' y1='{y}' x2='{w - 10}' y2='{y}' stroke='#e5e5e5'/>" for y in range(30, h - 30, 55))
    if kind == "bars":
        bw = (w - 70) / n
        marks = "".join(f"<rect x='{50 + i * bw:.0f}' y='{h - 30 - v * 2.8:.0f}' width='{bw * 0.7:.0f}' height='{v * 2.8:.0f}' fill='{colors[i] if rng.random() < 0.5 else c.acc}'/>" for i, v in enumerate(values))
        plot = f"<svg width='{w}' height='{h}'>{axis}{marks}</svg>"
    elif kind == "line":
        series = ""
        for s in range(rng.randint(1, 3)):
            pts = " ".join(f"{50 + i * (w - 70) / (n - 1):.0f},{h - 30 - rng.uniform(15, 100) * 2.7:.0f}" for i in range(n))
            series += f"<polyline points='{pts}' fill='none' stroke='{colors[s * 2 % n]}' stroke-width='3'/>"
        plot = f"<svg width='{w}' height='{h}'>{axis}{series}</svg>"
    else:
        total, at, stops = sum(values), 0.0, []
        for i, v in enumerate(values):
            stops.append(f"{colors[i]} {at:.1f}% {at + 100 * v / total:.1f}%")
            at += 100 * v / total
        plot = f"<div style='width:300px;height:300px;border-radius:50%;margin:10px auto;background:conic-gradient({','.join(stops)})'></div>"
    legend = "".join(f"<span style='margin-right:14px;white-space:nowrap'><span style='display:inline-block;width:10px;height:10px;background:{colors[i]};margin-right:5px'></span>{esc(c.words(1))}</span>" for i in range(min(n, 5)))
    return (f"<div style='width:{w + 40}px;padding:20px;font:12px {c.sans};color:#222;background:#fff'><div style='font-size:18px;font-weight:bold;margin-bottom:4px'>{esc(c.title(2, 5))}</div>"
            f"<div style='color:#777;margin-bottom:10px'>{esc(c.sentence(4, 8))}</div>{plot}<div style='margin-top:8px'>{legend}</div></div>"), w + 80, "screen"


def doc_slide(c: Ctx):
    rng = c.rng
    dark = rng.random() < 0.45
    bg = f"linear-gradient({rng.randint(0, 360)}deg,{hsl(c.hue, 55, 22)},{hsl(c.hue + 40, 55, 32)})" if dark else rng.choice(["#fff", "#f6f4ef", hsl(c.hue, 40, 95)])
    fg = "#fff" if dark else "#1f1f1f"
    bullets = "".join(f"<li style='margin:10px 0'>{esc(c.sentence(4, 9))}</li>" for _ in range(rng.randint(3, 5)))
    side = f"<div style='width:270px;height:230px;border-radius:8px;overflow:hidden'>{c.photo()}</div>" if rng.random() < 0.45 else ""
    return (f"<div style='width:800px;height:450px;box-sizing:border-box;padding:46px 56px;font:19px/1.35 {rng.choice([c.sans, c.serif])};color:{fg};background:{bg};position:relative'>"
            f"<div style='font-size:{rng.randint(32, 42)}px;font-weight:bold;margin-bottom:8px;color:{fg if dark else c.acc}'>{esc(c.title(3, 6))}</div><div style='display:flex;gap:30px;align-items:center'><ul style='flex:1;padding-left:24px;margin:8px 0'>{bullets}</ul>{side}</div>"
            f"<div style='position:absolute;bottom:16px;right:26px;font-size:12px;opacity:.6'>{rng.randint(2, 30)}</div><div style='position:absolute;bottom:16px;left:56px;font-size:12px;opacity:.6'>{esc(rng.choice(COMPANIES))}</div></div>"), 800, "screen"


def doc_code(c: Ctx):
    rng = c.rng
    dark = rng.random() < 0.75
    bg, fg, dim = ("#1e1f24", "#d6d9e0", "#5c6370") if dark else ("#fafafa", "#2b2b2b", "#a0a0a0")
    palette = [hsl(h, 65, 68 if dark else 40) for h in (205, 30, 95, 285, 350, 50)]
    idents = ["user", "items", "total", "result", "config", "request", "index", "value", "data", "state", "render", "fetch", "count", "price", "token", "path"]
    lines, indent = "", 0
    for n in range(1, rng.randint(18, 26)):
        if rng.random() < 0.12:
            body = ""
        elif rng.random() < 0.1:
            body = f"<span style='color:{dim}'>// {esc(c.words(rng.randint(3, 7)))}</span>"
        else:
            kw = rng.choice(["const", "let", "if", "return", "function", "for", "import", "await", "export", "class"])
            parts = [f"<span style='color:{rng.choice(palette)}'>{rng.choice(idents)}</span>" for _ in range(rng.randint(1, 4))]
            body = f"<span style='color:{palette[3]}'>{kw}</span> " + rng.choice([" = ", ".", "(", ", "]).join(parts) + rng.choice([";", " {", ")", "", "()"]) + (f" <span style='color:{palette[2]}'>'{esc(c.words(1))}'</span>" if rng.random() < 0.3 else "")
            indent = max(0, min(4, indent + rng.choice([-1, 0, 0, 1])))
        lines += f"<div style='white-space:pre'><span style='color:{dim};display:inline-block;width:34px;text-align:right;margin-right:14px'>{n}</span>{'  ' * indent}{body}</div>"
    tabs = "".join(f"<span style='padding:6px 14px;background:{bg if i == 0 else 'transparent'};color:{fg if i == 0 else dim}'>{rng.choice(idents)}.{rng.choice(['ts', 'py', 'js', 'go', 'rs'])}</span>" for i in range(rng.randint(2, 4)))
    return f"<div style='width:780px;font:13px/1.55 {c.mono};color:{fg};background:{bg}'><div style='background:{'#16171b' if dark else '#ececec'};font-size:12px;display:flex'>{tabs}</div><div style='padding:12px 8px'>{lines}</div></div>", 780, "screen"


def doc_chat(c: Ctx):
    rng = c.rng
    dark = rng.random() < 0.3
    bg, other, fg = ("#101214", "#23272b", "#eee") if dark else (rng.choice(["#fff", "#efeae2", "#f5f6f8"]), "#fff" if rng.random() < 0.5 else "#e9e9eb", "#111")
    mine = rng.choice([hsl(c.hue, 70, 46), "#0b84fe", "#d9fdd3", "#25d366"])
    mine_fg = "#111" if mine in ("#d9fdd3",) else "#fff"
    bubbles = ""
    for _ in range(rng.randint(7, 12)):
        me = rng.random() < 0.45
        text = esc(c.sentence(2, 12) if rng.random() < 0.85 else rng.choice(["👍", "😂", "OK!", "❤️"]))
        bubbles += (f"<div style='display:flex;justify-content:{'flex-end' if me else 'flex-start'};margin:5px 0'><div style='max-width:72%;padding:8px 12px;border-radius:16px;background:{mine if me else other};color:{mine_fg if me else fg};"
                    f"box-shadow:0 1px 1px rgba(0,0,0,.08)'>{text}<span style='font-size:9px;opacity:.6;margin-left:8px'>{rng.randint(8, 23)}:{rng.randint(0, 59):02d}</span></div></div>")
    head = (f"<div style='display:flex;align-items:center;gap:10px;padding:12px 14px;background:{'#1b1e21' if dark else '#f6f6f6'};border-bottom:1px solid rgba(0,0,0,.1)'><span style='font-size:18px;color:{mine if mine_fg == '#fff' else '#07c'}'>‹</span>"
            f"<div style='width:34px;height:34px;border-radius:50%;background:{hsl(c.hue + 120, 45, 60)}'></div><div><b>{esc(c.p['first'])}</b><div style='font-size:10px;opacity:.6'>online</div></div></div>")
    foot = f"<div style='padding:10px 12px;display:flex;gap:8px;align-items:center;background:{'#1b1e21' if dark else '#f6f6f6'}'><div style='flex:1;padding:9px 14px;border-radius:18px;background:{other};opacity:.8;color:#888'>{c.t('type_message')}</div><div style='width:34px;height:34px;border-radius:50%;background:{mine}'></div></div>"
    return f"<div style='width:390px;font:14px/1.35 {c.sans};color:{fg};background:{bg}'>{head}<div style='padding:10px 12px;min-height:520px'>{bubbles}</div>{foot}</div>", 390, "phone"


def doc_note(c: Ctx):
    rng = c.rng
    ink = rng.choice(["#1a2a6c", "#222", "#0b3d91", "#7a1f1f"])
    lined = rng.random() < 0.6
    lh = rng.randint(30, 38)
    paper = rng.choice(["#fffdf2", "#fff", "#fff9c4", "#f3f7ff"])
    bg = f"repeating-linear-gradient({paper} 0 {lh - 1}px,#b9c6dd {lh - 1}px {lh}px)" if lined else paper
    items = rng.sample(GROCERIES[c.lang], rng.randint(4, 9)) if rng.random() < 0.6 else [c.words(rng.randint(2, 5)) for _ in range(rng.randint(4, 8))]
    rows = "".join(f"<div style='{'text-decoration:line-through;opacity:.6;' if rng.random() < 0.2 else ''}transform:rotate({rng.uniform(-1.2, 1.2):.1f}deg)'>{rng.choice(['- ', '• ', '☐ ', ''])}{esc(i)}</div>" for i in items)
    return (f"<div style='width:{rng.randint(340, 440)}px;padding:{lh}px 30px {lh * 2}px;font:{rng.randint(22, 28)}px/{lh}px {fam(rng, HAND)};color:{ink};background:{bg}'>"
            f"<div style='font-size:1.25em;text-decoration:underline'>{rng.choice([c.t('shopping'), c.t('todo')])}</div>{rows}</div>"), 460, "paper"


def doc_statement(c: Ctx):
    rng = c.rng
    balance, rows = rng.uniform(300, 9000), [[c.t("date"), c.t("description"), c.t("amount"), c.t("balance")]]
    for _ in range(rng.randint(10, 18)):
        v = rng.uniform(-420, 120) if rng.random() < 0.85 else rng.uniform(900, 3200)
        balance += v
        rows.append([c.date(), esc(rng.choice(SHOPS + COMPANIES) + " " + c.words(rng.randint(0, 2))), f"<span style='color:{'#b00020' if v < 0 else '#11772d'}'>{'-' if v < 0 else '+'}{c.money(abs(v))}</span>", c.money(abs(balance))])
    head = f"<div style='display:flex;justify-content:space-between;align-items:flex-end;border-bottom:3px solid {c.acc};padding-bottom:10px'><b style='font-size:22px;color:{c.acc}'>{esc(rng.choice(BANKS))}</b><span>{c.t('statement')}</span></div>"
    who = f"<div style='margin:18px 0;display:flex;justify-content:space-between'><div><b>{esc(c.name())}</b><br>{'<br>'.join(esc(a) for a in c.address())}</div><div style='text-align:right'>{c.t('account')}<br><b>NL{rng.randint(10, 99)} TEST 0{rng.randint(100000000, 999999999)}</b></div></div>"
    return f"<div style='width:640px;min-height:800px;padding:40px 44px;font:12px/1.45 {c.sans};color:#222;background:#fff'>{head}{who}{table(rows, ['16%', '48%', '18%', '18%'], head='font-weight:bold;border-bottom:2px solid #333;', cell='border-bottom:1px solid #e6e6e6;')}</div>", 728, "paper"


def doc_label(c: Ctx):
    rng = c.rng
    other = Ctx(rng, c.lang)
    box = "border:3px solid #111;"
    return (f"<div style='width:400px;font:13px/1.35 {c.sans};color:#111;background:#fff;{box}'><div style='display:flex;border-bottom:3px solid #111'><div style='flex:1;padding:8px 10px;font-size:11px'><b>{c.t('sender')}</b><br>{esc(other.name())}<br>{'<br>'.join(esc(a) for a in other.address())}</div>"
            f"<div style='width:110px;border-left:3px solid #111;display:grid;place-items:center;font:bold 44px {c.sans}'>{rng.choice('ABCDEFGHKLMNPRST')}{rng.randint(1, 9)}</div></div>"
            f"<div style='padding:12px 14px;border-bottom:3px solid #111'><b>{c.t('ship_to')}</b><div style='font-size:20px;font-weight:bold;margin-top:4px'>{esc(c.name())}</div><div style='font-size:17px'>{'<br>'.join(esc(a) for a in c.address())}<br>{esc(c.p['country']).upper()}</div></div>"
            f"<div style='display:flex;border-bottom:3px solid #111'><div style='flex:1;padding:8px 10px'>{rng.choice(['EXPRESS', 'STANDARD', 'PRIORITY', '24H'])}<br><b>{rng.uniform(0.2, 18):.1f} kg</b></div><div style='padding:8px'>{qr(rng, 70)}</div></div>"
            f"<div style='padding:14px 20px;text-align:center'>{bars(rng, 340, 90)}<div style='font:14px {c.mono};letter-spacing:3px;margin-top:6px'>{rng.randint(10 ** 13, 10 ** 14)}</div><div style='font-size:10px'>{c.t('tracking')}</div></div></div>"), 406, "paper"


def doc_certificate(c: Ctx):
    rng = c.rng
    gold = rng.choice(["#a67c00", "#7a5c00", c.acc, "#1f3a5f"])
    seal = f"<div style='width:84px;height:84px;border-radius:50%;background:radial-gradient(circle,{gold} 45%,transparent 47%),repeating-conic-gradient({gold} 0 10deg,transparent 10deg 20deg);margin:0 auto'></div>"
    sign = "".join(f"<div style='width:190px;text-align:center'><div style='font:26px {fam(rng, HAND)};color:#1a2a6c'>{esc(Ctx(rng, c.lang).name())}</div><div style='border-top:1px solid #333;font-size:11px;padding-top:3px'>{c.t('signature')}</div></div>" for _ in range(2))
    return (f"<div style='width:760px;height:540px;box-sizing:border-box;padding:16px;background:{rng.choice(['#fffdf5', '#fff', '#fbf6e9'])}'><div style='height:100%;box-sizing:border-box;border:{rng.choice(['6px double', '4px solid', '10px ridge'])} {gold};padding:28px 40px;text-align:center;font:15px/1.5 {c.serif};color:#2a2a2a'>"
            f"<div style='font-size:{rng.randint(38, 50)}px;letter-spacing:6px;color:{gold}'>{c.t('certificate')}</div><div style='margin:12px 0 4px'>{c.t('awarded')}</div><div style='font:{rng.randint(40, 52)}px {fam(rng, HAND, SERIF)};margin:4px 0'>{esc(c.name())}</div>"
            f"<div>{c.t('for')}</div><div style='font-weight:bold;font-size:19px;margin:4px 0 14px'>{esc(c.title(2, 5))}</div>{seal}<div style='display:flex;justify-content:space-around;margin-top:10px'>{sign}</div></div></div>"), 760, "paper"


def doc_form(c: Ctx):
    rng = c.rng
    line = lambda k, wide=True: f"<div style='flex:{2 if wide else 1};margin:9px 6px'><div style='font-size:10px;color:#444'>{k}</div><div style='border:1px solid #555;height:{rng.choice([24, 28])}px;background:#fff'></div></div>"
    comb = lambda k, n: f"<div style='margin:9px 6px'><div style='font-size:10px;color:#444'>{k}</div><div style='display:flex'>{''.join('<div style=\"width:20px;height:24px;border:1px solid #555;margin-right:-1px;background:#fff\"></div>' for _ in range(n))}</div></div>"
    tick = lambda k: f"<label style='margin-right:18px'><span style='display:inline-block;width:12px;height:12px;border:1px solid #333;vertical-align:-2px;margin-right:5px;background:#fff'></span>{esc(k)}</label>"
    sections = ""
    for s in range(rng.randint(2, 3)):
        rows = f"<div style='display:flex'>{line(c.t('surname'))}{line(c.t('given'))}</div><div style='display:flex'>{line(c.t('address'))}{comb(c.t('born'), 8)}</div><div style='display:flex'>{line(c.t('phone'), False)}{line(c.t('email'))}</div>" if s == 0 else \
            "".join(f"<div style='display:flex'>{line(esc(c.title(1, 3)))}{line(esc(c.title(1, 2)), rng.random() < 0.5)}</div>" for _ in range(rng.randint(1, 3))) + f"<div style='margin:10px 6px'>{''.join(tick(c.title(1, 2)) for _ in range(rng.randint(2, 4)))}</div>"
        sections += f"<div style='background:{hsl(c.hue, 30, 90)};padding:4px 8px;font-weight:bold;margin-top:14px'>{s + 1}. {esc(c.title(2, 4))}</div>{rows}"
    sign = f"<div style='display:flex;margin-top:20px'>{line(c.t('date'), False)}<div style='flex:2;margin:9px 6px'><div style='font-size:10px;color:#444'>{c.t('signature')}</div><div style='border:1px solid #555;height:56px;background:#fff'></div></div></div>"
    return (f"<div style='width:620px;min-height:800px;padding:36px 40px;font:12px/1.4 {c.sans};color:#222;background:{rng.choice(['#fff', '#fdfcf7', hsl(c.hue, 25, 97)])}'><div style='display:flex;justify-content:space-between;align-items:center'><b style='font-size:20px'>{c.t('form')}</b>"
            f"<span style='border:1px solid #333;padding:2px 8px;font:11px {c.mono}'>{rng.choice('ABDFM')}-{rng.randint(10, 999)}</span></div><div style='color:#555;margin-top:4px'>{esc(c.sentence(6, 12))}</div>{sections}{sign}</div>"), 700, "paper"


def doc_map(c: Ctx):
    rng = c.rng
    w, h = 720, 520
    dark = rng.random() < 0.2
    land, road, water, park = ("#1f2530", "#3a4352", "#16324a", "#22382b") if dark else (rng.choice(["#f2efe9", "#e8eaed", "#f5f3ee"]), "#fff", "#aad3df", "#c8e6c0")
    shapes = ""
    for _ in range(rng.randint(2, 4)):
        x, y = rng.randint(0, w), rng.randint(0, h)
        shapes += f"<ellipse cx='{x}' cy='{y}' rx='{rng.randint(50, 170)}' ry='{rng.randint(30, 110)}' fill='{rng.choice([park, park, water])}'/>"
    if rng.random() < 0.6:
        y0 = rng.randint(100, h - 100)
        shapes += f"<path d='M0 {y0} C {w / 3} {y0 + rng.randint(-120, 120)}, {2 * w / 3} {y0 + rng.randint(-120, 120)}, {w} {y0 + rng.randint(-60, 60)}' stroke='{water}' stroke-width='{rng.randint(18, 40)}' fill='none'/>"
    for _ in range(rng.randint(14, 26)):
        if rng.random() < 0.5:
            y = rng.randint(0, h)
            shapes += f"<line x1='0' y1='{y}' x2='{w}' y2='{y + rng.randint(-90, 90)}' stroke='{road}' stroke-width='{rng.choice([3, 4, 6, 10])}'/>"
        else:
            x = rng.randint(0, w)
            shapes += f"<line x1='{x}' y1='0' x2='{x + rng.randint(-90, 90)}' y2='{h}' stroke='{road}' stroke-width='{rng.choice([3, 4, 6, 10])}'/>"
    shapes += f"<line x1='0' y1='{rng.randint(0, h)}' x2='{w}' y2='{rng.randint(0, h)}' stroke='#f9d77a' stroke-width='9'/>"
    labels = "".join(f"<text x='{rng.randint(20, w - 120)}' y='{rng.randint(20, h - 10)}' font-size='{rng.choice([10, 11, 13])}' fill='{'#aab' if dark else '#666'}' font-family='Arial'>{esc(rng.choice(STEP.STREETS[c.lang]))}</text>" for _ in range(rng.randint(5, 9)))
    px, py = rng.randint(120, w - 120), rng.randint(100, h - 100)
    pin = f"<path d='M{px} {py} c-14 -22 -22 -30 -22 -44 a22 22 0 1 1 44 0 c0 14 -8 22 -22 44z' fill='#e53935'/><circle cx='{px}' cy='{py - 44}' r='8' fill='#fff'/>"
    ui = f"<div style='position:absolute;left:14px;top:14px;background:#fff;border-radius:8px;padding:9px 14px;box-shadow:0 1px 4px rgba(0,0,0,.3);width:230px;color:#777;font:13px {c.sans}'>{c.t('search')}…</div><div style='position:absolute;right:14px;bottom:14px;background:#fff;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font:18px {c.sans};width:30px;text-align:center;line-height:30px;color:#444'>+<br>−</div>"
    return f"<div style='width:{w}px;height:{h}px;position:relative;overflow:hidden'><svg width='{w}' height='{h}' style='display:block;background:{land}'>{shapes}{labels}{pin}</svg>{ui if rng.random() < 0.7 else ''}</div>", w, "screen"


def doc_email(c: Ctx):
    rng = c.rng
    folders = "".join(f"<div style='padding:7px 14px;border-radius:0 16px 16px 0;{'background:' + hsl(c.hue, 60, 90) + ';font-weight:bold;' if i == 0 else ''}'>{c.t(k)}{f'<span style=float:right>{rng.randint(2, 60)}</span>' if i == 0 else ''}</div>" for i, k in enumerate(("inbox", "sent", "drafts", "trash")))
    rows = ""
    for _ in range(rng.randint(10, 15)):
        unread = rng.random() < 0.35
        who = Ctx(rng, c.lang)
        rows += (f"<div style='display:flex;gap:12px;padding:8px 14px;border-bottom:1px solid #eee;{'font-weight:bold;background:#fff;' if unread else 'background:#f6f8fc;'}white-space:nowrap'><span style='width:14px;color:#bbb'>☐</span>"
                 f"<span style='width:140px;overflow:hidden'>{esc(who.name() if rng.random() < 0.6 else rng.choice(COMPANIES))}</span><span style='flex:1;overflow:hidden'>{esc(c.title(2, 5))} <span style='font-weight:normal;color:#777'>– {esc(c.sentence(5, 10))}</span></span><span style='color:#555;font-size:11px'>{rng.randint(1, 28)} {MONTHS[c.lang][rng.randint(0, 11)][:3]}</span></div>")
    compose = f"<div style='margin:14px;padding:12px 18px;border-radius:16px;background:{c.acc};color:#fff;font-weight:bold;display:inline-block'>+ {esc(c.title(1, 1))}</div>"
    top = f"<div style='padding:10px 16px;display:flex;gap:16px;align-items:center;border-bottom:1px solid #e5e5e5'><b style='font-size:18px;color:{c.acc};width:170px'>{esc(rng.choice(STEP.BRANDS))} Mail</b><div style='flex:1;background:#eef1f6;border-radius:8px;padding:9px 14px;color:#777'>{c.t('search')}</div><div style='width:30px;height:30px;border-radius:50%;background:{hsl(c.hue + 150, 50, 55)}'></div></div>"
    return f"<div style='width:860px;font:13px/1.35 {c.sans};color:#222;background:#fff'>{top}<div style='display:flex'><div style='width:190px'>{compose}{folders}</div><div style='flex:1'>{rows}</div></div></div>", 860, "screen"


def doc_article(c: Ctx):
    rng = c.rng
    cols = rng.choice([2, 3, 3, 4])
    body = "".join(f"<p style='margin:0 0 8px;text-indent:12px'>{esc(c.para(rng.randint(3, 6)))}</p>" for _ in range(rng.randint(8, 12)))
    return (f"<div style='width:700px;padding:30px 36px;font:11.5px/1.45 {c.serif};color:#1b1b1b;background:{rng.choice(['#f7f3e8', '#fff', '#efe9da'])}'><div style='text-align:center;font:{rng.randint(40, 54)}px {rng.choice([c.serif, "Luminari, Didot, serif"])};border-bottom:3px double #222;padding-bottom:6px'>{esc(rng.choice(['The Daily Post', 'De Courant', 'Die Zeitung', 'Le Journal', 'El Diario', 'The Herald', 'Het Nieuwsblad', 'Morgenblatt']))}</div>"
            f"<div style='display:flex;justify-content:space-between;font-size:10px;border-bottom:1px solid #222;padding:3px 0;margin-bottom:12px'><span>{c.date()}</span><span>{esc(c.p['city'])}</span><span>{c.money(rng.uniform(1, 4))}</span></div>"
            f"<div style='font:bold {rng.randint(26, 36)}px/1.1 {c.serif};margin-bottom:6px'>{esc(c.title(5, 9))}</div><div style='font-style:italic;margin-bottom:10px'>{esc(c.sentence(8, 14))}</div>"
            f"<div style='column-count:{cols};column-gap:18px;column-rule:1px solid #bbb;text-align:justify'><div style='height:{rng.randint(110, 170)}px;margin-bottom:8px;filter:grayscale({rng.choice([0, 1, 1])})'>{c.photo()}</div>{body}</div></div>"), 772, "paper"


DOCS = {"receipt": doc_receipt, "invoice": doc_invoice, "boarding": doc_boarding, "ticket": doc_ticket, "letter": doc_letter, "idcard": doc_idcard, "bizcard": doc_bizcard, "menu": doc_menu,
        "calendar": doc_calendar, "spreadsheet": doc_spreadsheet, "chart": doc_chart, "slide": doc_slide, "code": doc_code, "chat": doc_chat, "note": doc_note, "statement": doc_statement,
        "label": doc_label, "certificate": doc_certificate, "form": doc_form, "map": doc_map, "email": doc_email, "article": doc_article}
SURFACES = [
    "repeating-linear-gradient(90deg,hsl(28 42% 36%) 0 7px,hsl(28 40% 32%) 7px 13px,hsl(30 44% 38%) 13px 22px)",  # wood
    "repeating-linear-gradient(0deg,hsl(32 30% 62%) 0 9px,hsl(32 28% 58%) 9px 17px)",
    "linear-gradient(135deg,#3b3f45,#22252a)", "linear-gradient(160deg,#e9e6df,#cfcabf)", "radial-gradient(circle at 30% 20%,#6d7b8a,#39424d)",
    "repeating-linear-gradient(45deg,hsl(210 18% 82%) 0 4px,hsl(210 18% 78%) 4px 8px)", "linear-gradient(#f4f4f4,#dedede)", "linear-gradient(120deg,hsl(150 25% 30%),hsl(150 25% 20%))",
]


def document_page(rng: random.Random, kind: str, lang: str, photos: list[str]) -> tuple[str, int, int, str]:
    """The page a document is drawn on: (html, viewport width, viewport height, how it is shown)."""
    c = Ctx(rng, lang, photos)
    sheet, w, surface = DOCS[kind](c)
    shown = rng.choices(["scan", "photo"], [0.4, 0.6])[0] if surface == "paper" else rng.choices(["raw", "window", "photo"], [0.5, 0.35, 0.15])[0]
    if shown == "raw":
        inner, pad, bg = sheet, 0, "#fff"
    elif shown == "scan":
        pad, bg = rng.randint(6, 40), rng.choice(["#fff", "#f1f1ee", "#e4e4e4", "#fafafa"])
        inner = f"<div style='display:inline-block;transform:rotate({rng.uniform(-1.6, 1.6):.2f}deg);box-shadow:0 1px 5px rgba(0,0,0,.25);filter:grayscale({rng.choice([0, 0, 1])}) contrast({rng.uniform(0.95, 1.15):.2f})'>{sheet}</div>"
    elif shown == "window":
        pad, bg = rng.randint(18, 60), f"linear-gradient({rng.randint(0, 360)}deg,{hsl(rng.uniform(0, 360), 50, 55)},{hsl(rng.uniform(0, 360), 55, 35)})"
        bar = f"<div style='height:28px;background:{rng.choice(['#e8e8e8', '#2b2b2f', '#f5f5f5'])};display:flex;align-items:center;gap:6px;padding:0 10px'>{''.join(f'<span style=width:11px;height:11px;border-radius:50%;background:{col};display:inline-block></span>' for col in ('#ff5f57', '#febc2e', '#28c840'))}</div>"
        inner = f"<div style='display:inline-block;border-radius:10px;overflow:hidden;box-shadow:0 18px 50px rgba(0,0,0,.45)'>{bar}{sheet}</div>"
    else:
        pad, bg = rng.randint(50, 130), rng.choice(SURFACES)
        tilt = f"perspective({rng.randint(900, 1800)}px) rotateX({rng.uniform(0, 24):.1f}deg) rotateY({rng.uniform(-14, 14):.1f}deg) rotateZ({rng.uniform(-13, 13):.1f}deg)"
        light = f"<div style='position:absolute;inset:0;pointer-events:none;background:linear-gradient({rng.randint(0, 360)}deg,rgba(255,255,255,{rng.uniform(0, 0.16):.2f}),rgba(0,0,0,{rng.uniform(0.05, 0.3):.2f}))'></div>"
        inner = f"<div style='display:inline-block;position:relative;transform:{tilt};box-shadow:0 {rng.randint(8, 26)}px {rng.randint(18, 46)}px rgba(0,0,0,.4);filter:blur({rng.uniform(0, 0.7):.1f}px) brightness({rng.uniform(0.86, 1.06):.2f})'>{sheet}{light}</div>"
    html = f"<body style='margin:0;background:#fff'><div id='scene' style='display:inline-block;padding:{pad}px;background:{bg};overflow:hidden;line-height:0'><div style='line-height:normal'>{inner}</div></div></body>"
    return html, w + 2 * pad + 60, 1800, shown


# ── pages ────────────────────────────────────────────────────────────────────────────────────────────────────────────

LAYOUTS = ["shop", "article", "results", "landing", "dashboard", "docs"]
HELD_OUT_LAYOUTS = {"dashboard", "docs"}
VIEWPORTS = [(1280, 800), (1440, 900), (1366, 768), (1024, 768), (1536, 864), (390, 844), (412, 915)]


class Site:
    """One made-up site's look: its colours, fonts and name."""

    def __init__(self, c: Ctx, vw: int) -> None:
        rng = c.rng
        self.c, self.vw, self.mobile = c, vw, vw < 600
        self.dark = rng.random() < 0.22
        self.bg, self.fg, self.muted, self.card, self.line = ("#111418", "#e8eaed", "#9aa0a6", "#1c2026", "#2d333b") if self.dark else (rng.choice(["#fff", "#fafafa", "#f6f5f1"]), "#1f2328", "#666d76", "#fff", "#e3e5e8")
        self.font = rng.choice([c.sans, c.sans, c.serif])
        self.name = rng.choice(STEP.SITES + STEP.BRANDS)
        self.radius = rng.choice([0, 4, 8, 14])

    def button(self, text: str, primary: bool = True, style: str = "") -> str:
        s = f"background:{self.c.acc};color:#fff;border:1px solid {self.c.acc}" if primary else f"background:transparent;color:{self.fg};border:1px solid {self.muted}"
        return f"<span style='display:inline-block;padding:9px 18px;border-radius:{self.radius}px;font-weight:600;white-space:nowrap;{s};{style}'>{esc(text)}</span>"

    def header(self) -> str:
        c, rng = self.c, self.c.rng
        tone = rng.choice(["plain", "plain", "accent", "dark"])
        bg, fg = {"plain": (self.card, self.fg), "accent": (c.acc, "#fff"), "dark": ("#15181c", "#fff")}[tone]
        logo = f"<span style='display:inline-flex;align-items:center;gap:8px;font-weight:800;font-size:19px'><span style='width:24px;height:24px;border-radius:{rng.choice([4, 12, 0])}px;background:{c.acc if tone != 'accent' else '#fff'};display:inline-block'></span>{esc(self.name)}</span>"
        nav = "" if self.mobile else "".join(f"<span style='margin-left:22px;opacity:.85'>{c.t(k)}</span>" for k in rng.sample(["home", "products", "about", "contact", "news", "pricing"], rng.randint(3, 5)))
        right = "<span style='font-size:20px'>☰</span>" if self.mobile else f"<span style='display:inline-block;padding:7px 14px;border-radius:18px;background:{'rgba(255,255,255,.18)' if tone != 'plain' else '#eef0f3' if not self.dark else '#2a3038'};width:150px;opacity:.75'>{c.t('search')}</span><span style='margin-left:16px'>{c.t('sign_in')}</span>"
        return f"<div style='display:flex;align-items:center;justify-content:space-between;padding:{rng.randint(12, 20)}px {16 if self.mobile else 36}px;background:{bg};color:{fg};border-bottom:1px solid {self.line}'><span>{logo}{nav}</span><span>{right}</span></div>"

    def product(self, h: int) -> str:
        c, rng = self.c, self.c.rng
        name, _ = rng.choice(STEP.PRODUCTS[c.lang])
        return (f"<div style='background:{self.card};border:1px solid {self.line};border-radius:{self.radius}px;overflow:hidden'><div style='height:{h}px'>{c.photo()}</div><div style='padding:10px 12px'><div style='font-weight:600'>{esc(name)}</div>"
                f"<div style='color:{self.muted};font-size:12px'>{esc(rng.choice(STEP.BRANDS))}</div><div style='display:flex;justify-content:space-between;align-items:center;margin-top:8px'><b>{c.money(rng.uniform(9, 240))}</b>{self.button(c.t('add_to_cart'), rng.random() < 0.6, 'padding:6px 10px;font-size:12px')}</div></div></div>")

    def body(self, layout: str) -> str:
        c, rng, pad = self.c, self.c.rng, 16 if self.mobile else 36
        if layout == "shop":
            cols = 2 if self.mobile else rng.choice([3, 4, 5])
            hero = f"<div style='height:{rng.randint(120, 220)}px;border-radius:{self.radius}px;overflow:hidden;margin-bottom:20px;position:relative'>{c.photo()}<div style='position:absolute;left:24px;bottom:18px;color:#fff;font-size:28px;font-weight:800;text-shadow:0 2px 8px rgba(0,0,0,.6)'>{esc(c.title(2, 4))}</div></div>" if rng.random() < 0.5 else ""
            return f"<div style='padding:{pad}px'>{hero}<div style='font-size:22px;font-weight:700;margin-bottom:14px'>{c.t('products')}</div><div style='display:grid;grid-template-columns:repeat({cols},1fr);gap:18px'>{''.join(self.product(rng.choice([120, 150, 180])) for _ in range(cols * 3))}</div></div>"
        if layout == "article":
            paras = "".join(f"<p style='margin:0 0 16px'>{esc(c.para(rng.randint(3, 6)))}</p>" for _ in range(8))
            return (f"<div style='max-width:{rng.choice([680, 760, 860])}px;margin:0 auto;padding:{pad}px'><div style='color:{c.acc};font-weight:700;text-transform:uppercase;font-size:12px;letter-spacing:1px'>{c.t('news')}</div>"
                    f"<h1 style='font-size:{26 if self.mobile else rng.randint(32, 44)}px;line-height:1.15;margin:8px 0 10px'>{esc(c.title(5, 10))}</h1><div style='color:{self.muted};margin-bottom:16px'>{esc(c.name())} · {c.date()}</div>"
                    f"<div style='height:{rng.randint(200, 340)}px;border-radius:{self.radius}px;overflow:hidden;margin-bottom:20px'>{c.photo()}</div><div style='font-size:17px;line-height:1.65'>{paras}</div></div>")
        if layout == "results":
            rows = ""
            for _ in range(9):
                thumb = f"<div style='width:{rng.choice([90, 120])}px;height:76px;border-radius:{self.radius}px;overflow:hidden;flex:none'>{c.photo()}</div>" if rng.random() < 0.5 else ""
                rows += f"<div style='display:flex;gap:14px;padding:14px 0;border-bottom:1px solid {self.line}'>{thumb}<div><div style='color:{self.muted};font-size:12px'>{esc(self.name.lower().replace(' ', ''))}.test › {esc(c.words(1))}</div><div style='color:{c.acc};font-size:18px;font-weight:600'>{esc(c.title(3, 7))}</div><div style='color:{self.muted}'>{esc(c.sentence(10, 20))}</div></div></div>"
            bar = f"<div style='padding:11px 18px;border:1px solid {self.line};border-radius:24px;background:{self.card};margin-bottom:8px'>{esc(c.words(rng.randint(1, 3)))}</div><div style='color:{self.muted};font-size:12px'>{rng.randint(12, 9000)} {c.t('results')}</div>"
            return f"<div style='max-width:820px;padding:{pad}px'>{bar}{rows}</div>"
        if layout == "landing":
            feats = "".join(f"<div style='flex:1'><div style='width:40px;height:40px;border-radius:10px;background:{hsl(c.hue + 40 * i, 60, 55)};margin-bottom:10px'></div><b>{esc(c.title(2, 3))}</b><div style='color:{self.muted};margin-top:4px'>{esc(c.sentence(8, 14))}</div></div>" for i in range(3))
            art = "" if self.mobile else f"<div style='flex:1;height:320px;border-radius:{self.radius + 6}px;overflow:hidden'>{c.photo()}</div>"
            return (f"<div style='padding:{pad + 20}px {pad}px'><div style='display:flex;gap:40px;align-items:center'><div style='flex:1'><h1 style='font-size:{32 if self.mobile else rng.randint(40, 58)}px;line-height:1.05;margin:0 0 14px'>{esc(c.title(4, 7))}</h1>"
                    f"<div style='font-size:18px;color:{self.muted};margin-bottom:22px'>{esc(c.sentence(10, 18))}</div>{self.button(c.t('get_started'))} &nbsp; {self.button(c.t('read_more'), False)}</div>{art}</div><div style='display:flex;gap:30px;margin-top:56px;{'flex-direction:column' if self.mobile else ''}'>{feats}</div></div>")
        if layout == "dashboard":
            stats = "".join(f"<div style='flex:1;background:{self.card};border:1px solid {self.line};border-radius:{self.radius}px;padding:16px'><div style='color:{self.muted};font-size:12px'>{c.t(k)}</div><div style='font-size:26px;font-weight:700'>{rng.randint(100, 98000):,}</div><div style='color:{'#1a8f3c' if rng.random() < 0.6 else '#c62828'};font-size:12px'>{rng.choice('+-')}{rng.uniform(0.5, 30):.1f}%</div></div>" for k in ("revenue", "users", "orders"))
            chart = "".join(f"<div style='flex:1;background:{c.acc};opacity:{rng.uniform(0.5, 1):.2f};height:{rng.randint(20, 100)}%;border-radius:3px 3px 0 0'></div>" for _ in range(rng.randint(12, 24)))
            rows = "".join(f"<div style='display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid {self.line}'><span>{esc(Ctx(rng, c.lang).name())}</span><span style='color:{self.muted}'>{c.date()}</span><b>{c.money(rng.uniform(10, 900))}</b></div>" for _ in range(6))
            current = f"background:{hsl(c.hue, 50, 22 if self.dark else 92)};font-weight:700;"
            menu = "".join(f"<div style='padding:9px 22px;{current if i == 0 else ''}'>{c.t(k)}</div>" for i, k in enumerate(("overview", "orders", "users", "revenue", "products", "contact")))
            side = "" if self.mobile else f"<div style='width:200px;background:{self.card};border-right:1px solid {self.line};padding:18px 0'>{menu}</div>"
            return f"<div style='display:flex;min-height:100%'>{side}<div style='flex:1;padding:{pad - 8}px'><div style='font-size:22px;font-weight:700;margin-bottom:14px'>{c.t('overview')}</div><div style='display:flex;gap:16px;{'flex-direction:column' if self.mobile else ''}'>{stats}</div><div style='background:{self.card};border:1px solid {self.line};border-radius:{self.radius}px;padding:16px;margin:16px 0;height:200px;display:flex;align-items:flex-end;gap:6px'>{chart}</div><div style='background:{self.card};border:1px solid {self.line};border-radius:{self.radius}px;padding:6px 16px'>{rows}</div></div></div>"
        tree = "".join(f"<div style='padding:5px 0;{'font-weight:700;color:' + c.acc if i == 2 else ''};padding-left:{rng.choice([0, 0, 14])}px'>{esc(c.title(1, 3))}</div>" for i in range(14))
        code = f"<div style='background:{'#0d1117' if self.dark else '#f3f4f6'};border-radius:8px;padding:14px;font:13px/1.5 {c.mono};margin:14px 0;white-space:pre'>$ npm install {esc(c.words(1))}\n$ {esc(c.words(1))} init --{esc(c.words(1))}</div>"
        content = f"<h1 style='font-size:32px;margin:0 0 10px'>{esc(c.title(2, 4))}</h1><p style='color:{self.muted};font-size:17px'>{esc(c.sentence(10, 18))}</p>" + "".join(f"<h2 style='font-size:21px;margin:24px 0 8px'>{esc(c.title(2, 4))}</h2><p>{esc(c.para(3))}</p>{code if rng.random() < 0.6 else ''}" for _ in range(3))
        side = "" if self.mobile else f"<div style='width:230px;padding:24px 22px;border-right:1px solid {self.line};color:{self.muted}'><b style='color:{self.fg}'>{c.t('docs')}</b>{tree}</div>"
        return f"<div style='display:flex'>{side}<div style='flex:1;padding:{pad}px;max-width:760px;line-height:1.6'>{content}</div></div>"


def page_html(rng: random.Random, state: str, layout: str, lang: str, vw: int, vh: int, photos: list[str]) -> str:
    c = Ctx(rng, lang, photos)
    s = Site(c, vw)
    backdrop = lambda inner, a=0.5: f"<div style='position:fixed;inset:0;background:rgba(0,0,0,{a});display:grid;place-items:center;z-index:9'>{inner}</div>"
    panel = lambda inner, w: f"<div style='background:{s.card};color:{s.fg};border-radius:{max(s.radius, 6)}px;box-shadow:0 20px 60px rgba(0,0,0,.4);padding:26px;width:{min(w, vw - 32)}px;box-sizing:border-box;position:relative'>{inner}</div>"
    field = lambda k: f"<div style='margin:12px 0'><div style='font-size:12px;color:{s.muted};margin-bottom:4px'>{k}</div><div style='border:1px solid {s.muted};border-radius:{s.radius}px;height:38px;background:{s.bg}'></div></div>"
    wrap = lambda inner, bg=None: f"<body style='margin:0;font:14px/1.45 {s.font};color:{s.fg};background:{bg or s.bg};min-height:100vh;overflow:hidden'>{inner}</body>"
    base = s.header() + s.body(layout)
    if state == "ready":
        chat = f"<div style='position:fixed;right:22px;bottom:22px;width:54px;height:54px;border-radius:50%;background:{c.acc};box-shadow:0 4px 14px rgba(0,0,0,.3)'></div>" if rng.random() < 0.2 else ""
        return wrap(base + chat)
    if state == "cookies":
        buttons = f"{s.button(c.t('accept'))} {s.button(c.t('reject'), rng.random() < 0.3)}" + (f" {s.button(c.t('manage'), False)}" if rng.random() < 0.5 else "")
        text = f"<b style='font-size:16px'>{c.t('cookies_title')}</b><div style='color:{s.muted};margin:6px 0 12px'>{c.t('cookies_text')}</div>"
        where = rng.choice(["bottom", "bottom", "top", "corner", "modal", "modal"])
        if where == "modal":
            return wrap(base + backdrop(panel(text + buttons, 520), rng.uniform(0.35, 0.65)))
        if where == "corner":
            return wrap(base + f"<div style='position:fixed;{rng.choice(['left', 'right'])}:20px;bottom:20px;width:{min(380, vw - 40)}px;background:{s.card};color:{s.fg};border:1px solid {s.line};border-radius:{max(s.radius, 6)}px;box-shadow:0 10px 40px rgba(0,0,0,.3);padding:20px;z-index:9;box-sizing:border-box'>{text}{buttons}</div>")
        tone = rng.choice([(s.card, s.fg), ("#15181c", "#fff"), (c.acc, "#fff")])
        return wrap(base + f"<div style='position:fixed;left:0;right:0;{where}:0;background:{tone[0]};color:{tone[1]};box-shadow:0 0 30px rgba(0,0,0,.3);padding:{rng.randint(16, 30)}px {16 if s.mobile else 40}px;z-index:9;display:flex;gap:24px;align-items:center;{'flex-direction:column' if s.mobile else ''}'><div style='flex:1'>{text}</div><div style='white-space:nowrap'>{buttons}</div></div>")
    if state == "popup":
        close = f"<span style='position:absolute;right:14px;top:8px;font-size:22px;color:{s.muted}'>×</span>"
        art = f"<div style='height:150px;margin:-26px -26px 18px;overflow:hidden;border-radius:{max(s.radius, 6)}px {max(s.radius, 6)}px 0 0'>{c.photo()}</div>" if rng.random() < 0.5 else ""
        inner = f"{art}{close}<div style='font-size:24px;font-weight:800;text-align:center'>{c.t('newsletter')}</div><div style='color:{s.muted};text-align:center;margin:8px 0 14px'>{c.t('newsletter_text')}</div>{field(c.t('email'))}<div style='text-align:center'>{s.button(c.t('subscribe'), True, 'width:80%')}<div style='margin-top:10px;color:{s.muted};text-decoration:underline'>{c.t('no_thanks')}</div></div>"
        return wrap(base + backdrop(panel(inner, rng.choice([420, 480, 560])), rng.uniform(0.4, 0.7)))
    if state == "signin":
        social = "".join(f"<div style='border:1px solid {s.muted};border-radius:{s.radius}px;padding:9px;text-align:center;margin-top:8px'>{esc(rng.choice(STEP.BRANDS))}</div>" for _ in range(rng.choice([0, 0, 1, 2])))
        form = f"<div style='text-align:center'><span style='display:inline-block;width:40px;height:40px;border-radius:10px;background:{c.acc}'></span><div style='font-size:22px;font-weight:800;margin-top:10px'>{c.t('welcome_back')}</div><div style='color:{s.muted}'>{c.t('sign_in_to')}</div></div>{field(c.t('email'))}{field(c.t('password'))}{s.button(c.t('sign_in'), True, 'display:block;text-align:center')}<div style='text-align:center;margin-top:12px;color:{c.acc}'>{c.t('forgot')}</div>{social}"
        how = rng.choice(["page", "page", "split", "modal"])
        if how == "modal":
            return wrap(base + backdrop(panel(form, 400), rng.uniform(0.45, 0.7)))
        if how == "split" and not s.mobile:
            return wrap(f"<div style='display:flex;height:100vh'><div style='flex:1;display:grid;place-items:center'>{panel(form, 380)}</div><div style='flex:1'>{c.photo()}</div></div>")
        bg = rng.choice([s.bg, f"linear-gradient({rng.randint(0, 360)}deg,{hsl(c.hue, 50, 40)},{hsl(c.hue + 60, 50, 30)})", "#f0f2f5"])
        return wrap(f"<div style='min-height:100vh;display:grid;place-items:center'>{panel(form, 400)}</div>", bg)
    if state == "error":
        how = rng.choice(["site", "site", "browser", "bare"])
        if how == "browser":
            return wrap(f"<div style='max-width:560px;margin:{vh // 5}px auto 0;padding:0 24px;color:#5f6368'><div style='width:64px;height:72px;border:3px solid #9aa0a6;border-radius:6px;position:relative'><div style='position:absolute;left:14px;top:20px;width:8px;height:8px;background:#9aa0a6;box-shadow:22px 0 #9aa0a6'></div><div style='position:absolute;left:16px;bottom:16px;width:28px;height:3px;background:#9aa0a6'></div></div>"
                        f"<div style='font-size:24px;color:#202124;margin:24px 0 10px'>{c.t('cant_reach')}</div><div>{esc(s.name.lower().replace(' ', ''))}.test – {c.t('not_found_text')}</div><div style='font:12px {c.mono};margin:16px 0'>ERR_{rng.choice(['CONNECTION_REFUSED', 'NAME_NOT_RESOLVED', 'TIMED_OUT', 'CONNECTION_RESET'])}</div><span style='display:inline-block;background:#1a73e8;color:#fff;padding:8px 18px;border-radius:4px'>{c.t('try_again')}</span></div>", "#fff")
        code = rng.choice(["404", "404", "500", "503", "403"])
        msg = c.t("not_found") if code == "404" else c.t("went_wrong")
        core = f"<div style='text-align:center;padding:{vh // 7}px 20px'><div style='font-size:{rng.randint(90, 160)}px;font-weight:900;line-height:1;color:{c.acc if rng.random() < 0.6 else s.muted}'>{code}</div><div style='font-size:26px;font-weight:700;margin:12px 0 6px'>{msg}</div><div style='color:{s.muted};margin-bottom:22px'>{c.t('not_found_text')}</div>{s.button(c.t('go_home'))} {s.button(c.t('try_again'), False) if rng.random() < 0.4 else ''}</div>"
        return wrap((s.header() if how == "site" else "") + core)
    if state == "loading":
        how = rng.choice(["skeleton", "skeleton", "spinner", "bar"])
        tone = "#2a3038" if s.dark else "#e4e7eb"
        block = lambda w, h, r=6: f"<div style='width:{w};height:{h}px;border-radius:{r}px;background:linear-gradient(90deg,{tone},{'#353c46' if s.dark else '#f1f3f5'},{tone});margin-bottom:12px'></div>"
        if how == "skeleton":
            cols = 2 if s.mobile else rng.choice([3, 4])
            cards = "".join(f"<div>{block('100%', rng.choice([120, 160]))}{block('80%', 14)}{block('50%', 14)}</div>" for _ in range(cols * 2))
            return wrap(s.header() + f"<div style='padding:{16 if s.mobile else 36}px'>{block('40%', 28)}{block('70%', 14)}<div style='display:grid;grid-template-columns:repeat({cols},1fr);gap:20px;margin-top:22px'>{cards}</div></div>")
        if how == "spinner":
            return wrap(f"<div style='height:100vh;display:grid;place-items:center'><div style='text-align:center'><div style='width:{rng.randint(36, 64)}px;aspect-ratio:1;border-radius:50%;border:5px solid {tone};border-top-color:{c.acc};margin:0 auto 14px'></div><div style='color:{s.muted}'>{c.t('loading')}</div></div></div>")
        return wrap(f"<div style='height:3px;width:{rng.randint(20, 80)}%;background:{c.acc}'></div>" + s.header() + f"<div style='padding:60px;color:{s.muted};text-align:center'>{c.t('loading')}</div>")
    if state == "robot":
        how = rng.choice(["checkbox", "checkbox", "grid"])
        box = f"<div style='display:flex;align-items:center;justify-content:space-between;border:1px solid #d3d3d3;background:#f9f9f9;color:#222;border-radius:4px;padding:14px 16px;width:280px;box-shadow:0 0 4px rgba(0,0,0,.1)'><span style='display:flex;align-items:center;gap:14px;font-size:15px'><span style='width:26px;height:26px;border:2px solid #b8b8b8;border-radius:3px;background:#fff;display:inline-block'></span>{c.t('robot')}</span><span style='width:34px;height:34px;border-radius:50%;border:4px solid #4a90e2;border-right-color:#bbb;display:inline-block'></span></div>"
        if how == "checkbox":
            return wrap(f"<div style='max-width:560px;margin:{vh // 6}px auto 0;padding:0 24px'><div style='font-size:{rng.randint(24, 34)}px;font-weight:700;margin-bottom:8px'>{c.t('verify')}</div><div style='color:{s.muted};margin-bottom:22px'>{esc(s.name.lower().replace(' ', ''))}.test · {c.t('verify_text')}</div>{box}<div style='color:{s.muted};font-size:11px;margin-top:40px;border-top:1px solid {s.line};padding-top:10px'>Ray ID: {rng.randint(10 ** 15, 10 ** 16):x}</div></div>")
        tiles = "".join(f"<div style='aspect-ratio:1;overflow:hidden'>{c.photo()}</div>" for _ in range(9))
        card = f"<div style='width:{min(340, vw - 30)}px;background:#fff;color:#222;box-shadow:0 10px 40px rgba(0,0,0,.4)'><div style='background:#4a90e2;color:#fff;padding:18px'>{c.t('select_all')}<div style='font-size:22px;font-weight:800'>{c.t('things')}</div></div><div style='display:grid;grid-template-columns:repeat(3,1fr);gap:4px;padding:6px'>{tiles}</div><div style='display:flex;justify-content:flex-end;padding:8px'><span style='background:#4a90e2;color:#fff;padding:9px 20px;border-radius:2px;font-weight:700'>OK</span></div></div>"
        return wrap(base + backdrop(card, rng.uniform(0.4, 0.7)))
    # paywall: an article that fades out, then the offer
    s2 = s.body("article")
    prices = "".join(f"<div style='border:2px solid {c.acc if i == 0 else s.line};border-radius:8px;padding:12px 18px'><b style='font-size:20px'>{c.money(rng.uniform(2, 15))}</b><div style='color:{s.muted};font-size:12px'>{c.t('per_month')}</div></div>"
                     for i in range(rng.choice([1, 2, 3])))
    offer = f"<div style='font-size:24px;font-weight:800'>{c.t('paywall')}</div><div style='color:{s.muted};margin:8px 0 16px'>{c.t('paywall_text')}</div><div style='display:flex;gap:12px;justify-content:center;margin-bottom:12px'>{prices}</div>{s.button(c.t('subscribe'))}<div style='margin-top:12px;color:{s.muted}'>{c.t('already')}</div>"
    top = rng.randint(vh // 4, vh // 2)
    return wrap(s.header() + f"<div style='height:{top}px;overflow:hidden'>{s2}</div><div style='margin-top:-140px;height:140px;position:relative;background:linear-gradient(transparent,{s.bg})'></div><div style='text-align:center;padding:20px 24px 60px;border-top:1px solid {s.line};background:{s.bg};position:relative'>{offer}</div>")


# ── buttons ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def control_html(rng: random.Random, svg: str, set_name: str, style: str) -> tuple[str, int]:
    """A button with only this icon on it, on a bit of page: (html, the margin to keep around the button in the crop)."""
    dark = rng.random() < 0.3
    hue = rng.uniform(0, 360)
    page = rng.choice(["#111418", "#1c1f24", "#0f172a", hsl(hue, 30, 16)]) if dark else rng.choice(["#fff", "#fff", "#f6f7f9", "#fafaf7", hsl(hue, 30, 96)])
    fg = rng.choice(["#f3f4f6", "#e5e7eb", "#fff"]) if dark else rng.choice(["#111827", "#1f2937", "#374151", "#000", "#4b5563"])
    look = rng.choices(["ghost", "outline", "soft", "solid"], [0.42, 0.2, 0.18, 0.2])[0]
    accent = hsl(hue, rng.uniform(55, 85), rng.uniform(38, 55))
    bg, border, color = {"ghost": ("transparent", "0", fg if rng.random() < 0.8 else accent), "outline": ("transparent" if rng.random() < 0.6 else ("#1f242b" if dark else "#fff"), f"1px solid {'#3b424c' if dark else '#d1d5db'}", fg),
                         "soft": (hsl(hue, 45, 22) if dark else hsl(hue, 55, 93), "0", accent if not dark else "#fff"), "solid": (accent, "0", "#fff")}[look]
    size, pad = rng.choice([14, 16, 16, 18, 20, 20, 24, 24, 28, 32]), rng.choice([4, 6, 8, 8, 10, 12])
    radius = rng.choice(["0", "4px", "6px", "8px", "50%", "999px"])
    wide = rng.choice([0, 0, 0, 6, 12]) if radius != "50%" else 0
    if "<svg" not in svg:
        raise ValueError("not an svg")
    head, rest = svg[svg.index("<svg") :].split(">", 1)
    if "fill=" not in head and "stroke=" not in head:
        head += " fill='currentColor'"  # sets drawn as filled shapes that leave the colour to the page
    stroke = f"stroke-width:{rng.choice([1.25, 1.5, 1.75, 2, 2.25, 2.5])};" if set_name in ("lucide", "feather", "tabler") and style == "line" and rng.random() < 0.6 else ""
    icon = f"{head} style='width:{size}px;height:{size}px;display:block;{stroke}'>{rest}"
    badge = f"<span style='position:absolute;top:-3px;right:-3px;min-width:{rng.choice([8, 14, 16])}px;height:{rng.choice([8, 14, 16])}px;border-radius:99px;background:{rng.choice(['#ef4444', '#f97316', accent])};color:#fff;font:bold 9px Arial;display:grid;place-items:center'>{rng.choice(['', '', str(rng.randint(1, 9)), str(rng.randint(10, 99))])}</span>" if rng.random() < 0.1 else ""
    button = f"<span id='t' style='position:relative;display:inline-grid;place-items:center;padding:{pad}px {pad + wide}px;background:{bg};border:{border};border-radius:{radius};color:{color};opacity:{rng.choice([1, 1, 1, 1, 0.55])}'>{icon}{badge}</span>"
    strip = f"background:{'#1c2026' if dark else '#fff'};border-bottom:1px solid {'#2d333b' if dark else '#e5e7eb'};" if rng.random() < 0.3 else ""
    return f"<body style='margin:0;background:{page}'><div style='padding:40px 60px;{strip}'>{button}</div></body>", rng.choice([0, 0, 1, 2, 3, 4, 6, 8])


# ── what is drawn, and in which split ────────────────────────────────────────────────────────────────────────────────

def held_out_language(kind: str, kinds: list[str]) -> str:
    """One non-English language per kind is never trained on for that kind (as step/synth.py HELD_OUT_LANG)."""
    return LANGS[1 + sorted(kinds).index(kind) % 4]


def photo_urls(split: str, n: int, seed: int) -> list[str]:
    """Photos for pages to show, as data URLs: training photos for training pages, held-out ones for the rest."""
    from store import Pack

    pack = Pack(CACHE / "photos")
    keys = sorted(k for k, r in pack.rows.items() if (r["split"] == "train") == (split == "train"))
    rng = random.Random(seed)
    return ["data:image/jpeg;base64," + base64.b64encode(pack.raw(k)).decode() for k in rng.sample(keys, min(n, len(keys)))] if keys else []


def plan(family: str, scale: float = 1.0) -> list[dict]:
    """Every picture of a family: its id, its label, its split, and what to draw it with."""
    out = []
    if family == "control":
        for job, per_set in sorted(icon_sets.files().items()):
            for set_name, styles in sorted(per_set.items()):
                for style, path in styles:
                    if job in icon_sets.HELD_OUT_JOBS:
                        counts = {"heldout-kinds": 3}
                    elif set_name in icon_sets.HELD_OUT_SETS:
                        counts = {"heldout-look": 4}
                    else:
                        counts = {"train": 12, "heldout-items": 1}
                    for split, n in counts.items():
                        for i in range(max(1, round(n * scale))):
                            out.append({"id": f"{job}-{set_name}-{style}-{split}-{i}", "family": "control", "label": job, "split": split, "set": set_name, "style": style, "svg": str(path)})
        for set_name, paths in sorted(icon_sets.others().items()):
            split = "heldout-none" if set_name in icon_sets.HELD_OUT_SETS else "train"
            for path in paths:
                for i in range(max(1, round((4 if split == "train" else 3) * scale))):
                    out.append({"id": f"none-{set_name}-{path.stem}-{i}", "family": "control", "label": NONE, "split": split, "set": set_name, "style": "line", "svg": str(path)})
    elif family == "document":
        kinds = sorted(DOCS)
        for kind in kinds:
            n = round((220 if kind in HELD_OUT_DOCUMENTS else 420) * scale)
            for i in range(max(2, n)):
                rng = random.Random(f"document/{kind}/{i}")
                lang = rng.choices(LANGS, LANG_WEIGHTS)[0]
                split = "heldout-kinds" if kind in HELD_OUT_DOCUMENTS else "heldout-look" if lang == held_out_language(kind, kinds) else "heldout-items" if i % 10 == 0 else "train"
                out.append({"id": f"{kind}-{i}", "family": "document", "label": kind, "split": split, "lang": lang})
    elif family == "page":
        states = sorted(PAGES)
        for state in states:
            n = round((260 if state in HELD_OUT_PAGES else 620) * scale)
            for i in range(max(2, n)):
                rng = random.Random(f"page/{state}/{i}")
                lang, layout = rng.choices(LANGS, LANG_WEIGHTS)[0], rng.choice(LAYOUTS)
                split = "heldout-kinds" if state in HELD_OUT_PAGES else "heldout-look" if layout in HELD_OUT_LAYOUTS or lang == held_out_language(state, states) else "heldout-items" if i % 10 == 0 else "train"
                out.append({"id": f"{state}-{i}", "family": "page", "label": state, "split": split, "lang": lang, "layout": layout})
    else:
        raise ValueError(family)
    return out


def spec(item: dict, photos: dict[str, list[str]]) -> dict:
    """The page to draw for one planned picture (deterministic: the id is the seed)."""
    rng = random.Random(f"{item['family']}/{item['id']}")
    pool = photos.get("train" if item["split"] == "train" else "heldout", [])
    if item["family"] == "control":
        html, margin = control_html(rng, Path(item["svg"]).read_text(encoding="utf-8"), item["set"], item["style"])
        return {"id": item["id"], "html": html, "width": 400, "height": 200, "scale": rng.choices([1, 2, 3], [0.3, 0.6, 0.1])[0], "clip": "#t", "margin": margin}
    if item["family"] == "document":
        html, w, h, shown = document_page(rng, item["label"], item["lang"], pool)
        item["shown"] = shown
        return {"id": item["id"], "html": html, "width": min(w, 1300), "height": h, "scale": 1, "clip": "#scene"}
    vw, vh = rng.choice(VIEWPORTS)
    item["viewport"] = [vw, vh]
    return {"id": item["id"], "html": page_html(rng, item["label"], item["layout"], item["lang"], vw, vh, pool), "width": vw, "height": vh, "scale": 1}


PACKS = {"control": "controls", "document": "documents", "page": "pages"}
NONE = "none"  # a button whose icon is none of the 63 jobs


def render(families: list[str], scale: float, browser: str | None) -> None:
    from PIL import Image

    from preprocess import flatten, resize
    from store import Pack, jpeg
    import numpy as np

    photos = {"train": photo_urls("train", 400, 1), "heldout": photo_urls("heldout", 200, 2)}
    for family in families:
        pack = Pack(CACHE / PACKS[family])
        todo = [i for i in plan(family, scale) if i["id"] not in pack]
        print(f"{family}: {len(todo)} pictures to draw ({len(pack)} already there)", flush=True)
        chunk = 1500 if family == "control" else 300  # a whole page's PNG is about 1 MB until it is packed
        for start in range(0, len(todo), chunk):
            items = todo[start : start + chunk]
            with tempfile.TemporaryDirectory() as tmp:
                specs = Path(tmp) / "specs.jsonl"
                specs.write_text("".join(json.dumps(spec(i, photos)) + "\n" for i in items), encoding="utf-8")
                cmd = ["node", str(ROOT / "vision" / "render.mjs"), "--specs", str(specs), "--out", str(Path(tmp) / "png")] + (["--browser", browser] if browser else [])
                subprocess.run(cmd, check=False)
                for item in items:
                    png = Path(tmp) / "png" / f"{item['id']}.png"
                    if not png.exists():
                        continue
                    meta = {k: v for k, v in item.items() if k not in ("id", "svg")}
                    if family == "control":
                        pack.add(item["id"], png.read_bytes(), **meta)  # small already: kept as drawn
                    else:
                        with Image.open(png) as im:
                            pixels = np.asarray(im.convert("RGB"))
                        pack.add(item["id"], jpeg(resize(flatten(pixels))), w=pixels.shape[1], h=pixels.shape[0], **meta)
            print(f"  {family}: {len(pack)} packed", flush=True)


def sheet(out: Path, browser: str | None) -> None:
    """A few of each kind, as files to look at."""
    out.mkdir(parents=True, exist_ok=True)
    photos = {"train": photo_urls("train", 40, 1), "heldout": photo_urls("heldout", 40, 2)}
    items = []
    for family in PACKS:
        seen: dict[str, int] = {}
        for item in plan(family, 1.0):
            if seen.get(item["label"], 0) < (1 if family == "control" else 2):
                seen[item["label"]] = seen.get(item["label"], 0) + 1
                items.append(item)
    specs = out / "specs.jsonl"
    specs.write_text("".join(json.dumps(spec(i, photos)) + "\n" for i in items), encoding="utf-8")
    subprocess.run(["node", str(ROOT / "vision" / "render.mjs"), "--specs", str(specs), "--out", str(out)] + (["--browser", browser] if browser else []), check=False)
    specs.unlink()


# ── questions ────────────────────────────────────────────────────────────────────────────────────────────────────────

_ICON_WORDS = icon_sets.words()
_DOC_WORDS = documents()


def _control_words(label: str, lang: str) -> dict:
    name, does = _ICON_WORDS[label][lang]
    return {"x": name, "name": name, "does": does + "."}


def _document_words(label: str, lang: str) -> dict:
    x, does = _DOC_WORDS[label][lang]
    from classes import bare

    return {"x": x, "name": bare(x, lang), "does": does + "."}


def _page_words(label: str, lang: str) -> dict:
    w = PAGES[label][lang]
    return {"x": w["say"][0], "name": w["say"][0], "say": w["say"], "ask": w["ask"], "other": PAGE_OTHER[lang]}


CONTROL, DOCUMENT, PAGE = Family("control", _control_words), Family("document", _document_words), Family("page", _page_words)
SPACES = {"control": (CONTROL, sorted(_ICON_WORDS), icon_sets.HELD_OUT_JOBS), "document": (DOCUMENT, sorted(DOCS), HELD_OUT_DOCUMENTS), "page": (PAGE, sorted(PAGES), HELD_OUT_PAGES)}


def synth_families() -> dict[str, Family]:
    return {"control": CONTROL, "document": DOCUMENT, "page": PAGE}


def item_questions(family: str, key: str, row: dict, rng: random.Random, unseen: bool, every_job: bool = False) -> list[dict]:
    """A drawn picture's questions: which of these is it, is it this (yes), is it that (no)."""
    fam_, labels, held = SPACES[family]
    kinds = row["split"] == "heldout-kinds"
    label = row["label"]
    from data import half_of

    if label == NONE:
        # None of the jobs: every "is it this?" is a no, and Atelier's own question has no right answer to be sure of.
        asked = rng.choice([x for x in labels if row["split"] != "train" or x not in held])
        out = [fam_.noul(rng, language(rng), asked, False, unseen)]
        if unseen:
            lang = language(rng)
            q = {"type": "choice", "instructions": "What does this button do?", "criteria": {job: _ICON_WORDS[job][lang][0] for job in labels}}
            out.append({"q": q, "answer": None, "kind": "choice", "form": "every job", "lang": lang, "unseen_wording": False, "n": len(labels)})
        for r in out:
            r.update(family=family, label=NONE, image=f"{PACKS[family]}/{key}", split=row["split"], half=half_of(f"{family}/{key}"))
        return out
    pool = [x for x in labels if x != label and (kinds or x not in held)]
    if kinds:  # at least half of a new kind's wrong options are new kinds too
        new = [x for x in pool if x in held]
        rng.shuffle(new)
        old = [x for x in pool if x not in held]
        rng.shuffle(old)
        n = rng.choice([1, 2, 3, 3, 4, 5, 7])
        others = (new[: max(1, n // 2)] + old)[:n]
    else:
        others = rng.sample(pool, min(len(pool), rng.choice([1, 2, 3, 3, 4, 5, 7])))
    out = [fam_.choice(rng, language(rng), label, others, unseen), fam_.noul(rng, language(rng), label, True, unseen), fam_.noul(rng, language(rng), rng.choice(pool), False, unseen)]
    if family == "control" and unseen:
        # Atelier's own question about a button it found no name for (its browser_steps tool): which of all 63 jobs, by
        # name, in the page's language. Only asked of held-out pictures: it is what that use is measured by.
        lang = language(rng)
        q = {"type": "choice", "instructions": "What does this button do?", "criteria": {job: _ICON_WORDS[job][lang][0] for job in labels}}
        out.append({"q": q, "answer": label, "kind": "choice", "form": "every job", "lang": lang, "unseen_wording": False, "n": len(labels)})
    elif family == "control" and every_job and rng.random() < 0.3:
        # In training, sometimes the same question among many jobs at once (the answer and 31 other trained ones: all 55
        # of them in every such question would not fit the GPU's memory), so many options are nothing new.
        lang = language(rng)
        trained = [job for job in labels if job not in held and job != label]
        trained = sorted(rng.sample(trained, 31) + [label])
        q = {"type": "choice", "instructions": "What does this button do?", "criteria": {job: _ICON_WORDS[job][lang][0] for job in trained}}
        out.append({"q": q, "answer": label, "kind": "choice", "form": "every job", "lang": lang, "unseen_wording": False, "n": len(trained)})
    for r in out:
        r.update(family=family, label=label, image=f"{PACKS[family]}/{key}", split=row["split"], half=half_of(f"{family}/{key}"))
    return out


def synth_rows(split: str, seed: int, unseen: bool, every_job: bool = False, none: bool = True) -> list[dict]:
    """every_job: training also asks a button's job among every trained one. none: with the buttons that have none."""
    from store import Pack

    rows = []
    for family in PACKS:
        pack = Pack(CACHE / PACKS[family])
        rng = random.Random(zlib.crc32(f"{family}/{split}/{seed}".encode()))
        for key, row in pack.rows.items():
            if row["split"] == split and (none or row["label"] != NONE):
                rows += item_questions(family, key, row, rng, unseen, every_job)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["render", "sheet", "plan"])
    parser.add_argument("--family", nargs="+", default=list(PACKS))
    parser.add_argument("--scale", type=float, default=1.0, help="how many pictures, as a share of the full plan")
    parser.add_argument("--browser", help="a Chromium to draw with (default: the first one found)")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "vision" / "sheet")
    args = parser.parse_args()
    if args.cmd == "plan":
        for family in args.family:
            counts: dict[str, int] = {}
            for i in plan(family, args.scale):
                counts[i["split"]] = counts.get(i["split"], 0) + 1
            print(family, counts)
    elif args.cmd == "sheet":
        sheet(args.out, args.browser)
    else:
        if not shutil.which("node"):
            raise SystemExit("node is needed to draw (vision/render.mjs)")
        render(args.family, args.scale, args.browser)


if __name__ == "__main__":
    main()
