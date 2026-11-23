# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Numeric browser decisions with answers computed by code: budgets, totals, ratings, dates, opening hours, file
sizes, limits, unit prices, counts. Pages in five languages with their own number and date formats; questions in
English or the page's language.

The teachers waver here (is $119 + $38.40 + $31.07 under $180?), so these answers come from arithmetic, not from a
teacher. Every row keeps its true answer and how close it is to the threshold (meta.truth, meta.margin: near, mid,
far). Five families are held out whole (a test of new kinds of numeric questions); the trained families also get
fresh held-out instances.

    python data/build_numeric.py --out data/pool/pool-numeric.jsonl --test data/pool/numeric-test.jsonl \
        --truth-dir data/teacher
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from collections import Counter
from pathlib import Path

LANGS = ["en", "nl", "de", "fr", "es"]
LANG_WEIGHTS = [0.4, 0.2, 0.15, 0.125, 0.125]
MONTHS = {
    "en": "January February March April May June July August September October November December".split(),
    "nl": "januari februari maart april mei juni juli augustus september oktober november december".split(),
    "de": "Januar Februar März April Mai Juni Juli August September Oktober November Dezember".split(),
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split(),
    "es": "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split(),
}
DAYS = {
    "en": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
    "nl": "maandag dinsdag woensdag donderdag vrijdag zaterdag zondag".split(),
    "de": "Montag Dienstag Mittwoch Donnerstag Freitag Samstag Sonntag".split(),
    "fr": "lundi mardi mercredi jeudi vendredi samedi dimanche".split(),
    "es": "lunes martes miércoles jueves viernes sábado domingo".split(),
}
W = {  # page words
    "subtotal": dict(en="Subtotal", nl="Subtotaal", de="Zwischensumme", fr="Sous-total", es="Subtotal"),
    "shipping": dict(en="Shipping", nl="Verzendkosten", de="Versand", fr="Livraison", es="Envío"),
    "free": dict(en="Free", nl="Gratis", de="Kostenlos", fr="Offerte", es="Gratis"),
    "discount": dict(en="Discount", nl="Korting", de="Rabatt", fr="Remise", es="Descuento"),
    "total": dict(en="Order total", nl="Totaal", de="Gesamtsumme", fr="Total", es="Total"),
    "today": dict(en="Today", nl="Vandaag", de="Heute", fr="Aujourd'hui", es="Hoy"),
    "cart": dict(en="Your cart", nl="Winkelwagen", de="Warenkorb", fr="Votre panier", es="Tu carrito"),
    "add": dict(en="[Add to cart]", nl="[In winkelwagen]", de="[In den Warenkorb]", fr="[Ajouter au panier]", es="[Añadir al carrito]"),
    "checkout": dict(en="[Checkout]", nl="[Afrekenen]", de="[Zur Kasse]", fr="[Commander]", es="[Tramitar pedido]"),
    "stock": dict(en="In stock", nl="Op voorraad", de="Auf Lager", fr="En stock", es="En stock"),
    "qty": dict(en="Quantity", nl="Aantal", de="Menge", fr="Quantité", es="Cantidad"),
    "closed": dict(en="Closed", nl="Gesloten", de="Geschlossen", fr="Fermé", es="Cerrado"),
    "now": dict(en="Now", nl="Nu", de="Jetzt", fr="Maintenant", es="Ahora"),
}
PRODUCTS = ["Sony WH-1000XM5 headphones", "Philips Airfryer XL", "KALLAX shelf unit", "Bosch GSR 12V drill",
            "Nike Pegasus 41", "LEGO Technic 42151", "Samsung 27\" monitor", "Kindle Paperwhite", "Logitech MX Keys",
            "De'Longhi Magnifica S", "Garmin Forerunner 265", "Dyson V12 vacuum", "Fjällräven Kånken backpack",
            "Moccamaster KBG", "JBL Flip 6 speaker", "Tefal frying pan set", "Brabantia pedal bin 30 L",
            "Anker 737 power bank", "Weber kettle grill 57 cm", "Smeg kettle", "Ray-Ban Wayfarer", "Stanley 1.4 L flask"]
NAMES = ["Anna", "Ben", "Chloé", "Daan", "Elif", "Finn", "Greta", "Hugo", "Inés", "Jonas", "Kim", "Lars", "Mila",
         "Noah", "Olga", "Pieter", "Quinn", "Rosa", "Sam", "Tess", "Umar", "Vera", "Wout", "Yara", "Zoë"]


def num(x: float, lang: str, decimals: int = 2) -> str:
    s = f"{x:,.{decimals}f}"
    if lang == "en":
        return s
    if lang == "fr":
        return s.replace(",", " ").replace(".", ",")
    return s.replace(",", "#").replace(".", ",").replace("#", ".")


def money(x: float, lang: str, cur: str = "€", decimals: int = 2) -> str:
    n = num(x, lang, decimals)
    if lang == "en":
        return f"{'-' if x < 0 else ''}{cur}{num(abs(x), lang, decimals)}"
    if lang == "nl":
        return f"€ {n}"
    return f"{n} €"


def date(d: dt.date, lang: str, rng: random.Random) -> str:
    style, m = rng.random(), d.month - 1
    if style < 0.2:
        return d.isoformat()
    if lang == "en":
        return f"{MONTHS['en'][m]} {d.day}, {d.year}" if style < 0.7 else f"{d.day} {MONTHS['en'][m][:3]} {d.year}"
    long = {"nl": f"{d.day} {MONTHS['nl'][m]} {d.year}", "de": f"{d.day}. {MONTHS['de'][m]} {d.year}",
            "fr": f"{d.day} {MONTHS['fr'][m]} {d.year}", "es": f"{d.day} de {MONTHS['es'][m]} de {d.year}"}[lang]
    short = {"nl": f"{d.day:02d}-{d.month:02d}-{d.year}", "de": f"{d.day:02d}.{d.month:02d}.{d.year}",
             "fr": f"{d.day:02d}/{d.month:02d}/{d.year}", "es": f"{d.day:02d}/{d.month:02d}/{d.year}"}[lang]
    return long if style < 0.7 else short


def clock_style(rng: random.Random, lang: str) -> str:
    """One time format per page: 12-hour (English), 14h30 (French) or 24-hour."""
    return "12" if lang == "en" and rng.random() < 0.5 else ("h" if lang == "fr" and rng.random() < 0.5 else "24")


def clock(minutes: int, style: str) -> str:
    h, m = divmod(minutes, 60)
    if style == "12":
        return f"{(h - 1) % 12 + 1}:{m:02d} {'AM' if h < 12 else 'PM'}"
    return f"{h}h{m:02d}" if style == "h" else f"{h:02d}:{m:02d}"


def today(rng: random.Random) -> dt.date:
    return dt.date(2026, 1, 1) + dt.timedelta(days=rng.randrange(700))


def bucket(rng: random.Random) -> str:
    return rng.choices(["near", "mid", "far"], [0.3, 0.35, 0.35])[0]


def rel(rng: random.Random, margin: str) -> float:
    """A relative distance from the threshold for a margin bucket."""
    return {"near": rng.uniform(0.01, 0.03), "mid": rng.uniform(0.04, 0.15), "far": rng.uniform(0.2, 0.6)}[margin]


def ask(rng: random.Random, lang: str, templates: dict[str, list[str]], **kw) -> str:
    q_lang = "en" if lang == "en" or rng.random() < 0.55 else lang
    return rng.choice(templates[q_lang]).format(**kw)


def noul(text: str) -> dict:
    return {"type": "noul", "instructions": text}


def side(rng: random.Random) -> bool:
    return rng.random() < 0.5


# Each family: (rng, lang) -> (state, question, truth, margin). Truth is "true"/"false" for noul, a key for choice.

def price_vs_budget(rng, lang):
    cur = rng.choice("$£€") if lang == "en" else "€"
    budget = rng.choice([20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 150, 200, 250, 300, 400, 500, 800, 1000])
    margin, ok = bucket(rng), side(rng)
    price = round(budget * (1 - rel(rng, margin) if ok else 1 + rel(rng, margin)), 2)
    if rng.random() < 0.5:
        price = round(price) - 0.01 if round(price) - 0.01 > 0 and (round(price) - 0.01 <= budget) == ok else price
    p = rng.choice(PRODUCTS)
    lines = [p, money(price, lang, cur), W["stock"][lang]]
    if rng.random() < 0.4:
        lines.insert(2, f"★ {num(rng.uniform(3.6, 4.9), lang, 1)}")
    state = "\n".join(lines + [W["add"][lang]])
    q = ask(rng, lang, {
        "en": ["Is the price at or below {b}?", "Can I buy this for {b} or less?", "Does this fit a budget of {b}?"],
        "nl": ["Kost dit {b} of minder?", "Past dit binnen een budget van {b}?"],
        "de": ["Kostet das höchstens {b}?", "Passt das in ein Budget von {b}?"],
        "fr": ["Est-ce que ça coûte {b} ou moins ?", "Est-ce que ça rentre dans un budget de {b} ?"],
        "es": ["¿Cuesta {b} o menos?", "¿Entra en un presupuesto de {b}?"]}, b=money(budget, lang, cur, 0))
    return state, noul(q), str(price <= budget).lower(), margin


def _cart(rng, lang, show_total):
    cur = rng.choice("$£€") if lang == "en" else "€"
    items = rng.sample(PRODUCTS, rng.randint(1, 3))
    amounts = [round(rng.uniform(4, 180), 2) for _ in items]
    qty = [rng.choice([1, 1, 1, 2, 3]) for _ in items]
    ship = rng.choice([0, 0, 3.95, 4.99, 5.95, 6.5, 9.99, 12.99])
    disc = round(sum(a * q for a, q in zip(amounts, qty)) * rng.choice([0, 0, 0.1, 0.15]), 2)
    total = round(sum(a * q for a, q in zip(amounts, qty)) + ship - disc, 2)
    margin, ok = bucket(rng), side(rng)
    budget = round(total / (1 - rel(rng, margin)) if ok else total / (1 + rel(rng, margin)))
    if (total <= budget) != ok:
        budget = int(total) + 1 if ok else int(total)
    gap = abs(total - budget) / budget
    margin = "near" if gap < 0.035 else ("mid" if gap < 0.17 else "far")
    lines = [W["cart"][lang]]
    for p, a, q in zip(items, amounts, qty):
        lines.append(f"{p} — {q} × {money(a, lang, cur)}" if q > 1 else f"{p} — {money(a, lang, cur)}")
    lines.append(f"{W['shipping'][lang]}: {money(ship, lang, cur) if ship else W['free'][lang]}")
    if disc:
        lines.append(f"{W['discount'][lang]}: {money(-disc, lang, cur)}")
    if show_total:
        lines.append(f"{W['total'][lang]}: {money(total, lang, cur)}")
    lines.append(W["checkout"][lang])
    q = ask(rng, lang, {
        "en": ["Will the order total, shipping included, stay within {b}?", "Is everything in this cart together {b} or less?",
               "Can I check out without going over {b}?"],
        "nl": ["Blijft het totaalbedrag inclusief verzendkosten binnen {b}?", "Kan ik afrekenen zonder boven {b} uit te komen?"],
        "de": ["Bleibt die Gesamtsumme inklusive Versand bei höchstens {b}?", "Kann ich bestellen, ohne {b} zu überschreiten?"],
        "fr": ["Le total, livraison comprise, reste-t-il dans la limite de {b} ?", "Puis-je commander sans dépasser {b} ?"],
        "es": ["¿El total, envío incluido, se queda en {b} o menos?", "¿Puedo pagar sin pasarme de {b}?"]},
        b=money(budget, lang, cur, 0))
    return "\n".join(lines), noul(q), str(total <= budget).lower(), margin


def cart_sum_vs_budget(rng, lang):
    return _cart(rng, lang, False)


def cart_total_shown_vs_budget(rng, lang):
    return _cart(rng, lang, True)


def rating_vs_minimum(rng, lang):
    minimum = rng.choice([3.5, 4.0, 4.2, 4.5, 4.7])
    margin, ok = rng.choices(["near", "mid", "far"], [0.35, 0.35, 0.3])[0], side(rng)
    step = {"near": rng.choice([0.0, 0.1]) if ok else 0.1, "mid": rng.choice([0.2, 0.3]), "far": rng.choice([0.5, 0.7, 0.9])}[margin]
    rating = round(min(5.0, minimum + step) if ok else max(1.0, minimum - step), 1)
    n = rng.randint(3, 9000)
    stars = {"en": f"{num(rating, lang, 1)} out of 5 stars ({num(n, lang, 0)} reviews)",
             "nl": f"{num(rating, lang, 1)} van 5 sterren ({num(n, lang, 0)} beoordelingen)",
             "de": f"{num(rating, lang, 1)} von 5 Sternen ({num(n, lang, 0)} Bewertungen)",
             "fr": f"{num(rating, lang, 1)} sur 5 étoiles ({num(n, lang, 0)} avis)",
             "es": f"{num(rating, lang, 1)} de 5 estrellas ({num(n, lang, 0)} opiniones)"}[lang]
    state = "\n".join([rng.choice(PRODUCTS), stars, money(rng.uniform(15, 400), lang), W["add"][lang]])
    q = ask(rng, lang, {
        "en": ["Is the rating at least {m} stars?", "Does this meet my minimum of {m} stars?"],
        "nl": ["Heeft dit minstens {m} sterren?", "Haalt dit mijn minimum van {m} sterren?"],
        "de": ["Hat das mindestens {m} Sterne?", "Erreicht das mein Minimum von {m} Sternen?"],
        "fr": ["La note est-elle d'au moins {m} étoiles ?"], "es": ["¿Tiene al menos {m} estrellas?"]},
        m=num(minimum, "en" if lang == "en" else lang, 1))
    return state, noul(q), str(rating >= minimum).lower(), margin


def _days(rng, margin, ok):
    """Days past a limit: ok means within the limit."""
    d = {"near": rng.choice([0, 1]) if ok else 1, "mid": rng.randint(2, 5), "far": rng.randint(8, 60)}[margin]
    return -d if ok else d


def return_window_open(rng, lang):
    t, n = today(rng), rng.choice([7, 14, 14, 28, 30, 30, 60, 100])
    margin, ok = bucket(rng), side(rng)
    elapsed = n + _days(rng, margin, ok)
    delivered = t - dt.timedelta(days=max(0, elapsed))
    lines = {"en": [f"Order #{rng.randint(10000, 99999)}", f"Delivered: {date(delivered, lang, rng)}", f"Returns accepted within {n} days of delivery"],
             "nl": [f"Bestelling {rng.randint(10000, 99999)}", f"Bezorgd op: {date(delivered, lang, rng)}", f"Retourneren kan binnen {n} dagen na ontvangst"],
             "de": [f"Bestellung {rng.randint(10000, 99999)}", f"Zugestellt am: {date(delivered, lang, rng)}", f"Rücksendung innerhalb von {n} Tagen nach Erhalt"],
             "fr": [f"Commande n° {rng.randint(10000, 99999)}", f"Livrée le : {date(delivered, lang, rng)}", f"Retours acceptés sous {n} jours après réception"],
             "es": [f"Pedido n.º {rng.randint(10000, 99999)}", f"Entregado el: {date(delivered, lang, rng)}", f"Devoluciones en un plazo de {n} días desde la entrega"]}[lang]
    state = "\n".join(lines + [f"{W['today'][lang]}: {date(t, lang, rng)}"])
    q = ask(rng, lang, {
        "en": ["Can this order still be returned today?", "Is the return window still open?"],
        "nl": ["Kan deze bestelling vandaag nog geretourneerd worden?"], "de": ["Kann die Bestellung heute noch zurückgeschickt werden?"],
        "fr": ["Cette commande peut-elle encore être retournée aujourd'hui ?"], "es": ["¿Todavía se puede devolver este pedido hoy?"]})
    return state, noul(q), str((t - delivered).days <= n).lower(), margin


def deadline_passed(rng, lang):
    t = today(rng)
    margin, late = bucket(rng), side(rng)
    gap = {"near": 1, "mid": rng.randint(2, 5), "far": rng.randint(8, 90)}[margin]
    due = t - dt.timedelta(days=gap) if late else t + dt.timedelta(days=gap)
    amount = money(rng.uniform(12, 900), lang)
    lines = {"en": [f"Invoice {rng.randint(2025, 2027)}-{rng.randint(100, 9999)}", f"Amount due: {amount}", f"Due date: {date(due, lang, rng)}"],
             "nl": [f"Factuur {rng.randint(100, 9999)}", f"Te betalen: {amount}", f"Vervaldatum: {date(due, lang, rng)}"],
             "de": [f"Rechnung Nr. {rng.randint(100, 9999)}", f"Offener Betrag: {amount}", f"Fällig am: {date(due, lang, rng)}"],
             "fr": [f"Facture n° {rng.randint(100, 9999)}", f"Montant dû : {amount}", f"Date d'échéance : {date(due, lang, rng)}"],
             "es": [f"Factura n.º {rng.randint(100, 9999)}", f"Importe pendiente: {amount}", f"Fecha de vencimiento: {date(due, lang, rng)}"]}[lang]
    t_line = f"{W['today'][lang]}: {DAYS[lang][t.weekday()]} {date(t, lang, rng)}"
    state = "\n".join([t_line] + lines if rng.random() < 0.5 else lines + [t_line])
    q = ask(rng, lang, {
        "en": ["Is this invoice overdue?", "Has the payment deadline passed?"], "nl": ["Is de vervaldatum van deze factuur verstreken?"],
        "de": ["Ist die Rechnung überfällig?"], "fr": ["Cette facture est-elle en retard de paiement ?"], "es": ["¿Esta factura está vencida?"]})
    return state, noul(q), str(t > due).lower(), margin


def open_now(rng, lang):
    cs = clock_style(rng, lang)
    week_open, week_close = rng.choice([8, 9, 10]) * 60 + rng.choice([0, 30]), rng.choice([17, 18, 19, 20, 21]) * 60 + rng.choice([0, 30])
    sat_open, sat_close = rng.choice([9, 10]) * 60, rng.choice([14, 16, 17, 18]) * 60
    sunday_open = rng.random() < 0.3
    hours = {d: (week_open, week_close) for d in range(5)} | {5: (sat_open, sat_close), 6: (12 * 60, 17 * 60) if sunday_open else None}
    day = rng.randrange(7)
    margin = bucket(rng)
    span = hours[day]
    if span is None:
        minute, margin, truth = rng.randint(9 * 60, 19 * 60), "far", False
    else:
        edge = rng.choice(span)
        d = {"near": rng.randint(1, 15), "mid": rng.randint(16, 60), "far": rng.randint(90, 240)}[margin] * rng.choice([-1, 1])
        minute = max(6 * 60, min(23 * 60 + 30, edge + d))
        truth = span[0] <= minute < span[1]
        margin = "near" if min(abs(minute - span[0]), abs(minute - span[1])) <= 15 else ("mid" if min(abs(minute - span[0]), abs(minute - span[1])) <= 60 else "far")
    names = DAYS[lang]
    lines = []
    if rng.random() < 0.6:
        lines.append(f"{names[0]}–{names[4]}: {clock(week_open, cs)}–{clock(week_close, cs)}")
    else:
        lines += [f"{names[i]}: {clock(week_open, cs)}–{clock(week_close, cs)}" for i in range(5)]
    lines.append(f"{names[5]}: {clock(sat_open, cs)}–{clock(sat_close, cs)}")
    lines.append(f"{names[6]}: {clock(12 * 60, cs)}–{clock(17 * 60, cs)}" if sunday_open else f"{names[6]}: {W['closed'][lang]}")
    state = "\n".join([rng.choice(["Bakkerij De Korenbloem", "Fahrradladen Speiche", "Librairie du Parc", "Ferretería López",
                                   "Corner Hardware", "Apotheek Centrum", "Bäckerei Krume", "Optique Lumière"])] + lines
                      + [f"{W['now'][lang]}: {names[day]} {clock(minute, cs)}"])
    q = ask(rng, lang, {
        "en": ["Is the shop open right now?", "Can I go there now and find it open?"], "nl": ["Is de winkel nu open?"],
        "de": ["Hat das Geschäft gerade geöffnet?"], "fr": ["Le magasin est-il ouvert en ce moment ?"], "es": ["¿La tienda está abierta ahora mismo?"]})
    return state, noul(q), str(truth).lower(), margin


def upload_size_allowed(rng, lang):
    limit_mb = rng.choice([2, 5, 10, 20, 25, 50, 100, 250, 1000, 2000])
    margin, ok = bucket(rng), side(rng)
    r = rel(rng, margin) + (0.03 if margin == "near" else 0)  # keep clear of the 1000-vs-1024 ambiguity
    size_mb = limit_mb * (1 - r if ok else 1 + r)

    def show(mb):
        if mb < 1 and rng.random() < 0.8:
            return f"{num(mb * 1000, lang, 0)} KB"
        if mb >= 1000 and rng.random() < 0.7:
            return f"{num(mb / 1000, lang, 2)} GB"
        return f"{num(mb, lang, 1)} MB"
    limit = show(limit_mb).replace(",0 ", " ").replace(".0 ", " ").replace(",00 ", " ").replace(".00 ", " ")
    name = rng.choice(["scan_passport.pdf", "holiday.mov", "thesis_final_v3.docx", "invoice-0425.pdf", "backup.zip", "IMG_4021.HEIC", "presentation.pptx"])
    lines = {"en": [f"Upload a file (max. {limit})", f"Selected: {name} — {show(size_mb)}"],
             "nl": [f"Bestand uploaden (max. {limit})", f"Gekozen: {name} — {show(size_mb)}"],
             "de": [f"Datei hochladen (max. {limit})", f"Ausgewählt: {name} — {show(size_mb)}"],
             "fr": [f"Envoyer un fichier ({limit} max.)", f"Sélectionné : {name} — {show(size_mb)}"],
             "es": [f"Subir archivo (máx. {limit})", f"Seleccionado: {name} — {show(size_mb)}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Is the selected file small enough to upload?", "Will this file fit under the upload limit?"],
        "nl": ["Is het gekozen bestand klein genoeg om te uploaden?"], "de": ["Ist die ausgewählte Datei klein genug zum Hochladen?"],
        "fr": ["Le fichier sélectionné est-il assez petit pour être envoyé ?"], "es": ["¿El archivo seleccionado es lo bastante pequeño para subirlo?"]})
    return "\n".join(lines + ["[Upload]"]), noul(q), str(size_mb <= limit_mb).lower(), margin


def per_customer_limit(rng, lang):
    limit = rng.choice([1, 2, 3, 4, 5, 6, 10])
    margin, ok = rng.choices(["near", "mid", "far"], [0.45, 0.35, 0.2])[0], side(rng)
    d = {"near": rng.choice([0, 1]) if ok else 1, "mid": rng.randint(2, 3), "far": rng.randint(4, 9)}[margin]
    n = limit - d if ok else limit + d
    if n < 1:
        n, ok = 1, 1 <= limit
    rule = {"en": f"Limit {limit} per customer", "nl": f"Maximaal {limit} per klant", "de": f"Maximal {limit} pro Kunde",
            "fr": f"Limité à {limit} par client", "es": f"Máximo {limit} por cliente"}[lang]
    state = "\n".join([rng.choice(PRODUCTS), money(rng.uniform(10, 600), lang), rule, f"{W['qty'][lang]}: {n}", W["add"][lang]])
    q = ask(rng, lang, {
        "en": ["Is the quantity I picked allowed?", "Can I order this many?"], "nl": ["Mag ik dit aantal bestellen?"],
        "de": ["Darf ich diese Menge bestellen?"], "fr": ["Puis-je commander cette quantité ?"], "es": ["¿Puedo pedir esta cantidad?"]})
    return state, noul(q), str(n <= limit).lower(), margin


def data_usage_over(rng, lang):
    total = rng.choice([5, 10, 15, 20, 30, 50, 100])
    p = rng.choice([50, 75, 80, 90])
    margin, over = bucket(rng), side(rng)
    used = total * p / 100 * (1 + rel(rng, margin) if over else 1 - rel(rng, margin))
    used = min(used, total * 1.2)
    lines = {"en": ["Mobile data", f"{num(used, lang, 1)} GB of {total} GB used this month"],
             "nl": ["Mobiele data", f"{num(used, lang, 1)} GB van {total} GB verbruikt deze maand"],
             "de": ["Mobile Daten", f"{num(used, lang, 1)} GB von {total} GB in diesem Monat verbraucht"],
             "fr": ["Données mobiles", f"{num(used, lang, 1)} Go sur {total} Go utilisés ce mois-ci"],
             "es": ["Datos móviles", f"{num(used, lang, 1)} GB de {total} GB usados este mes"]}[lang]
    q = ask(rng, lang, {
        "en": ["Has more than {p}% of the data allowance been used?"], "nl": ["Is meer dan {p}% van de databundel verbruikt?"],
        "de": ["Ist mehr als {p} % des Datenvolumens verbraucht?"], "fr": ["Plus de {p} % du forfait data a-t-il été utilisé ?"],
        "es": ["¿Se ha usado más del {p} % de los datos?"]}, p=p)
    shown = round(used, 1)
    return "\n".join(lines), noul(q), str(shown > total * p / 100).lower(), margin


def cheapest_offer(rng, lang):
    unit = rng.choice([("capsules", 0.35), ("cans", 0.9), ("rolls", 0.5), ("tablets", 0.12), ("batteries", 0.8), ("filters", 2.5)])
    sizes = sorted(rng.sample([4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 40, 48, 50, 60, 100], 3))
    prices = [round(s * unit[1] * rng.uniform(0.75, 1.25), 2) for s in sizes]
    per = [p / s for p, s in zip(prices, sizes)]
    order = sorted(range(3), key=lambda i: per[i])
    gap = (per[order[1]] - per[order[0]]) / per[order[0]]
    margin = "near" if gap < 0.03 else ("mid" if gap < 0.15 else "far")
    keys = [f"pack-{s}" for s in sizes]
    word = {"en": "pack", "nl": "pak", "de": "Packung", "fr": "paquet", "es": "paquete"}[lang]
    lines = [f"{unit[0].capitalize()} — {word} {s}: {money(p, lang)}" for s, p in zip(sizes, prices)]
    rng.shuffle(lines)
    q = {"type": "choice", "instructions": ask(rng, lang, {
        "en": ["Which pack has the lowest price per piece?", "Which option is cheapest per unit?"],
        "nl": ["Welk pak heeft de laagste prijs per stuk?"], "de": ["Welche Packung ist pro Stück am günstigsten?"],
        "fr": ["Quel paquet a le prix à l'unité le plus bas ?"], "es": ["¿Qué paquete tiene el precio por unidad más bajo?"]}),
        "criteria": {k: f"the pack of {s}" for k, s in zip(keys, sizes)}}
    return "\n".join(lines), q, keys[order[0]], margin


def currency_converted_within_budget(rng, lang):
    rate = round(rng.uniform(0.82, 0.97), 2)  # EUR per USD
    budget = rng.choice([50, 80, 100, 150, 200, 300, 500])
    margin, ok = bucket(rng), side(rng)
    eur = budget * (1 - rel(rng, margin) if ok else 1 + rel(rng, margin))
    usd = round(eur / rate, 2)
    truth = usd * rate <= budget
    lines = {"en": [rng.choice(PRODUCTS), f"Price: US${num(usd, lang)}", f"Exchange rate: 1 USD = {num(rate, lang)} EUR"],
             "nl": [rng.choice(PRODUCTS), f"Prijs: US$ {num(usd, lang)}", f"Wisselkoers: 1 USD = {num(rate, lang)} EUR"],
             "de": [rng.choice(PRODUCTS), f"Preis: {num(usd, lang)} US$", f"Wechselkurs: 1 USD = {num(rate, lang)} EUR"],
             "fr": [rng.choice(PRODUCTS), f"Prix : {num(usd, lang)} $ US", f"Taux de change : 1 USD = {num(rate, lang)} EUR"],
             "es": [rng.choice(PRODUCTS), f"Precio: {num(usd, lang)} US$", f"Tipo de cambio: 1 USD = {num(rate, lang)} EUR"]}[lang]
    q = ask(rng, lang, {
        "en": ["Converted to euros, is this within my budget of €{b}?"], "nl": ["Past dit omgerekend in euro binnen mijn budget van € {b}?"],
        "de": ["Liegt das umgerechnet in Euro innerhalb meines Budgets von {b} €?"], "fr": ["Converti en euros, est-ce dans mon budget de {b} € ?"],
        "es": ["Convertido a euros, ¿entra en mi presupuesto de {b} €?"]}, b=budget)
    return "\n".join(lines), noul(q), str(truth).lower(), margin


def count_at_least(rng, lang):
    people = rng.sample(NAMES, rng.randint(4, 12))
    going = [rng.random() < 0.6 for _ in people]
    yes = sum(going)
    margin = rng.choices(["near", "mid", "far"], [0.45, 0.35, 0.2])[0]
    d = {"near": rng.choice([0, 1]), "mid": rng.randint(2, 3), "far": rng.randint(4, 6)}[margin]
    n = max(1, yes - d) if side(rng) else yes + max(1, d)
    mark = {"en": ("✓ going", "✗ can't make it"), "nl": ("✓ komt", "✗ kan niet"), "de": ("✓ kommt", "✗ kann nicht"),
            "fr": ("✓ vient", "✗ ne peut pas"), "es": ("✓ va", "✗ no puede")}[lang]
    title = {"en": "Friday dinner — RSVPs", "nl": "Etentje vrijdag — aanmeldingen", "de": "Abendessen am Freitag — Zusagen",
             "fr": "Dîner de vendredi — réponses", "es": "Cena del viernes — respuestas"}[lang]
    state = "\n".join([title] + [f"{p}: {mark[0] if g else mark[1]}" for p, g in zip(people, going)])
    q = ask(rng, lang, {
        "en": ["Have at least {n} people said they are coming?"], "nl": ["Hebben minstens {n} mensen gezegd dat ze komen?"],
        "de": ["Haben mindestens {n} Leute zugesagt?"], "fr": ["Au moins {n} personnes ont-elles dit qu'elles venaient ?"],
        "es": ["¿Al menos {n} personas han dicho que vienen?"]}, n=n)
    return state, noul(q), str(yes >= n).lower(), "near" if abs(yes - n) <= 1 else ("mid" if abs(yes - n) <= 3 else "far")


# Held out whole: new kinds of numeric questions.

def age_old_enough(rng, lang):
    t, minimum = today(rng), rng.choice([13, 16, 18, 21])
    margin, ok = bucket(rng), side(rng)
    days = {"near": rng.randint(1, 3), "mid": rng.randint(5, 40), "far": rng.randint(200, 2000)}[margin]
    birthday_at_min = dt.date(t.year - minimum, t.month, min(t.day, 28))
    dob = birthday_at_min - dt.timedelta(days=days) if ok else birthday_at_min + dt.timedelta(days=days)
    age = t.year - dob.year - ((t.month, t.day) < (dob.month, dob.day))
    lines = {"en": ["Create your account", f"Date of birth: {date(dob, lang, rng)}", f"You must be at least {minimum} to sign up."],
             "nl": ["Account aanmaken", f"Geboortedatum: {date(dob, lang, rng)}", f"Je moet minstens {minimum} jaar zijn om je aan te melden."],
             "de": ["Konto erstellen", f"Geburtsdatum: {date(dob, lang, rng)}", f"Du musst mindestens {minimum} Jahre alt sein."],
             "fr": ["Créer un compte", f"Date de naissance : {date(dob, lang, rng)}", f"Vous devez avoir au moins {minimum} ans."],
             "es": ["Crear cuenta", f"Fecha de nacimiento: {date(dob, lang, rng)}", f"Debes tener al menos {minimum} años."]}[lang]
    state = "\n".join(lines + [f"{W['today'][lang]}: {date(t, lang, rng)}"])
    q = ask(rng, lang, {
        "en": ["Is this person old enough to sign up?"], "nl": ["Is deze persoon oud genoeg om zich aan te melden?"],
        "de": ["Ist die Person alt genug für die Anmeldung?"], "fr": ["Cette personne a-t-elle l'âge requis pour s'inscrire ?"],
        "es": ["¿Esta persona tiene edad suficiente para registrarse?"]})
    return state, noul(q), str(age >= minimum).lower(), margin


def discount_at_least(rng, lang):
    old = round(rng.uniform(20, 900), 2)
    p = rng.choice([10, 20, 25, 30, 40, 50])
    margin, ok = bucket(rng), side(rng)
    frac = p / 100 * (1 + rel(rng, margin) if ok else 1 - rel(rng, margin))
    new = round(old * (1 - frac), 2)
    lines = {"en": [rng.choice(PRODUCTS), f"Was {money(old, lang)}  Now {money(new, lang)}"],
             "nl": [rng.choice(PRODUCTS), f"Van {money(old, lang)} voor {money(new, lang)}"],
             "de": [rng.choice(PRODUCTS), f"Statt {money(old, lang)} jetzt {money(new, lang)}"],
             "fr": [rng.choice(PRODUCTS), f"Avant {money(old, lang)}  Maintenant {money(new, lang)}"],
             "es": [rng.choice(PRODUCTS), f"Antes {money(old, lang)}  Ahora {money(new, lang)}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Is the discount at least {p}%?"], "nl": ["Is de korting minstens {p}%?"], "de": ["Beträgt der Rabatt mindestens {p} %?"],
        "fr": ["La remise est-elle d'au moins {p} % ?"], "es": ["¿El descuento es de al menos un {p} %?"]}, p=p)
    return "\n".join(lines + [W["add"][lang]]), noul(q), str((old - new) / old >= p / 100).lower(), margin


def delivery_radius(rng, lang):
    radius = rng.choice([3, 5, 8, 10, 15, 20, 25, 40])
    margin, ok = bucket(rng), side(rng)
    d = round(radius * (1 - rel(rng, margin) if ok else 1 + rel(rng, margin)), 1)
    lines = {"en": [f"We deliver within {radius} km of our kitchen.", f"Distance to your address: {num(d, lang, 1)} km"],
             "nl": [f"We bezorgen binnen {radius} km van onze keuken.", f"Afstand tot je adres: {num(d, lang, 1)} km"],
             "de": [f"Wir liefern im Umkreis von {radius} km.", f"Entfernung zu deiner Adresse: {num(d, lang, 1)} km"],
             "fr": [f"Nous livrons dans un rayon de {radius} km.", f"Distance jusqu'à votre adresse : {num(d, lang, 1)} km"],
             "es": [f"Repartimos a menos de {radius} km del local.", f"Distancia a tu dirección: {num(d, lang, 1)} km"]}[lang]
    q = ask(rng, lang, {
        "en": ["Will they deliver to my address?"], "nl": ["Bezorgen ze op mijn adres?"], "de": ["Liefern sie an meine Adresse?"],
        "fr": ["Livrent-ils à mon adresse ?"], "es": ["¿Reparten a mi dirección?"]})
    return "\n".join(lines), noul(q), str(d <= radius).lower(), margin


def group_fits_seats(rng, lang):
    cs = clock_style(rng, lang)
    seats = rng.randint(1, 12)
    margin, ok = rng.choices(["near", "mid", "far"], [0.45, 0.35, 0.2])[0], side(rng)
    d = {"near": rng.choice([0, 1]) if ok else 1, "mid": rng.randint(2, 3), "far": rng.randint(4, 7)}[margin]
    n = seats - d if ok else seats + d
    if n < 2:
        n = 2
    left = {"en": f"Only {seats} seats left", "nl": f"Nog maar {seats} plaatsen beschikbaar", "de": f"Nur noch {seats} Plätze frei",
            "fr": f"Plus que {seats} places disponibles", "es": f"Solo quedan {seats} asientos"}[lang]
    t = today(rng)
    state = "\n".join([rng.choice(["Dune: Part Three", "Hamlet", "Jazz at the Bimhuis", "Stand-up night", "Swan Lake"]),
                       f"{DAYS[lang][t.weekday()]} {date(t, lang, rng)} {clock(rng.choice([19, 20, 21]) * 60 + rng.choice([0, 15, 30]), cs)}", left, "[Book]"])
    q = ask(rng, lang, {
        "en": ["We are {n} people. Can we all book seats for this showing?"], "nl": ["We zijn met {n}. Kunnen we allemaal een plek boeken?"],
        "de": ["Wir sind {n} Personen. Können wir alle Plätze buchen?"], "fr": ["Nous sommes {n}. Pouvons-nous tous réserver ?"],
        "es": ["Somos {n}. ¿Podemos reservar todos?"]}, n=n)
    return state, noul(q), str(n <= seats).lower(), margin


def meeting_fits(rng, lang):
    cs = clock_style(rng, lang)
    start = rng.randint(8 * 4, 17 * 4) * 15
    length = rng.choice([30, 45, 60, 90, 120])
    margin, ok = bucket(rng), side(rng)
    gap = {"near": 5, "mid": rng.choice([10, 15, 20, 30]), "far": rng.choice([60, 90, 120])}[margin]
    end = start + length
    nxt = end + gap if ok else end - gap
    title1 = rng.choice(["Project review", "1:1 with Sam", "Budget call", "Design critique", "Client demo"])
    title2 = rng.choice(["Dentist", "School pickup", "Gym class", "Train to Utrecht", "Team lunch"])
    state = "\n".join([f"{DAYS[lang][rng.randrange(5)]}", f"{clock(start, cs)}–{clock(end, cs)} {title1}",
                       f"{clock(nxt, cs)} {title2}"])
    q = ask(rng, lang, {
        "en": ["Does the first meeting end before the next appointment starts?"], "nl": ["Is de eerste afspraak afgelopen voordat de volgende begint?"],
        "de": ["Endet der erste Termin, bevor der nächste beginnt?"], "fr": ["La première réunion se termine-t-elle avant le rendez-vous suivant ?"],
        "es": ["¿Termina la primera reunión antes de que empiece la siguiente cita?"]})
    return state, noul(q), str(end <= nxt).lower(), margin


TRAINED = [price_vs_budget, cart_sum_vs_budget, cart_total_shown_vs_budget, rating_vs_minimum, return_window_open,
           deadline_passed, open_now, upload_size_allowed, per_customer_limit, data_usage_over, cheapest_offer,
           currency_converted_within_budget, count_at_least]
HELD_OUT = [age_old_enough, discount_at_least, delivery_radius, group_fits_seats, meeting_fits]


def rows_for(fn, n: int, rng: random.Random, split: str) -> list[dict]:
    out, seen = [], set()
    while len(out) < n:
        lang = rng.choices(LANGS, LANG_WEIGHTS)[0]
        state, q, truth, margin = fn(rng, lang)
        if (state, q["instructions"]) in seen:
            continue
        seen.add((state, q["instructions"]))
        out.append({"meta": {"use": "open", "source": "numeric", "family": fn.__name__.replace("_", "-"), "lang": lang,
                             "truth": truth, "margin": margin, "split": split},
                    "request": {"state": state, "questions": {"q": q}}})
    return out


def truth_row(row: dict, sure: float = 4.0) -> dict:
    """A teacher-format row whose distribution is the true answer (p = 0.98 for two options)."""
    q = row["request"]["questions"]["q"]
    labels = list(q["criteria"]) if q["type"] == "choice" else ["false", "true"]
    return {**row, "logits": {"q": [sure if label == row["meta"]["truth"] else 0.0 for label in labels]},
            "temperature": 1.0, "teacher": "truth"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, help="also write teacher-format rows labelled with the true answer")
    parser.add_argument("--per-family", type=int, default=600)
    parser.add_argument("--test-per-family", type=int, default=60)
    parser.add_argument("--held-out-per-family", type=int, default=100)
    args = parser.parse_args()
    rng = random.Random(2026)
    train, test = [], []
    for fn in TRAINED:
        rows = rows_for(fn, args.per_family + args.test_per_family, rng, "train")
        texts = set()
        for r in rows[: args.test_per_family]:
            r["meta"]["split"] = "family-seen"
            texts.add(r["request"]["state"])
        test += rows[: args.test_per_family]
        train += [r for r in rows[args.test_per_family:] if r["request"]["state"] not in texts]
    for fn in HELD_OUT:
        test += rows_for(fn, args.held_out_per_family, rng, "family-unseen")
    for path, rows in ((args.out, train), (args.test, test)):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    if args.truth_dir:
        for path, rows in ((args.out, train), (args.test, test)):
            (args.truth_dir / f"{path.stem}.truth.jsonl").write_text(
                "".join(json.dumps(truth_row(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for name, rows in (("train", train), ("test", test)):
        c = Counter((r["meta"]["family"], r["meta"]["truth"] in ("true",)) for r in rows)
        m = Counter(r["meta"]["margin"] for r in rows)
        print(f"{name}: {len(rows)} rows, margins {dict(m)}, langs {dict(Counter(r['meta']['lang'] for r in rows))}")
        for fam in sorted({f for f, _ in c}):
            print(f"  {fam:<34} true {c[(fam, True)]:>4}  other {c[(fam, False)]:>4}")


if __name__ == "__main__":
    main()
