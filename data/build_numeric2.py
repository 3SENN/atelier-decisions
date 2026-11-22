"""Numeric decisions, second suite: questions that take several steps (add up, multiply, convert units or
currencies, count days or business days, work across midnight), answered by code.

v0.4's mini compared single numbers well but failed when a step came first (a cart sum 63%, an age from a birth
date 37%, a discount percentage 59%, currency conversion sure and wrong 22%). This suite trains 20 multi-step
families and holds out five more, next to the five held-out families of data/build_numeric.py, which stay
untrained: age from a birth date and discount percentage are still tested only through families that resemble them
(warranty and passport dates, tip and price-increase percentages).

Rows whose answer would depend on a convention (does today count as a business day, is a GB 1000 or 1024 MB) are
generated under both conventions and dropped when they disagree.

    python data/build_numeric2.py --out data/pool/pool-numeric2.jsonl --test data/pool/numeric2-test.jsonl \
        --truth-dir data/teacher
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import build_numeric as bn
from build_numeric import DAYS, NAMES, PRODUCTS, W, ask, bucket, clock, clock_style, date, money, noul, num, rel, side, today


class Skip(Exception):
    """The row would depend on a convention or came out degenerate: draw another."""


def add_months(d: dt.date, n: int) -> dt.date:
    y, m = divmod(d.month - 1 + n, 12)
    y += d.year
    return dt.date(y, m + 1, min(d.day, calendar.monthrange(y, m + 1)[1]))


def grade(gap: float, near: float = 0.035, mid: float = 0.17) -> str:
    return "near" if gap < near else ("mid" if gap < mid else "far")


def grade_minutes(m: float) -> str:
    return "near" if m <= 10 else ("mid" if m <= 45 else "far")


def grade_days(d: int) -> str:
    return "near" if d <= 3 else ("mid" if d <= 14 else "far")


def cur_for(rng, lang):
    return rng.choice("$£€") if lang == "en" else "€"


def duration(minutes: int, lang: str) -> str:
    h, m = divmod(minutes, 60)
    hw = {"en": "h", "nl": "u", "de": "Std.", "fr": "h", "es": "h"}[lang]
    if lang == "fr":
        return f"{h} h {m:02d}" if h else f"{m} min"
    return (f"{h} {hw} {m} min" if m else f"{h} {hw}") if h else f"{m} min"


TODAY = W["today"]


def ask2(rng, lang, templates, **kw):
    """Like ask(), but a keyword given as {lang: text} is filled in the language the question ends up in."""
    q_lang = "en" if lang == "en" or rng.random() < 0.55 else lang
    kw = {k: (v[q_lang] if isinstance(v, dict) else v) for k, v in kw.items()}
    return rng.choice(templates[q_lang]).format(**kw)


def page_dates(rng, lang):
    """One date format per page."""
    seed = rng.randrange(1 << 30)
    return lambda d: date(d, lang, random.Random(seed))


def slug(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return "-".join("".join(c if c.isalnum() else " " for c in t).split())


# --------------------------------------------------------------------------------------------- trained families

def warranty_still_valid(rng, lang):
    t = today(rng)
    months = rng.choice([6, 12, 12, 24, 24, 36, 60])
    margin, ok = bucket(rng), side(rng)
    gap = {"near": rng.randint(1, 3), "mid": rng.randint(4, 14), "far": rng.randint(30, 300)}[margin]
    expiry = t + dt.timedelta(days=gap if ok else -gap)
    bought = add_months(expiry, -months)
    expiry = add_months(bought, months)
    if expiry == t:
        raise Skip
    span = (f"{months // 12} " + {"en": "years" if months > 12 else "year", "nl": "jaar", "de": "Jahre" if months > 12 else "Jahr",
                                  "fr": "ans" if months > 12 else "an", "es": "años" if months > 12 else "año"}[lang]
            if months % 12 == 0 and rng.random() < 0.7 else f"{months} " + {"en": "months", "nl": "maanden", "de": "Monate", "fr": "mois", "es": "meses"}[lang])
    p, pd = rng.choice(PRODUCTS), page_dates(rng, lang)
    lines = {"en": ["Order history", p, f"Purchased on {pd(bought)}", f"Warranty: {span} from the date of purchase"],
             "nl": ["Bestelgeschiedenis", p, f"Gekocht op {pd(bought)}", f"Garantie: {span} vanaf aankoopdatum"],
             "de": ["Bestellübersicht", p, f"Gekauft am {pd(bought)}", f"Garantie: {span} ab Kaufdatum"],
             "fr": ["Historique des commandes", p, f"Acheté le {pd(bought)}", f"Garantie : {span} à compter de l'achat"],
             "es": ["Historial de pedidos", p, f"Comprado el {pd(bought)}", f"Garantía: {span} desde la fecha de compra"]}[lang]
    state = "\n".join(lines + [f"{TODAY[lang]}: {pd(t)}"])
    q = ask(rng, lang, {
        "en": ["Is this product still under warranty today?", "Can I still make a warranty claim today?"],
        "nl": ["Valt dit product vandaag nog onder de garantie?"], "de": ["Ist das Produkt heute noch in der Garantie?"],
        "fr": ["Ce produit est-il encore sous garantie aujourd'hui ?"], "es": ["¿Este producto sigue en garantía hoy?"]})
    return state, noul(q), str(t <= expiry).lower(), grade_days(abs((expiry - t).days))


def passport_valid_after_trip(rng, lang):
    months = rng.choice([3, 6, 6])
    leave = today(rng) + dt.timedelta(days=rng.randint(10, 200))
    margin, ok = bucket(rng), side(rng)
    need = add_months(leave, months)
    gap = {"near": rng.randint(2, 4), "mid": rng.randint(5, 25), "far": rng.randint(40, 400)}[margin]
    expiry = need + dt.timedelta(days=gap if ok else -gap)
    arrive = leave - dt.timedelta(days=rng.randint(3, 21))
    country = rng.choice(["Thailand", "Indonesia", "Turkey", "Egypt", "Vietnam", "Morocco", "Kenya", "Brazil"])
    pd = page_dates(rng, lang)
    lines = {"en": [f"Entry requirements: {country}", f"Your passport must be valid for at least {months} months after the day you leave {country}.",
                    f"Trip: {pd(arrive)} – {pd(leave)}", f"Passport expiry date: {pd(expiry)}"],
             "nl": [f"Inreisvoorwaarden: {country}", f"Je paspoort moet nog minstens {months} maanden geldig zijn na de dag van vertrek uit {country}.",
                    f"Reis: {pd(arrive)} – {pd(leave)}", f"Paspoort geldig tot: {pd(expiry)}"],
             "de": [f"Einreisebestimmungen: {country}", f"Der Reisepass muss nach der Ausreise noch mindestens {months} Monate gültig sein.",
                    f"Reise: {pd(arrive)} – {pd(leave)}", f"Pass gültig bis: {pd(expiry)}"],
             "fr": [f"Conditions d'entrée : {country}", f"Le passeport doit être valable au moins {months} mois après la date de sortie du pays.",
                    f"Voyage : {pd(arrive)} – {pd(leave)}", f"Passeport valable jusqu'au : {pd(expiry)}"],
             "es": [f"Requisitos de entrada: {country}", f"El pasaporte debe tener una validez mínima de {months} meses tras la salida del país.",
                    f"Viaje: {pd(arrive)} – {pd(leave)}", f"Pasaporte válido hasta: {pd(expiry)}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Does my passport meet the validity rule for this trip?", "Is my passport valid long enough for this trip?"],
        "nl": ["Voldoet mijn paspoort aan de geldigheidseis voor deze reis?"], "de": ["Erfüllt mein Pass die Gültigkeitsregel für diese Reise?"],
        "fr": ["Mon passeport respecte-t-il la règle de validité pour ce voyage ?"], "es": ["¿Mi pasaporte cumple la norma de validez para este viaje?"]})
    return "\n".join(lines), noul(q), str(expiry >= need).lower(), margin


def tip_at_least_percent(rng, lang):
    cur = cur_for(rng, lang)
    bill = round(rng.uniform(12, 320), 2)
    p = rng.choice([10, 12, 15, 18, 20])
    margin, ok = bucket(rng), side(rng)
    tip = round(bill * p / 100 * (1 + rel(rng, margin) if ok else 1 - rel(rng, margin)), 2)
    if abs(tip - bill * p / 100) < 0.01:
        raise Skip
    word = {"en": ("Food & drinks", "Tip", "Total charged"), "nl": ("Eten & drinken", "Fooi", "Totaal afgeschreven"),
            "de": ("Speisen & Getränke", "Trinkgeld", "Belastet"), "fr": ("Repas & boissons", "Pourboire", "Total débité"),
            "es": ("Comida y bebida", "Propina", "Total cobrado")}[lang]
    lines = [rng.choice(["Trattoria Da Marco", "Café de Jaren", "Brasserie Lipp", "Taberna El Sur", "The Ivy Kitchen"]),
             f"{word[0]}: {money(bill, lang, cur)}", f"{word[1]}: {money(tip, lang, cur)}"]
    if rng.random() < 0.6:
        lines.append(f"{word[2]}: {money(bill + tip, lang, cur)}")
    q = ask(rng, lang, {
        "en": ["Is the tip at least {p}% of the food and drinks?", "Did we tip {p}% or more?"],
        "nl": ["Is de fooi minstens {p}% van het bedrag voor eten en drinken?"], "de": ["Beträgt das Trinkgeld mindestens {p} %?"],
        "fr": ["Le pourboire représente-t-il au moins {p} % ?"], "es": ["¿La propina es de al menos el {p} %?"]}, p=p)
    return "\n".join(lines), noul(q), str(tip / bill >= p / 100).lower(), margin


def price_increase_over_percent(rng, lang):
    cur = cur_for(rng, lang)
    old = rng.choice([4.99, 7.99, 9.99, 11.99, 12.99, 13.49, 15.99, 17.99, 22.5, 29.99, 34.95, 44.0, 59.0])
    p = rng.choice([5, 8, 10, 15, 20, 25])
    margin, over = bucket(rng), side(rng)
    new = round(old * (1 + p / 100 * (1 + rel(rng, margin) if over else 1 - rel(rng, margin))), 2)
    if new <= old or abs((new - old) / old - p / 100) < 0.001:
        raise Skip
    service = rng.choice(["Netflix", "Spotify Family", "Basic-Fit", "Disney+", "Ziggo Internet", "Vodafone Red", "The Guardian", "Adobe Photography"])
    start = today(rng) + dt.timedelta(days=rng.randint(14, 60))
    lines = {"en": [f"Subject: Changes to your {service} plan", f"From {date(start, lang, rng)} your monthly price changes from {money(old, lang, cur)} to {money(new, lang, cur)}.",
                    "You can cancel at any time before then."],
             "nl": [f"Onderwerp: Wijziging van je {service}-abonnement", f"Vanaf {date(start, lang, rng)} gaat je maandprijs van {money(old, lang)} naar {money(new, lang)}.",
                    "Je kunt tot die tijd kosteloos opzeggen."],
             "de": [f"Betreff: Änderung Ihres {service}-Tarifs", f"Ab dem {date(start, lang, rng)} ändert sich Ihr Monatspreis von {money(old, lang)} auf {money(new, lang)}.",
                    "Sie können bis dahin jederzeit kündigen."],
             "fr": [f"Objet : Évolution de votre abonnement {service}", f"À partir du {date(start, lang, rng)}, votre prix mensuel passe de {money(old, lang)} à {money(new, lang)}.",
                    "Vous pouvez résilier à tout moment d'ici là."],
             "es": [f"Asunto: Cambios en tu plan de {service}", f"A partir del {date(start, lang, rng)}, tu precio mensual pasa de {money(old, lang)} a {money(new, lang)}.",
                    "Puedes cancelar en cualquier momento antes."]}[lang]
    q = ask(rng, lang, {
        "en": ["Is the price going up by more than {p}%?", "Is this a price increase of more than {p} percent?"],
        "nl": ["Stijgt de prijs met meer dan {p}%?"], "de": ["Steigt der Preis um mehr als {p} %?"],
        "fr": ["Le prix augmente-t-il de plus de {p} % ?"], "es": ["¿El precio sube más de un {p} %?"]}, p=p)
    gap = abs((new - old) / old - p / 100) / (p / 100)
    return "\n".join(lines), noul(q), str((new - old) / old > p / 100).lower(), grade(gap)


VAT = {"en": [20, 5], "nl": [21, 9], "de": [19, 7], "fr": [20, 10, 5.5], "es": [21, 10]}


def vat_amount_correct(rng, lang):
    cur = "£" if lang == "en" else "€"
    net = round(rng.uniform(15, 2400), 2)
    rate = rng.choice(VAT[lang])
    vat = round(net * rate / 100, 2)
    correct = side(rng)
    margin = "far"
    if not correct:
        margin = bucket(rng)
        err = rel(rng, margin) * rng.choice([-1, 1])
        if rng.random() < 0.5:
            vat_shown, total = round(vat * (1 + err), 2), None
            if abs(vat_shown - vat) < 0.02:
                raise Skip
            total = round(net + vat_shown, 2)
        else:
            vat_shown = vat
            total = round((net + vat) * (1 + err / 4), 2)
            if abs(total - (net + vat)) < 0.02:
                raise Skip
            margin = grade(abs(err / 4), 0.01, 0.04)
    else:
        vat_shown, total = vat, round(net + vat, 2)
    label = {"en": ("Net amount", f"VAT {rate}%", "Total incl. VAT"), "nl": ("Bedrag excl. btw", f"Btw {num(rate, lang, 0 if rate == int(rate) else 1)}%", "Totaal incl. btw"),
             "de": ("Nettobetrag", f"MwSt. {num(rate, lang, 0 if rate == int(rate) else 1)} %", "Gesamtbetrag brutto"),
             "fr": ("Montant HT", f"TVA {num(rate, lang, 0 if rate == int(rate) else 1)} %", "Total TTC"),
             "es": ("Base imponible", f"IVA {num(rate, lang, 0 if rate == int(rate) else 1)} %", "Total con IVA")}[lang]
    lines = [rng.choice(["Invoice 2026-0412", "Factuur F-10233", "Rechnung RE-5521", "Facture FA-7781", "Factura A-3310"]),
             f"{label[0]}: {money(net, lang, cur)}", f"{label[1]}: {money(vat_shown, lang, cur)}", f"{label[2]}: {money(total, lang, cur)}"]
    q = ask(rng, lang, {
        "en": ["Are the VAT and the total on this invoice calculated correctly?", "Is the VAT arithmetic on this invoice right?"],
        "nl": ["Kloppen de btw en het totaal op deze factuur?"], "de": ["Sind MwSt. und Gesamtbetrag auf dieser Rechnung richtig berechnet?"],
        "fr": ["La TVA et le total de cette facture sont-ils corrects ?"], "es": ["¿El IVA y el total de esta factura están bien calculados?"]})
    ok = abs(vat_shown - net * rate / 100) <= 0.011 and abs(total - (net + vat_shown)) <= 0.011
    return "\n".join(lines), noul(q), str(ok).lower(), margin


def split_bill_share_within(rng, lang):
    cur = cur_for(rng, lang)
    n = rng.randint(2, 8)
    menu = [("Pizza margherita", 9, 15), ("Burger", 12, 19), ("Pad thai", 11, 17), ("Caesar salad", 9, 15), ("Risotto", 14, 22),
            ("Steak frites", 19, 32), ("Wine (bottle)", 24, 48), ("Beer", 3.5, 6.5), ("Lemonade", 3, 5.5), ("Tiramisu", 6, 9),
            ("Nachos", 8, 13), ("Sushi platter", 22, 38)]
    items = [(m[0], round(rng.uniform(m[1], m[2]), 2)) for m in (rng.choice(menu) for _ in range(rng.randint(n, n + 5)))]
    total = round(sum(a for _, a in items), 2)
    tip = rng.choice([0, 0, 5, 10, 15])
    share = total * (1 + tip / 100) / n
    margin, ok = bucket(rng), side(rng)
    cap = round(share * (1 + rel(rng, margin)) if ok else share * (1 - rel(rng, margin)))
    if cap <= 0 or (share <= cap) != ok:
        raise Skip
    lines = [f"{name}  {money(a, lang, cur)}" for name, a in items]
    if rng.random() < 0.5:
        lines.append(f"{W['total'][lang]}: {money(total, lang, cur)}")
    tip_txt = {"en": f" plus a {tip}% tip", "nl": f" plus {tip}% fooi", "de": f" plus {tip} % Trinkgeld", "fr": f" plus {tip} % de pourboire",
               "es": f" más un {tip} % de propina"} if tip else ""
    q = ask2(rng, lang, {
        "en": ["If we split this bill{t} evenly between {n} people, does each person pay {c} or less?"],
        "nl": ["Als we deze rekening{t} gelijk verdelen over {n} personen, betaalt dan ieder {c} of minder?"],
        "de": ["Wenn wir die Rechnung{t} gleichmäßig auf {n} Personen aufteilen, zahlt dann jeder höchstens {c}?"],
        "fr": ["Si on partage l'addition{t} à parts égales entre {n} personnes, chacun paie-t-il {c} ou moins ?"],
        "es": ["Si dividimos la cuenta{t} a partes iguales entre {n} personas, ¿cada uno paga {c} o menos?"]},
        t=tip_txt, n=n, c=money(cap, lang, cur, 0))
    return "\n".join(lines), noul(q), str(share <= cap).lower(), grade(abs(share - cap) / cap)


UNIT_GOODS = [("coffee beans", "g", 0.022), ("rice", "g", 0.003), ("olive oil", "ml", 0.011), ("washing liquid", "ml", 0.006),
              ("muesli", "g", 0.008), ("dog food", "g", 0.005), ("shampoo", "ml", 0.02), ("orange juice", "ml", 0.002)]
BRANDS = ["Lavazza", "Jumbo", "Albert Heijn", "Carrefour", "Lidl", "Edeka", "Mercadona", "Tesco", "Aldi", "Rewe", "Plus", "Dia"]


def best_value_per_unit(rng, lang):
    good, base, per = rng.choice(UNIT_GOODS)
    k = rng.choice([2, 3])
    brands = rng.sample(BRANDS, k)
    sizes = rng.sample([250, 400, 500, 750, 900, 1000, 1200, 1500, 2000, 2500, 3000], k)
    prices = [round(s * per * rng.uniform(0.7, 1.3), 2) for s in sizes]
    unit_price = [p / s for p, s in zip(prices, sizes)]
    order = sorted(range(k), key=lambda i: unit_price[i])
    gap = (unit_price[order[1]] - unit_price[order[0]]) / unit_price[order[0]]
    if gap < 0.01:
        raise Skip

    def show(s):
        if s >= 1000 and rng.random() < 0.7:
            return f"{num(s / 1000, lang, 2 if s % 100 else 1).rstrip('0').rstrip(',.')} {'kg' if base == 'g' else 'L'}"
        if base == "ml" and s % 10 == 0 and rng.random() < 0.3:
            return f"{s // 10} cl"
        return f"{s} {base}"
    lines = [f"{b} {good} {show(s)} — {money(p, lang)}" for b, s, p in zip(brands, sizes, prices)]
    unit = "kilo" if base == "g" else "litre"
    q = {"type": "choice", "instructions": ask(rng, lang, {
        "en": [f"Which is the best value per {unit}?", f"Which offer is cheapest per {unit}?"],
        "nl": [f"Welke is het voordeligst per {'kilo' if base == 'g' else 'liter'}?"], "de": [f"Welches Angebot ist pro {'Kilo' if base == 'g' else 'Liter'} am günstigsten?"],
        "fr": [f"Quelle offre est la moins chère au {'kilo' if base == 'g' else 'litre'} ?"], "es": [f"¿Qué oferta es más barata por {'kilo' if base == 'g' else 'litro'}?"]}),
        "criteria": {slug(b): f"{b} {show(s)}" for b, s in zip(brands, sizes)}}
    return "\n".join(lines), q, slug(brands[order[0]]), grade(gap, 0.03, 0.15)


def installments_cost_more(rng, lang):
    cur = cur_for(rng, lang)
    cash = rng.choice([299, 349, 499, 599, 749, 899, 999, 1199, 1499, 1999]) + rng.choice([0, 0.99, 0.95])
    n = rng.choice([3, 6, 10, 12, 18, 24])
    fee = rng.choice([0, 0, 0, 4.95, 9.95, 19.0])
    kind = rng.random()
    if kind < 0.15:  # truly 0% with no fee: equal, so not more
        monthly, fee, margin = round(cash / n, 2), 0, "near"
        if abs(monthly * n - cash) > 0.05:
            raise Skip
        more = False
    else:
        margin, more = bucket(rng), side(rng)
        target = cash * (1 + rel(rng, margin)) if more else cash * (1 - rel(rng, margin))
        monthly = round((target - fee) / n, 2)
        if monthly <= 0:
            raise Skip
        more = monthly * n + fee > cash + 0.05
    total = monthly * n + fee
    p = rng.choice(PRODUCTS)
    lines = {"en": [p, f"{money(cash, lang, cur)} or {n} × {money(monthly, lang, cur)}/month" + (f" (one-off fee {money(fee, lang, cur)})" if fee else "")],
             "nl": [p, f"{money(cash, lang)} of {n} × {money(monthly, lang)} per maand" + (f" (eenmalige kosten {money(fee, lang)})" if fee else "")],
             "de": [p, f"{money(cash, lang)} oder {n} × {money(monthly, lang)} monatlich" + (f" (einmalige Gebühr {money(fee, lang)})" if fee else "")],
             "fr": [p, f"{money(cash, lang)} ou {n} × {money(monthly, lang)}/mois" + (f" (frais uniques {money(fee, lang)})" if fee else "")],
             "es": [p, f"{money(cash, lang)} o {n} × {money(monthly, lang)}/mes" + (f" (comisión única {money(fee, lang)})" if fee else "")]}[lang]
    q = ask(rng, lang, {
        "en": ["Does paying in instalments cost more in total than paying at once?", "Is the instalment plan more expensive overall?"],
        "nl": ["Kost betalen in termijnen in totaal meer dan in één keer betalen?"], "de": ["Kostet die Ratenzahlung insgesamt mehr als die Einmalzahlung?"],
        "fr": ["Payer en plusieurs fois coûte-t-il plus cher au total ?"], "es": ["¿Pagar a plazos sale más caro en total?"]})
    return "\n".join(lines + [W["add"][lang]]), noul(q), str(more).lower(), margin if kind >= 0.15 else "near"


def hours_total_at_least(rng, lang):
    contract = rng.choice([24, 32, 36, 38, 40])
    days = DAYS[lang][:5] if contract >= 32 else rng.sample(DAYS[lang][:5], 3)
    margin, ok = rng.choices(["near", "mid", "far"], [0.35, 0.35, 0.3])[0], side(rng)
    gap = {"near": rng.choice([5, 10, 15]), "mid": rng.choice([30, 45, 60, 75, 90]), "far": rng.choice([180, 240, 300, 420])}[margin]
    total = contract * 60 + (gap if ok else -gap)
    if margin == "near" and ok and rng.random() < 0.3:
        total = contract * 60  # exactly enough
    k = len(days)
    parts = [total // k] * k
    parts[-1] += total - sum(parts)
    for _ in range(k):  # shuffle minutes between days in quarter hours
        i, j = rng.randrange(k), rng.randrange(k)
        d = rng.choice([0, 15, 30, 45])
        if parts[i] - d > 120:
            parts[i] -= d
            parts[j] += d

    def show(m):
        h, mm = divmod(m, 60)
        style = rng.random()
        if style < 0.35:
            return f"{h}:{mm:02d}"
        if style < 0.6 and mm % 15 == 0:
            return f"{num(m / 60, lang, 2).rstrip('0').rstrip(',.')} h"
        return f"{h} h {mm} min" if mm else f"{h} h"
    name = rng.choice(NAMES)
    lines = [{"en": f"Timesheet — {name}", "nl": f"Urenstaat — {name}", "de": f"Stundenzettel — {name}", "fr": f"Feuille de temps — {name}",
              "es": f"Parte de horas — {name}"}[lang]] + [f"{d}: {show(m)}" for d, m in zip(days, parts)]
    q = ask(rng, lang, {
        "en": ["Has {n} worked at least {c} hours this week?", "Do these entries add up to {c} hours or more?"],
        "nl": ["Heeft {n} deze week minstens {c} uur gewerkt?"], "de": ["Hat {n} diese Woche mindestens {c} Stunden gearbeitet?"],
        "fr": ["{n} a-t-il travaillé au moins {c} heures cette semaine ?"], "es": ["¿{n} ha trabajado al menos {c} horas esta semana?"]}, n=name, c=contract)
    return "\n".join(lines), noul(q), str(sum(parts) >= contract * 60).lower(), "near" if abs(sum(parts) - contract * 60) <= 15 else grade_minutes(abs(sum(parts) - contract * 60))


ROUTES = [("Amsterdam Centraal", "Wien Hbf"), ("Brussel-Zuid", "Milano Centrale"), ("Berlin Hbf", "Zürich HB"), ("Paris Bercy", "Venezia S. Lucia"),
          ("Madrid Estación Sur", "Lisboa Oriente"), ("München ZOB", "Roma Tiburtina"), ("Utrecht Centraal", "Innsbruck Hbf")]


def journey_within_hours(rng, lang):
    cs = clock_style(rng, lang)
    dep = rng.randint(17 * 4, 23 * 4 + 3) * 15 + rng.choice([0, 5, 10])
    hours = rng.choice([6, 8, 9, 10, 12, 14])
    margin, ok = rng.choices(["near", "mid", "far"], [0.3, 0.35, 0.35])[0], side(rng)
    gap = {"near": rng.randint(5, 10), "mid": rng.randint(20, 45), "far": rng.randint(90, 200)}[margin]
    length = hours * 60 + (-gap if ok else gap)
    arr = dep + length
    nxt = arr // (24 * 60)
    a, b = rng.choice(ROUTES)
    kind = rng.choice(["Nightjet", "FlixBus", "European Sleeper", "BlaBlaCar Bus"])
    plus = f" (+{nxt})" if nxt else ""
    lines = [f"{kind} · {a} → {b}", f"{clock(dep, cs)} {a}", f"{clock(arr % (24 * 60), cs)}{plus} {b}",
             {"en": "All times local (CET)", "nl": "Alle tijden lokaal (MET)", "de": "Alle Zeiten Ortszeit (MEZ)", "fr": "Heures locales (CET)", "es": "Horas locales (CET)"}[lang]]
    q = ask(rng, lang, {
        "en": ["Does this journey take {h} hours or less?", "Is the trip at most {h} hours long?"],
        "nl": ["Duurt deze reis {h} uur of korter?"], "de": ["Dauert die Fahrt höchstens {h} Stunden?"],
        "fr": ["Ce trajet dure-t-il {h} heures ou moins ?"], "es": ["¿El viaje dura {h} horas o menos?"]}, h=hours)
    return "\n".join(lines), noul(q), str(length <= hours * 60).lower(), grade_minutes(gap)


AIRPORTS = ["AMS", "FRA", "CDG", "MAD", "MUC", "ZRH", "CPH", "IST", "DOH", "LHR", "BCN", "VIE"]


def connection_long_enough(rng, lang):
    cs = clock_style(rng, lang)
    mct = rng.choice([40, 45, 50, 60, 75, 90])
    arr = rng.randint(6 * 4, 23 * 4 + 3) * 15 + rng.choice([0, 5, 10])
    margin, ok = rng.choices(["near", "mid", "far"], [0.35, 0.35, 0.3])[0], side(rng)
    if ok:
        layover = mct + {"near": rng.choice([0, 5, 5, 10]), "mid": rng.randint(15, 35), "far": rng.randint(60, 180)}[margin]
    else:
        layover = max(mct - {"near": rng.choice([5, 10]), "mid": rng.randint(15, 35), "far": rng.randint(36, 60)}[margin], rng.choice([15, 20, 25]))
        if layover >= mct:
            raise Skip
    dep = arr + layover
    hub = rng.choice(AIRPORTS)
    o, d = rng.sample([a for a in AIRPORTS if a != hub], 2)
    f1, f2 = f"{rng.choice(['KL', 'LH', 'AF', 'IB', 'LX', 'SK', 'TK'])}{rng.randint(100, 1999)}", f"{rng.choice(['KL', 'LH', 'AF', 'IB', 'LX', 'SK', 'TK'])}{rng.randint(100, 1999)}"
    lines = [f"{f1}  {o} → {hub}  arr. {clock(arr % 1440, cs)}", f"{f2}  {hub} → {d}  dep. {clock(dep % 1440, cs)}" + (" (+1)" if dep // 1440 > arr // 1440 else ""),
             {"en": f"Minimum connection time at {hub}: {mct} min", "nl": f"Minimale overstaptijd op {hub}: {mct} min", "de": f"Mindestumsteigezeit in {hub}: {mct} Min.",
              "fr": f"Temps de correspondance minimum à {hub} : {mct} min", "es": f"Tiempo mínimo de conexión en {hub}: {mct} min"}[lang]]
    q = ask(rng, lang, {
        "en": ["Does this connection meet the minimum connection time?", "Is there enough time to make the connecting flight?"],
        "nl": ["Is er genoeg tijd om de aansluitende vlucht te halen?"], "de": ["Reicht die Umsteigezeit für den Anschlussflug?"],
        "fr": ["Le temps de correspondance est-il suffisant ?"], "es": ["¿Hay tiempo suficiente para la conexión?"]})
    return "\n".join(lines), noul(q), str(layover >= mct).lower(), "near" if abs(layover - mct) <= 10 else grade_minutes(abs(layover - mct))


INGREDIENTS = [("flour", "g"), ("butter", "g"), ("rice", "g"), ("milk", "ml"), ("cream", "ml"), ("chickpeas", "g"), ("pasta", "g"), ("stock", "ml")]


def recipe_scaled_enough(rng, lang):
    ing, unit = rng.choice(INGREDIENTS)
    serves = rng.choice([2, 4, 4, 6, 8])
    per_recipe = rng.choice([100, 150, 200, 250, 300, 400, 500, 750])
    people = rng.choice([p for p in [2, 3, 4, 5, 6, 8, 10, 12] if p != serves])
    need = per_recipe * people / serves
    margin, ok = bucket(rng), side(rng)
    have = need * (1 + rel(rng, margin)) if ok else need * (1 - rel(rng, margin))
    have = round(have / 5) * 5
    if have <= 0 or (have >= need) != ok:
        raise Skip
    have_txt = f"{num(have / 1000, lang, 2).rstrip('0').rstrip(',.')} {'kg' if unit == 'g' else 'L'}" if have >= 1000 and rng.random() < 0.6 else f"{have} {unit}"
    lines = {"en": [f"Recipe (serves {serves})", f"- {per_recipe} {unit} {ing}", "- 2 eggs", "- a pinch of salt", f"In my pantry: {have_txt} {ing}"],
             "nl": [f"Recept (voor {serves} personen)", f"- {per_recipe} {unit} {ing}", "- 2 eieren", "- snufje zout", f"In mijn voorraadkast: {have_txt} {ing}"],
             "de": [f"Rezept (für {serves} Personen)", f"- {per_recipe} {unit} {ing}", "- 2 Eier", "- 1 Prise Salz", f"Im Vorrat: {have_txt} {ing}"],
             "fr": [f"Recette (pour {serves} personnes)", f"- {per_recipe} {unit} de {ing}", "- 2 œufs", "- une pincée de sel", f"Dans mon placard : {have_txt} de {ing}"],
             "es": [f"Receta (para {serves} personas)", f"- {per_recipe} {unit} de {ing}", "- 2 huevos", "- una pizca de sal", f"En mi despensa: {have_txt} de {ing}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Do I have enough {i} to make this for {p} people?"], "nl": ["Heb ik genoeg {i} om dit voor {p} personen te maken?"],
        "de": ["Habe ich genug {i}, um das für {p} Personen zu machen?"], "fr": ["Ai-je assez de {i} pour {p} personnes ?"],
        "es": ["¿Tengo suficiente {i} para {p} personas?"]}, i=ing, p=people)
    return "\n".join(lines), noul(q), str(have >= need).lower(), margin


def average_rating_at_least(rng, lang):
    k = rng.randint(4, 10)
    stars = [rng.choices([1, 2, 3, 4, 5], [0.08, 0.1, 0.17, 0.3, 0.35])[0] for _ in range(k)]
    avg = sum(stars) / k
    m = rng.choice([3.0, 3.5, 4.0, 4.2, 4.5])
    if abs(avg - m) < 1e-9 and rng.random() < 0.5:
        pass  # exactly equal counts as at least
    gap = abs(avg - m)
    style = rng.random()
    names = rng.sample(NAMES, k)
    lines = [f"{n}: {'★' * s}{'☆' * (5 - s)}" if style < 0.5 else f"{n} — {s}/5" for n, s in zip(names, stars)]
    q = ask(rng, lang, {
        "en": ["Is the average of these reviews at least {m}?", "Do these ratings average {m} stars or more?"],
        "nl": ["Is het gemiddelde van deze beoordelingen minstens {m}?"], "de": ["Liegt der Durchschnitt dieser Bewertungen bei mindestens {m}?"],
        "fr": ["La moyenne de ces avis est-elle d'au moins {m} ?"], "es": ["¿La media de estas opiniones es de al menos {m}?"]},
        m=num(m, "en" if lang == "en" else lang, 1))
    return "\n".join(lines), noul(q), str(avg >= m - 1e-9).lower(), "near" if gap <= 0.1 else ("mid" if gap <= 0.4 else "far")


def add_business_days(d: dt.date, n: int, count_today: bool) -> dt.date:
    if count_today and d.weekday() < 5:
        n -= 1
    while n > 0:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def arrives_by_business_days(rng, lang):
    t = today(rng)
    while t.weekday() >= 5:
        t += dt.timedelta(days=1)
    on_time = side(rng)
    while True:
        lo = rng.choice([1, 2, 3])
        hi = lo + rng.choice([0, 1, 2, 3])
        worst = [add_business_days(t, hi, c) for c in (False, True)]
        if on_time:
            deadline = max(worst) + dt.timedelta(days=rng.choice([0, 1, 2, 3, 5, 8]))
            break
        room = (min(worst) - t).days - 1  # days strictly between today and the earliest reading of the slowest arrival
        if room >= 1:
            deadline = min(worst) - dt.timedelta(days=rng.choice([d for d in (1, 2, 3, 5, 8) if d <= room]))
            break
    verdicts = {w <= deadline for w in worst}
    if len(verdicts) > 1:
        raise Skip
    truth = verdicts.pop()
    gap = min(abs((w - deadline).days) for w in worst)
    pd = page_dates(rng, lang)
    span = f"{lo}–{hi}" if hi > lo else f"{hi}"
    lines = {"en": [f"Ships today ({DAYS['en'][t.weekday()]} {pd(t)})", f"Delivery: {span} business days (Mon–Fri)"],
             "nl": [f"Vandaag verzonden ({DAYS['nl'][t.weekday()]} {pd(t)})", f"Levertijd: {span} werkdagen (ma–vr)"],
             "de": [f"Versand heute ({DAYS['de'][t.weekday()]}, {pd(t)})", f"Lieferzeit: {span} Werktage (Mo–Fr)"],
             "fr": [f"Expédié aujourd'hui ({DAYS['fr'][t.weekday()]} {pd(t)})", f"Livraison : {span} jours ouvrés (lun.–ven.)"],
             "es": [f"Se envía hoy ({DAYS['es'][t.weekday()]} {pd(t)})", f"Entrega: {span} días laborables (lun–vie)"]}[lang]
    q = ask2(rng, lang, {
        "en": ["Will it arrive by {d} even at the slowest stated delivery time?", "Is delivery guaranteed by {d}?"],
        "nl": ["Is het er uiterlijk {d}, ook bij de langste levertijd?"], "de": ["Kommt es auch bei der längsten Lieferzeit bis {d} an?"],
        "fr": ["Arrivera-t-il au plus tard le {d}, même avec le délai le plus long ?"], "es": ["¿Llegará el {d} como tarde, incluso con el plazo más largo?"]},
        d={ql: f"{DAYS[ql][deadline.weekday()]} {pd(deadline)}" for ql in ("en", lang)})
    return "\n".join(lines + [W["add"][lang]]), noul(q), str(truth).lower(), "near" if gap <= 1 else ("mid" if gap <= 4 else "far")


CATEGORIES = {"en": "groceries", "nl": "boodschappen", "de": "Lebensmittel", "fr": "courses", "es": "compra"}


def budget_left_allows(rng, lang):
    cur = cur_for(rng, lang)
    budget = rng.choice([150, 200, 250, 300, 400, 500, 600])
    k = rng.randint(3, 8)
    spent = [round(rng.uniform(5, budget / k * 1.3), 2) for _ in range(k)]
    left = budget - sum(spent)
    if left < 5:
        raise Skip
    margin, ok = bucket(rng), side(rng)
    x = round(left * (1 - rel(rng, margin)) if ok else left * (1 + rel(rng, margin)), 2)
    if (x <= left) != ok or x <= 0:
        raise Skip
    shops = ["Albert Heijn", "Jumbo", "Lidl", "Aldi", "Rewe", "Carrefour", "Mercadona", "Tesco", "Picnic", "Monoprix"]
    t, pd = today(rng), page_dates(rng, lang)
    lines = [{"en": f"Budget — {CATEGORIES['en']}: {money(budget, lang, cur, 0)} per month", "nl": f"Budget {CATEGORIES['nl']}: {money(budget, lang, cur, 0)} per maand",
              "de": f"Budget {CATEGORIES['de']}: {money(budget, lang, cur, 0)} pro Monat", "fr": f"Budget {CATEGORIES['fr']} : {money(budget, lang, cur, 0)} par mois",
              "es": f"Presupuesto de {CATEGORIES['es']}: {money(budget, lang, cur, 0)} al mes"}[lang]]
    for s in spent:
        lines.append(f"{pd(t - dt.timedelta(days=rng.randint(0, 25)))}  {rng.choice(shops)}  {money(-s, lang, cur)}")
    q = ask2(rng, lang, {
        "en": ["Can I spend another {x} on {c} this month without going over budget?"],
        "nl": ["Kan ik deze maand nog {x} aan {c} uitgeven zonder over mijn budget te gaan?"],
        "de": ["Kann ich diesen Monat noch {x} für {c} ausgeben, ohne das Budget zu überschreiten?"],
        "fr": ["Puis-je encore dépenser {x} en {c} ce mois-ci sans dépasser mon budget ?"],
        "es": ["¿Puedo gastar otros {x} en {c} este mes sin pasarme del presupuesto?"]}, x=money(x, lang, cur), c=CATEGORIES)
    return "\n".join(lines), noul(q), str(x <= left + 1e-9).lower(), grade(abs(left - x) / max(x, 1))


FX = [("GBP", "£", 0.86), ("CHF", "CHF", 0.94), ("USD", "US$", 1.08), ("SEK", "SEK", 11.4), ("PLN", "zł", 4.3), ("JPY", "¥", 162.0), ("DKK", "DKK", 7.46)]


def currency_either_direction_within_budget(rng, lang):
    code, sym, eur_rate = rng.choice(FX)
    eur_rate = round(eur_rate * rng.uniform(0.96, 1.04), 4 if eur_rate < 2 else 2)
    budget = rng.choice([40, 60, 80, 100, 150, 200, 300, 500])
    margin, ok = bucket(rng), side(rng)
    eur = budget * (1 - rel(rng, margin) if ok else 1 + rel(rng, margin))
    price = round(eur * eur_rate, 0 if code == "JPY" else 2)
    if rng.random() < 0.5:
        rate_txt = f"1 EUR = {num(eur_rate, lang, 4 if eur_rate < 2 else 2)} {code}"
    else:
        rate_txt = f"1 {code} = {num(1 / eur_rate, lang, 4)} EUR"
    true_eur = price / eur_rate
    p = rng.choice(PRODUCTS)
    label = {"en": "Price", "nl": "Prijs", "de": "Preis", "fr": "Prix", "es": "Precio"}[lang]
    rate_label = {"en": "Today's rate", "nl": "Koers vandaag", "de": "Kurs heute", "fr": "Taux du jour", "es": "Cambio de hoy"}[lang]
    lines = [p, f"{label}: {num(price, lang, 0 if code == 'JPY' else 2)} {code}" if rng.random() < 0.5 else f"{label}: {sym} {num(price, lang, 0 if code == 'JPY' else 2)}",
             f"{rate_label}: {rate_txt}"]
    q = ask(rng, lang, {
        "en": ["In euros, is this within my budget of €{b}?", "Converted to euros, does this cost €{b} or less?"],
        "nl": ["Past dit omgerekend binnen mijn budget van € {b}?"], "de": ["Liegt das umgerechnet bei höchstens {b} €?"],
        "fr": ["Converti en euros, est-ce que ça reste sous {b} € ?"], "es": ["En euros, ¿entra en mi presupuesto de {b} €?"]}, b=budget)
    return "\n".join(lines), noul(q), str(true_eur <= budget).lower(), grade(abs(true_eur - budget) / budget)


def reach_before_closing(rng, lang):
    cs = clock_style(rng, lang)
    close = rng.choice([17, 18, 18, 19, 20, 21, 22]) * 60 + rng.choice([0, 0, 30])
    mode = rng.choice([("walk", 5), ("bike", 15), ("walk", 4.5), ("bike", 18)])
    km = round(rng.uniform(0.6, 9 if mode[0] == "bike" else 4), 1)
    travel = km / mode[1] * 60
    margin, ok = rng.choices(["near", "mid", "far"], [0.3, 0.35, 0.35])[0], side(rng)
    slack = {"near": rng.randint(3, 6), "mid": rng.randint(10, 30), "far": rng.randint(45, 120)}[margin]
    now = int(round(close - travel - slack if ok else close - travel + slack))
    if now < 6 * 60:
        raise Skip
    shop = rng.choice(["Apotheek Centrum", "Post office", "DHL ServicePoint", "Bibliotheek", "Bike repair Spaak", "Mairie annexe"])
    how = {"walk": {"en": "walking", "nl": "lopend", "de": "zu Fuß", "fr": "à pied", "es": "a pie"},
           "bike": {"en": "by bike", "nl": "met de fiets", "de": "mit dem Rad", "fr": "à vélo", "es": "en bici"}}[mode[0]][lang]
    lines = {"en": [f"{shop} — closes at {clock(close, cs)}", f"Distance: {num(km, lang, 1)} km", f"My speed {how}: about {num(mode[1], lang, 1).rstrip('0').rstrip('.')} km/h", f"Now: {clock(now, cs)}"],
             "nl": [f"{shop} — sluit om {clock(close, cs)}", f"Afstand: {num(km, lang, 1)} km", f"Mijn tempo {how}: ongeveer {num(mode[1], lang, 1).rstrip('0').rstrip(',')} km/u", f"Nu: {clock(now, cs)}"],
             "de": [f"{shop} — schließt um {clock(close, cs)}", f"Entfernung: {num(km, lang, 1)} km", f"Mein Tempo {how}: etwa {num(mode[1], lang, 1).rstrip('0').rstrip(',')} km/h", f"Jetzt: {clock(now, cs)}"],
             "fr": [f"{shop} — ferme à {clock(close, cs)}", f"Distance : {num(km, lang, 1)} km", f"Ma vitesse {how} : environ {num(mode[1], lang, 1).rstrip('0').rstrip(',')} km/h", f"Maintenant : {clock(now, cs)}"],
             "es": [f"{shop} — cierra a las {clock(close, cs)}", f"Distancia: {num(km, lang, 1)} km", f"Mi ritmo {how}: unos {num(mode[1], lang, 1).rstrip('0').rstrip(',')} km/h", f"Ahora: {clock(now, cs)}"]}[lang]
    q = ask(rng, lang, {
        "en": ["If I leave now, will I get there before it closes?"], "nl": ["Als ik nu vertrek, ben ik er dan voor sluitingstijd?"],
        "de": ["Wenn ich jetzt losgehe, bin ich vor Ladenschluss da?"], "fr": ["Si je pars maintenant, arriverai-je avant la fermeture ?"],
        "es": ["Si salgo ahora, ¿llegaré antes de que cierre?"]})
    return "\n".join(lines), noul(q), str(now + travel <= close).lower(), "near" if slack <= 6 else grade_minutes(slack)


def storage_fits(rng, lang):
    free_gb = round(rng.uniform(0.8, 60), 1)
    k = rng.randint(2, 6)
    margin, ok = bucket(rng), side(rng)
    target = free_gb * (1 - rel(rng, margin) - 0.03 if ok else 1 + rel(rng, margin) + 0.03)
    weights = [rng.random() + 0.2 for _ in range(k)]
    sizes_mb = [target * 1000 * w / sum(weights) for w in weights]

    def show(mb):
        if mb >= 1000 and rng.random() < 0.7:
            return f"{num(mb / 1000, lang, 2)} GB"
        return f"{num(mb, lang, 0)} MB"
    total = sum(sizes_mb)
    verdicts = {total <= free_gb * 1000, total <= free_gb * 1024}
    if len(verdicts) > 1:
        raise Skip
    files = rng.sample(["Season 2 E01.mkv", "vacation_2025.zip", "Photos June", "podcast_backup.mp3", "Xcode_16.xip", "thesis_drafts", "Spotify cache",
                        "game_update.pkg", "wedding_video.mov", "scans_2024.pdf"], k)
    lines = [{"en": f"Storage: {num(free_gb, lang, 1)} GB available", "nl": f"Opslag: {num(free_gb, lang, 1)} GB beschikbaar", "de": f"Speicher: {num(free_gb, lang, 1)} GB verfügbar",
              "fr": f"Stockage : {num(free_gb, lang, 1)} Go disponibles", "es": f"Almacenamiento: {num(free_gb, lang, 1)} GB disponibles"}[lang],
             {"en": "Ready to download:", "nl": "Klaar om te downloaden:", "de": "Bereit zum Herunterladen:", "fr": "Prêt à télécharger :", "es": "Listo para descargar:"}[lang]]
    lines += [f"- {f}  {show(s).replace('GB', 'Go').replace('MB', 'Mo') if lang == 'fr' else show(s)}" for f, s in zip(files, sizes_mb)]
    q = ask(rng, lang, {
        "en": ["Will all of these fit in the free space?", "Is there enough storage to download everything listed?"],
        "nl": ["Past dit allemaal in de vrije ruimte?"], "de": ["Passt das alles in den freien Speicher?"],
        "fr": ["Tout cela tient-il dans l'espace libre ?"], "es": ["¿Cabe todo esto en el espacio libre?"]})
    return "\n".join(lines), noul(q), str(verdicts.pop()).lower(), grade(abs(total - free_gb * 1000) / (free_gb * 1000), 0.05, 0.19)


def cheapest_plan_for_usage(rng, lang):
    cur = cur_for(rng, lang)
    unit, use = rng.choice([("GB", rng.randint(2, 60)), ("kWh", rng.randint(120, 600)), ("min", rng.randint(50, 900))])
    plans = []
    for name in rng.sample(["Basic", "Smart", "Flex", "Plus", "Max", "Start", "Unlimited", "Go"], 3):
        fee = round(rng.uniform(3, 45), 2)
        incl = rng.choice([0, use // 3, use // 2, use, use * 2])
        per = round(rng.uniform(0.05, 0.6) if unit != "kWh" else rng.uniform(0.18, 0.38), 2)
        plans.append((name, fee, incl, per))
    cost = [fee + max(0, use - incl) * per for _, fee, incl, per in plans]
    order = sorted(range(3), key=lambda i: cost[i])
    gap = (cost[order[1]] - cost[order[0]]) / cost[order[0]]
    if gap < 0.01:
        raise Skip
    incl_w = {"en": "included", "nl": "inbegrepen", "de": "inklusive", "fr": "inclus", "es": "incluidos"}[lang]
    then = {"en": "then", "nl": "daarna", "de": "danach", "fr": "puis", "es": "después"}[lang]
    month = {"en": "/month", "nl": "/maand", "de": "/Monat", "fr": "/mois", "es": "/mes"}[lang]
    lines = [f"{n}: {money(fee, lang, cur)}{month}" + (f", {incl} {unit} {incl_w}" if incl else "") + f", {then} {money(per, lang, cur)}/{unit}"
             for n, fee, incl, per in plans]
    lines.append({"en": f"My usage: about {use} {unit} a month", "nl": f"Mijn verbruik: ongeveer {use} {unit} per maand", "de": f"Mein Verbrauch: etwa {use} {unit} im Monat",
                  "fr": f"Ma consommation : environ {use} {unit} par mois", "es": f"Mi consumo: unos {use} {unit} al mes"}[lang])
    q = {"type": "choice", "instructions": ask(rng, lang, {
        "en": ["Which plan is cheapest for my usage?"], "nl": ["Welk abonnement is het goedkoopst voor mijn verbruik?"],
        "de": ["Welcher Tarif ist für meinen Verbrauch am günstigsten?"], "fr": ["Quelle offre est la moins chère pour ma consommation ?"],
        "es": ["¿Qué tarifa es la más barata para mi consumo?"]}), "criteria": {p[0].lower(): f"the {p[0]} plan" for p in plans}}
    return "\n".join(lines), q, plans[order[0]][0].lower(), grade(gap, 0.03, 0.15)


def earliest_arrival(rng, lang):
    cs = clock_style(rng, lang)
    options, base = [], rng.randint(7 * 12, 19 * 12) * 5
    for label in rng.sample(["Train", "Bus", "Train + tram", "Car share", "Ferry", "Intercity", "Sprinter"], 3):
        dep = base + rng.randint(0, 24) * 5
        dur = rng.randint(4, 40) * 5
        options.append((label, dep, dur))
    arrivals = [d + u for _, d, u in options]
    order = sorted(range(3), key=lambda i: arrivals[i])
    gap = arrivals[order[1]] - arrivals[order[0]]
    if gap < 2:
        raise Skip
    words = {"en": ("dep.", "travel time"), "nl": ("vertrek", "reistijd"), "de": ("ab", "Fahrzeit"), "fr": ("départ", "durée"), "es": ("salida", "duración")}[lang]
    lines = [f"{lab}: {words[0]} {clock(d, cs)}, {words[1]} {duration(u, lang)}" for lab, d, u in options]
    keys = [slug(o[0]) for o in options]
    q = {"type": "choice", "instructions": ask(rng, lang, {
        "en": ["Which option gets me there first?"], "nl": ["Met welke optie ben ik er het eerst?"], "de": ["Mit welcher Option bin ich zuerst da?"],
        "fr": ["Quelle option me fait arriver en premier ?"], "es": ["¿Con qué opción llego antes?"]}),
        "criteria": {k: o[0] for k, o in zip(keys, options)}}
    return "\n".join(lines), q, keys[order[0]], "near" if gap <= 5 else ("mid" if gap <= 20 else "far")


# -------------------------------------------------------------------------------------------- held-out families

def yearly_plan_saves_percent(rng, lang):
    cur = cur_for(rng, lang)
    monthly = rng.choice([4.99, 7.99, 9.99, 11.99, 14.99, 19.99, 24.99])
    p = rng.choice([10, 15, 20, 25, 30])
    margin, ok = bucket(rng), side(rng)
    saving = p / 100 * (1 + rel(rng, margin) if ok else 1 - rel(rng, margin))
    yearly = round(monthly * 12 * (1 - saving), 2)
    true_saving = 1 - yearly / (monthly * 12)
    app = rng.choice(["Duolingo", "Headspace", "NordVPN", "Strava", "Canva", "1Password", "Blinkist"])
    lines = [app, f"{money(monthly, lang, cur)} {'/month' if lang == 'en' else {'nl': 'per maand', 'de': 'monatlich', 'fr': 'par mois', 'es': 'al mes'}[lang]}",
             f"{money(yearly, lang, cur)} {'/year' if lang == 'en' else {'nl': 'per jaar', 'de': 'jährlich', 'fr': 'par an', 'es': 'al año'}[lang]}"]
    q = ask(rng, lang, {
        "en": ["Does the yearly plan save at least {p}% compared with paying monthly for a year?"],
        "nl": ["Bespaart het jaarabonnement minstens {p}% ten opzichte van een jaar maandelijks betalen?"],
        "de": ["Spart das Jahresabo mindestens {p} % gegenüber zwölf Monatszahlungen?"],
        "fr": ["L'abonnement annuel fait-il économiser au moins {p} % par rapport à 12 mois ?"],
        "es": ["¿El plan anual ahorra al menos un {p} % frente a pagar 12 meses?"]}, p=p)
    return "\n".join(lines), noul(q), str(true_saving >= p / 100).lower(), margin


def dose_interval_ok(rng, lang):
    cs = clock_style(rng, lang)
    hours = rng.choice([4, 6, 8, 12])
    last = rng.randint(0, 24 * 4 - 1) * 15
    margin, ok = rng.choices(["near", "mid", "far"], [0.3, 0.35, 0.35])[0], side(rng)
    gap = {"near": rng.choice([5, 10, 15]), "mid": rng.choice([30, 45, 60]), "far": rng.choice([120, 180, 240])}[margin]
    now = last + hours * 60 + (gap if ok else -gap)
    if now <= last:
        raise Skip
    yday = {"en": "yesterday", "nl": "gisteren", "de": "gestern", "fr": "hier", "es": "ayer"}[lang]
    today_w = {"en": "today", "nl": "vandaag", "de": "heute", "fr": "aujourd'hui", "es": "hoy"}[lang]
    last_txt = f"{clock(last, cs)} {yday if now >= 1440 else today_w}"
    med = rng.choice(["Ibuprofen 400 mg", "Paracetamol 1000 mg", "Naproxen 250 mg", "Amoxicillin 500 mg"])
    lines = {"en": [med, f"Take one tablet, at least {hours} hours apart.", f"Last dose: {last_txt}", f"Now: {clock(now % 1440, cs)} {today_w}"],
             "nl": [med, f"Neem één tablet, met minstens {hours} uur ertussen.", f"Laatste dosis: {last_txt}", f"Nu: {clock(now % 1440, cs)} {today_w}"],
             "de": [med, f"Eine Tablette, mindestens {hours} Stunden Abstand.", f"Letzte Einnahme: {last_txt}", f"Jetzt: {clock(now % 1440, cs)} {today_w}"],
             "fr": [med, f"Un comprimé, en espaçant les prises d'au moins {hours} heures.", f"Dernière prise : {last_txt}", f"Maintenant : {clock(now % 1440, cs)} {today_w}"],
             "es": [med, f"Un comprimido, con al menos {hours} horas entre tomas.", f"Última toma: {last_txt}", f"Ahora: {clock(now % 1440, cs)} {today_w}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Has enough time passed to take the next dose now?"], "nl": ["Is er genoeg tijd verstreken om nu de volgende dosis te nemen?"],
        "de": ["Ist genug Zeit vergangen, um jetzt die nächste Tablette zu nehmen?"], "fr": ["Assez de temps s'est-il écoulé pour reprendre un comprimé ?"],
        "es": ["¿Ha pasado suficiente tiempo para tomar la siguiente dosis?"]})
    return "\n".join(lines), noul(q), str(now - last >= hours * 60).lower(), "near" if gap <= 15 else grade_minutes(gap)


def fuel_enough_for_trip(rng, lang):
    use = round(rng.uniform(4.5, 9.5), 1)
    dist = rng.randint(40, 700)
    need = dist * use / 100
    margin, ok = bucket(rng), side(rng)
    tank = round(need * (1 + rel(rng, margin)) if ok else need * (1 - rel(rng, margin)), 1)
    if tank <= 0 or (tank >= need) != ok:
        raise Skip
    lines = {"en": [f"Fuel left: {num(tank, lang, 1)} L", f"Average consumption: {num(use, lang, 1)} L/100 km", f"Route to destination: {dist} km"],
             "nl": [f"Brandstof over: {num(tank, lang, 1)} L", f"Gemiddeld verbruik: {num(use, lang, 1)} L/100 km", f"Route naar bestemming: {dist} km"],
             "de": [f"Restkraftstoff: {num(tank, lang, 1)} l", f"Durchschnittsverbrauch: {num(use, lang, 1)} l/100 km", f"Strecke zum Ziel: {dist} km"],
             "fr": [f"Carburant restant : {num(tank, lang, 1)} L", f"Consommation moyenne : {num(use, lang, 1)} L/100 km", f"Itinéraire : {dist} km"],
             "es": [f"Combustible restante: {num(tank, lang, 1)} L", f"Consumo medio: {num(use, lang, 1)} L/100 km", f"Ruta al destino: {dist} km"]}[lang]
    q = ask(rng, lang, {
        "en": ["Is there enough fuel to reach the destination without refuelling?"], "nl": ["Is er genoeg brandstof om zonder tanken de bestemming te halen?"],
        "de": ["Reicht der Kraftstoff bis zum Ziel ohne Tanken?"], "fr": ["Y a-t-il assez de carburant pour arriver sans faire le plein ?"],
        "es": ["¿Hay combustible suficiente para llegar sin repostar?"]})
    return "\n".join(lines), noul(q), str(tank >= need).lower(), margin


def parcel_weight_limit(rng, lang):
    limit = rng.choice([2, 5, 10, 20, 23, 31.5])
    item_g = rng.choice([120, 250, 340, 450, 600, 850, 1200, 1800])
    pack_g = rng.choice([150, 200, 300, 450, 600])
    n_fit = (limit * 1000 - pack_g) / item_g
    if n_fit < 1.5:
        raise Skip
    margin, ok = bucket(rng), side(rng)
    n = max(1, int(n_fit * (1 - rel(rng, margin))) if ok else int(n_fit * (1 + rel(rng, margin))) + 1)
    total = n * item_g + pack_g
    item = rng.choice(["books", "jars of honey", "candles", "board games", "bottles of olive oil", "ceramic mugs", "puzzles"])
    lines = {"en": [f"Parcel: max. {num(limit, lang, 1).rstrip('0').rstrip('.')} kg", f"Contents: {n} {item}, {item_g} g each", f"Box and padding: {pack_g} g"],
             "nl": [f"Pakket: max. {num(limit, lang, 1).rstrip('0').rstrip(',')} kg", f"Inhoud: {n} × {item}, {item_g} g per stuk", f"Doos en vulmateriaal: {pack_g} g"],
             "de": [f"Paket: max. {num(limit, lang, 1).rstrip('0').rstrip(',')} kg", f"Inhalt: {n} × {item}, je {item_g} g", f"Karton und Polster: {pack_g} g"],
             "fr": [f"Colis : {num(limit, lang, 1).rstrip('0').rstrip(',')} kg max.", f"Contenu : {n} × {item}, {item_g} g chacun", f"Carton et calage : {pack_g} g"],
             "es": [f"Paquete: máx. {num(limit, lang, 1).rstrip('0').rstrip(',')} kg", f"Contenido: {n} × {item}, {item_g} g cada uno", f"Caja y relleno: {pack_g} g"]}[lang]
    q = ask(rng, lang, {
        "en": ["Is the parcel within the weight limit?"], "nl": ["Blijft het pakket onder de gewichtslimiet?"], "de": ["Bleibt das Paket unter dem Höchstgewicht?"],
        "fr": ["Le colis respecte-t-il la limite de poids ?"], "es": ["¿El paquete está dentro del límite de peso?"]})
    return "\n".join(lines), noul(q), str(total <= limit * 1000).lower(), grade(abs(total - limit * 1000) / (limit * 1000))


def cheapest_with_shipping(rng, lang):
    cur = cur_for(rng, lang)
    base = rng.uniform(15, 250)
    sellers = []
    for name in rng.sample(["bol.com", "Coolblue", "Amazon.de", "MediaMarkt", "Fnac", "El Corte Inglés", "Otto", "Cdiscount", "Zalando"], 3):
        price = round(base * rng.uniform(0.85, 1.15), 2)
        ship = rng.choice([0, 2.99, 4.95, 5.99, 6.95])
        free_from = rng.choice([None, None, 20, 25, 50])
        sellers.append((name, price, ship, free_from))
    total = [p + (0 if f is not None and p >= f else s) for _, p, s, f in sellers]
    order = sorted(range(3), key=lambda i: total[i])
    gap = (total[order[1]] - total[order[0]]) / total[order[0]]
    if gap < 0.005:
        raise Skip
    sw = {"en": ("shipping", "free shipping from"), "nl": ("verzending", "gratis verzending vanaf"), "de": ("Versand", "versandkostenfrei ab"),
          "fr": ("livraison", "livraison offerte dès"), "es": ("envío", "envío gratis desde")}[lang]
    lines = [f"{n}: {money(p, lang, cur)} + {sw[0]} {money(s, lang, cur) if s else W['free'][lang]}" + (f" ({sw[1]} {money(f, lang, cur, 0)})" if f and s else "")
             for n, p, s, f in sellers]
    q = {"type": "choice", "instructions": ask(rng, lang, {
        "en": ["Which seller is cheapest once shipping is included?"], "nl": ["Welke verkoper is het goedkoopst inclusief verzendkosten?"],
        "de": ["Welcher Händler ist inklusive Versand am günstigsten?"], "fr": ["Quel vendeur est le moins cher livraison comprise ?"],
        "es": ["¿Qué vendedor es el más barato con el envío incluido?"]}), "criteria": {slug(s[0]): s[0] for s in sellers}}
    return "\n".join(lines), q, slug(sellers[order[0]][0]), grade(gap, 0.03, 0.15)


TRAINED = [warranty_still_valid, passport_valid_after_trip, tip_at_least_percent, price_increase_over_percent, vat_amount_correct,
           split_bill_share_within, best_value_per_unit, installments_cost_more, hours_total_at_least, journey_within_hours,
           connection_long_enough, recipe_scaled_enough, average_rating_at_least, arrives_by_business_days, budget_left_allows,
           currency_either_direction_within_budget, reach_before_closing, storage_fits, cheapest_plan_for_usage, earliest_arrival]
HELD_OUT = [yearly_plan_saves_percent, dose_interval_ok, fuel_enough_for_trip, parcel_weight_limit, cheapest_with_shipping]


def safe(fn):
    def draw(rng, lang):
        while True:
            try:
                return fn(rng, lang)
            except Skip:
                continue
    draw.__name__ = fn.__name__
    return draw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path)
    parser.add_argument("--per-family", type=int, default=600)
    parser.add_argument("--test-per-family", type=int, default=40)
    parser.add_argument("--held-out-per-family", type=int, default=120)
    args = parser.parse_args()
    # Each family draws its test rows and its training rows from streams of its own, so the test set stays the same
    # whatever the training size (and teachers' labels of it stay valid).
    train, test = [], []
    for fn in TRAINED:
        rows = bn.rows_for(safe(fn), args.test_per_family, random.Random(f"2027-test-{fn.__name__}"), "family-seen")
        texts = {r["request"]["state"] for r in rows}
        test += rows
        train += [r for r in bn.rows_for(safe(fn), args.per_family, random.Random(f"2027-train-{fn.__name__}"), "train") if r["request"]["state"] not in texts]
    for fn in HELD_OUT:
        test += bn.rows_for(safe(fn), args.held_out_per_family, random.Random(f"2027-test-{fn.__name__}"), "family-unseen")
    for r in train + test:
        r["meta"]["source"] = "numeric2"
    for path, rows in ((args.out, train), (args.test, test)):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        if args.truth_dir:
            (args.truth_dir / f"{path.stem}.truth.jsonl").write_text(
                "".join(json.dumps(bn.truth_row(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for name, rows in (("train", train), ("test", test)):
        c = Counter((r["meta"]["family"], r["meta"]["truth"]) for r in rows)
        m = Counter(r["meta"]["margin"] for r in rows)
        print(f"{name}: {len(rows)} rows, margins {dict(m)}")
        for fam in sorted({f for f, _ in c}):
            parts = {t: n for (f, t), n in c.items() if f == fam}
            print(f"  {fam:<42} {parts}")


if __name__ == "__main__":
    main()
