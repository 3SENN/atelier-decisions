# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Rule application: yes/no questions where a page states a rule with conditions and an exception, and also gives
the specifics the rule has to be applied to. Answers are computed by code.

The mini picks one of several options well on families it never saw (92%), but a yes/no question that asks it to
apply a rule to the details on the page ("may this person do X, given these rules and these details") is right only
68% of the time. Code-labelled numeric data taught it dates and times, so this suite does the same for rules: 12
trained families and four held out whole, each a different everyday situation (renewing a library loan, booking a
gym class, play-off eligibility, a neighbourhood grant, noisy work in a flat, bulky waste, a meeting room, a
volunteer shift, lab access, licence seats, a classified ad, a drone flight; held out: giving blood, a photo
contest, a campsite fire, a tenant's pet).

Every family has two or three ordinary conditions (a limit, a count, a date, a time window, membership of a stated
list), one clause that waives a condition ("this minimum does not apply to goalkeepers") and one that blocks
whatever the rest says ("suspended players are not eligible"). A quarter of the rows are decided by one of those two
clauses (meta.margin "exception": half rescued, half blocked); the rest are decided by an ordinary condition that is
near, mid or far from its boundary. A false row usually fails exactly one condition, sometimes several.

Boundaries are stated on the page ("up to and including", "on or before", "at 00:00") or the row is kept clear of
them; a row whose computed answer differs from the planned one is dropped and drawn again.

    python data/build_conditions.py --out data/pool/pool-conditions.jsonl --test data/pool/conditions-test.jsonl \
        --truth-dir data/teacher
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from collections import Counter
from pathlib import Path

import build_numeric as bn
from build_numeric import DAYS, NAMES, ask, clock, clock_style, money, noul, num, rel, side, today
from build_numeric2 import TODAY, Skip, add_months, grade, page_dates, safe

YES = {"en": "yes", "nl": "ja", "de": "ja", "fr": "oui", "es": "sí"}
NO = {"en": "no", "nl": "nee", "de": "nein", "fr": "non", "es": "no"}
NOW = bn.W["now"]

# Distances from a boundary per margin, in whole units of `step`: (step, near, mid, far).
SCALES = {"days": (1, (0, 2), (3, 10), (12, 60)), "min": (5, (0, 2), (3, 9), (12, 36)), "quarter": (15, (0, 1), (2, 3), (4, 8)),
          "count": (1, (0, 1), (2, 2), (3, 6))}


def dist(rng, level: str, ok: bool, scale: str, least: int = 0) -> int:
    """A distance from the boundary: zero (exactly on it) only on the passing side, and never under `least` steps."""
    step, *spans = SCALES[scale]
    lo, hi = spans[("near", "mid", "far").index(level)]
    return max(rng.randint(lo, hi), least, 0 if ok else 1) * step


def grade_by(gap: float, scale: str) -> str:
    step, near, mid, _ = SCALES[scale]
    return "near" if gap <= near[1] * step else ("mid" if gap <= mid[1] * step else "far")


def amt(rng, limit: float, below: bool, level: str, q: float = 1.0) -> float:
    """An amount on the stated side of a limit, at a relative distance for the margin, in steps of q."""
    r = rel(rng, level)
    return round(round(limit * (1 - r if below else 1 + r) / q) * q, 2)


def rel_grade(value: float, limit: float) -> str:
    return grade(abs(value - limit) / limit)


def under_cap(rng, cap: int, ok: bool, level: str):
    """How many of `cap` are in use, when one more is wanted: full fails, otherwise 1, 2 or more are free."""
    if not ok:
        return cap, "near"
    used = max(0, cap - ({"near": 1, "mid": 2}.get(level) or rng.randint(3, 6)))
    return used, "near" if cap - used <= 1 else ("mid" if cap - used == 2 else "far")


class Plan:
    """What decides a row. `n` ordinary conditions; `hard` ones are yes/no (no distance); `waivable` is the one the
    waiver clause lifts. kind: plain (decided by condition `focus` at margin lv[focus]), rescue (the waivable
    condition fails and the waiver applies: true) or block (all pass, the blocking clause applies: false)."""

    def __init__(self, rng, n: int, hard=(), waivable: int = 0):
        self.fail, self.waiver, self.block, self.kind, self.focus, self.waivable = [False] * n, False, False, "plain", waivable, waivable
        r = rng.random()
        level = rng.choice(["near", "mid", "far"])
        if r < 0.125:
            self.kind, self.ok, self.waiver = "rescue", True, True
            self.fail[waivable] = True
        elif r < 0.25:
            self.kind, self.ok, self.block = "block", False, True
            self.waiver = rng.random() < 0.3
        else:
            self.ok = side(rng)
            self.focus = rng.choice([i for i in range(n) if i not in hard or (level == "far" and not self.ok)])
            if not self.ok:
                self.fail[self.focus] = True
                if rng.random() < 0.3:  # several conditions fail
                    for j in range(n):
                        if j != self.focus:
                            self.fail[j] = rng.random() < 0.6
            # A waiver that is present but beside the point (so "waiver on the page" does not mean yes).
            self.waiver = rng.random() < 0.3 and not (not self.ok and self.focus == waivable)
        self.lv = [level if i == self.focus else rng.choice(["mid", "far"]) for i in range(n)]
        self.passes = [not f for f in self.fail]


def verdict(P: Plan, passes: list[bool], grades: list[str]):
    """The answer from the values as generated; a row that drifted from its plan is dropped."""
    truth = all(p or (P.waiver and i == P.waivable) for i, p in enumerate(passes)) and not P.block
    if truth != P.ok or list(passes) != P.passes:
        raise Skip
    return str(truth).lower(), "exception" if P.kind != "plain" else grades[P.focus]


DISTRACT = {
    "en": ["Questions? Call 020 555 0142 or email info@example.org", "Front desk phone line: weekdays 9:00–12:30", "[Log out]",
           "Newsletter: 1 email per month, unsubscribe any time", "Lost property is kept at reception for 30 days.", "Page last updated 3 weeks ago",
           "Follow us on Instagram for news and photos", "We use cookies to keep you signed in. [OK]", "Annual contribution: €45.00, due in January"],
    "nl": ["Vragen? Bel 020 555 0142 of mail naar info@example.org", "Telefonisch bereikbaar: werkdagen 9.00–12.30 uur", "[Uitloggen]",
           "Nieuwsbrief: 1 mail per maand, afmelden kan altijd", "Gevonden voorwerpen bewaren we 30 dagen bij de receptie.", "Pagina 3 weken geleden bijgewerkt",
           "Volg ons op Instagram voor nieuws en foto's", "We gebruiken cookies om je ingelogd te houden. [OK]", "Jaarbijdrage: € 45,00, te betalen in januari"],
    "de": ["Fragen? Telefon 030 555 0142 oder info@example.org", "Telefonisch erreichbar: werktags 9:00–12:30 Uhr", "[Abmelden]",
           "Newsletter: 1 E-Mail pro Monat, jederzeit abbestellbar", "Fundsachen werden 30 Tage am Empfang aufbewahrt.", "Seite vor 3 Wochen aktualisiert",
           "Folge uns auf Instagram für Neuigkeiten und Fotos", "Wir verwenden Cookies, damit du angemeldet bleibst. [OK]", "Jahresbeitrag: 45,00 €, fällig im Januar"],
    "fr": ["Une question ? Appelez le 01 55 55 01 42 ou écrivez à info@example.org", "Accueil téléphonique : en semaine de 9h00 à 12h30", "[Se déconnecter]",
           "Lettre d'information : 1 e-mail par mois, désinscription à tout moment", "Les objets trouvés sont conservés 30 jours à l'accueil.",
           "Page mise à jour il y a 3 semaines", "Suivez-nous sur Instagram", "Nous utilisons des cookies pour vous garder connecté. [OK]",
           "Cotisation annuelle : 45,00 €, à régler en janvier"],
    "es": ["¿Dudas? Llama al 910 555 142 o escribe a info@example.org", "Atención telefónica: días laborables de 9:00 a 12:30", "[Cerrar sesión]",
           "Boletín: 1 correo al mes, baja en cualquier momento", "Los objetos perdidos se guardan 30 días en recepción.", "Página actualizada hace 3 semanas",
           "Síguenos en Instagram", "Usamos cookies para mantener tu sesión iniciada. [OK]", "Cuota anual: 45,00 €, se paga en enero"],
}


def page(rng, lang: str, rules: list[str], details: list[str], extra=()) -> str:
    """A page from a rules block and a details block (each: a heading, then lines in any order), in either order,
    with one to three lines that have nothing to do with the question."""
    bullet = rng.choice(["- ", "• ", "", ""])
    r_body, d_body = rules[1:], details[1:]
    rng.shuffle(r_body)
    rng.shuffle(d_body)
    blocks = [[rules[0]] + [bullet + line for line in r_body], [details[0]] + d_body]
    if rng.random() < 0.4:
        blocks.reverse()
    noise = rng.sample(DISTRACT[lang] + list(extra), rng.randint(1, 3))
    head = [noise.pop()] if len(noise) > 1 and rng.random() < 0.5 else []
    return "\n".join(head + blocks[0] + blocks[1] + noise)


def weekday(d: dt.date, lang: str) -> str:
    return DAYS[lang][d.weekday()]


# --------------------------------------------------------------------------------------------- trained families

BOOKS = ["The Overstory", "De avond is ongemak", "Der Zauberberg", "L'Étranger", "Cien años de soledad", "Sapiens", "Normal People",
         "Het diner", "Die Vermessung der Welt", "Patria", "Klara and the Sun", "La Peste"]


def library_renewal_allowed(rng, lang):
    P = Plan(rng, 2, waivable=1)
    t, pd = today(rng), page_dates(rng, lang)
    maxr = rng.choice([2, 3, 4, 5])
    used, g0 = under_cap(rng, maxr, P.passes[0], P.lv[0])
    d = dist(rng, P.lv[1], P.passes[1], "days")
    due = t + dt.timedelta(days=d if P.passes[1] else -d)
    book, yn = rng.choice(BOOKS), (YES[lang] if P.block else NO[lang])
    rules = {
        "en": [rng.choice(["Renewing your loans", "Renewals"]), f"Each item can be renewed up to {maxr} times.",
               "You can renew up to and including the due date. With a Plus card you can also renew after the due date.",
               "Items reserved by another member cannot be renewed."],
        "nl": [rng.choice(["Verlengen", "Leningen verlengen"]), f"Je kunt elk item maximaal {maxr} keer verlengen.",
               "Verlengen kan tot en met de inleverdatum. Met een Pluspas kan het ook na de inleverdatum.",
               "Items die door een ander lid zijn gereserveerd, kunnen niet worden verlengd."],
        "de": [rng.choice(["Ausleihen verlängern", "Verlängerung"]), f"Jedes Medium kann höchstens {maxr}-mal verlängert werden.",
               "Eine Verlängerung ist bis einschließlich des Rückgabedatums möglich. Mit der Plus-Karte geht das auch nach dem Rückgabedatum.",
               "Von einem anderen Mitglied vorgemerkte Medien können nicht verlängert werden."],
        "fr": [rng.choice(["Prolonger vos prêts", "Prolongations"]), f"Chaque document peut être prolongé {maxr} fois au maximum.",
               "La prolongation est possible jusqu'à la date de retour incluse. Avec la carte Plus, elle reste possible après cette date.",
               "Les documents réservés par un autre adhérent ne peuvent pas être prolongés."],
        "es": [rng.choice(["Renovar préstamos", "Renovaciones"]), f"Cada documento se puede renovar un máximo de {maxr} veces.",
               "Se puede renovar hasta la fecha de devolución inclusive. Con el carné Plus también se puede después de esa fecha.",
               "Los documentos reservados por otro socio no se pueden renovar."]}[lang]
    details = {
        "en": [f"My loans: {book}", f"Due date: {pd(due)}", f"Times renewed: {used}", f"Reserved by another member: {yn}",
               f"Card: {'Plus' if P.waiver else 'Basic'}", f"{TODAY[lang]}: {pd(t)}"],
        "nl": [f"Mijn leningen: {book}", f"Inleverdatum: {pd(due)}", f"Aantal keer verlengd: {used}", f"Gereserveerd door een ander lid: {yn}",
               f"Pas: {'Plus' if P.waiver else 'Basis'}", f"{TODAY[lang]}: {pd(t)}"],
        "de": [f"Meine Ausleihen: {book}", f"Rückgabedatum: {pd(due)}", f"Bereits verlängert: {used}-mal", f"Von einem anderen Mitglied vorgemerkt: {yn}",
               f"Karte: {'Plus' if P.waiver else 'Basis'}", f"{TODAY[lang]}: {pd(t)}"],
        "fr": [f"Mes prêts : {book}", f"Date de retour : {pd(due)}", f"Prolongations déjà effectuées : {used}", f"Réservé par un autre adhérent : {yn}",
               f"Carte : {'Plus' if P.waiver else 'Classique'}", f"{TODAY[lang]} : {pd(t)}"],
        "es": [f"Mis préstamos: {book}", f"Fecha de devolución: {pd(due)}", f"Veces renovado: {used}", f"Reservado por otro socio: {yn}",
               f"Carné: {'Plus' if P.waiver else 'Básico'}", f"{TODAY[lang]}: {pd(t)}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Can I renew this book today?", "Am I allowed to renew this loan today?", "Will the library let me renew this item today?"],
        "nl": ["Kan ik dit boek vandaag verlengen?", "Mag ik deze lening vandaag verlengen?"],
        "de": ["Kann ich dieses Buch heute verlängern?", "Darf ich diese Ausleihe heute verlängern?"],
        "fr": ["Puis-je prolonger ce livre aujourd'hui ?", "Ai-je le droit de prolonger ce prêt aujourd'hui ?"],
        "es": ["¿Puedo renovar este libro hoy?", "¿Se me permite renovar este préstamo hoy?"]})
    truth, margin = verdict(P, [used < maxr, t <= due], [g0, grade_by(d, "days")])
    return page(rng, lang, rules, details), noul(q), truth, margin


GYM_CLASSES = ["Spinning", "Yoga flow", "Bodypump", "Pilates", "HIIT 45", "Boxing basics", "Zumba", "Core & stretch"]


def gym_class_booking_allowed(rng, lang):
    P = Plan(rng, 2, waivable=0)
    t, pd = today(rng), page_dates(rng, lang)
    n_days, maxb = rng.choice([3, 5, 7]), rng.choice([2, 3, 4])
    if P.passes[0]:
        ahead = n_days - rng.choice([0, 1]) if P.lv[0] == "near" else rng.randint(1, max(1, n_days - 2))
    else:
        ahead = n_days + dist(rng, P.lv[0], False, "days")
    day = t + dt.timedelta(days=ahead)
    used, g1 = under_cap(rng, maxb, P.passes[1], P.lv[1])
    cls, cs = rng.choice(GYM_CLASSES), clock_style(rng, lang)
    at = clock(rng.choice([7, 9, 12, 17, 18, 19, 20]) * 60 + rng.choice([0, 15, 30]), cs)
    fee = money(rng.choice([5, 7.5, 10]), lang)
    tier = "Premium" if P.waiver else rng.choice(["Basic", "Flex"])
    rules = {
        "en": [rng.choice(["Booking rules", "How booking works"]),
               f"Booking opens {n_days} days before the date of the class, at 00:00. Premium members can book as soon as a class is on the schedule.",
               f"You can hold at most {maxb} upcoming bookings at a time.", "Members with an unpaid no-show fee cannot book."],
        "nl": [rng.choice(["Reserveringsregels", "Zo werkt reserveren"]),
               f"Reserveren kan vanaf {n_days} dagen vóór de datum van de les, vanaf 00.00 uur. Premiumleden kunnen reserveren zodra een les in het rooster staat.",
               f"Je kunt maximaal {maxb} openstaande reserveringen tegelijk hebben.", "Leden met een onbetaalde no-showboete kunnen niet reserveren."],
        "de": [rng.choice(["Buchungsregeln", "So funktioniert die Buchung"]),
               f"Kurse sind ab {n_days} Tagen vor dem Kurstag buchbar, ab 00:00 Uhr. Premium-Mitglieder können buchen, sobald ein Kurs im Plan steht.",
               f"Du kannst höchstens {maxb} offene Buchungen gleichzeitig haben.", "Mitglieder mit einer unbezahlten No-Show-Gebühr können nicht buchen."],
        "fr": [rng.choice(["Règles de réservation", "Comment réserver"]),
               f"La réservation ouvre {n_days} jours avant la date du cours, à 00h00. Les membres Premium peuvent réserver dès qu'un cours figure au planning.",
               f"Vous pouvez avoir au maximum {maxb} réservations à venir en même temps.", "Les membres ayant des frais d'absence impayés ne peuvent pas réserver."],
        "es": [rng.choice(["Normas de reserva", "Cómo reservar"]),
               f"La reserva se abre {n_days} días antes de la fecha de la clase, a las 00:00. Los socios Premium pueden reservar en cuanto la clase aparece en el horario.",
               f"Puedes tener como máximo {maxb} reservas pendientes a la vez.", "Los socios con una penalización por ausencia sin pagar no pueden reservar."]}[lang]
    none = {"en": "none", "nl": "geen", "de": "keine", "fr": "aucun", "es": "ninguna"}[lang]
    details = {
        "en": [f"Class: {cls}, {weekday(day, lang)} {pd(day)}, {at}", f"My upcoming bookings: {used}", f"Membership: {tier}",
               f"Unpaid no-show fee: {fee if P.block else none}", f"{TODAY[lang]}: {pd(t)}"],
        "nl": [f"Les: {cls}, {weekday(day, lang)} {pd(day)}, {at}", f"Mijn openstaande reserveringen: {used}", f"Lidmaatschap: {tier}",
               f"Onbetaalde no-showboete: {fee if P.block else none}", f"{TODAY[lang]}: {pd(t)}"],
        "de": [f"Kurs: {cls}, {weekday(day, lang)}, {pd(day)}, {at}", f"Meine offenen Buchungen: {used}", f"Mitgliedschaft: {tier}",
               f"Unbezahlte No-Show-Gebühr: {fee if P.block else none}", f"{TODAY[lang]}: {pd(t)}"],
        "fr": [f"Cours : {cls}, {weekday(day, lang)} {pd(day)}, {at}", f"Mes réservations à venir : {used}", f"Abonnement : {tier}",
               f"Frais d'absence impayés : {fee if P.block else none}", f"{TODAY[lang]} : {pd(t)}"],
        "es": [f"Clase: {cls}, {weekday(day, lang)} {pd(day)}, {at}", f"Mis reservas pendientes: {used}", f"Abono: {tier}",
               f"Penalización por ausencia sin pagar: {fee if P.block else none}", f"{TODAY[lang]}: {pd(t)}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Can I book this class today?", "Am I able to book this class today under the booking rules?", "Do the rules let me book this class today?"],
        "nl": ["Kan ik deze les vandaag reserveren?", "Mag ik deze les vandaag al boeken volgens de regels?"],
        "de": ["Kann ich diesen Kurs heute buchen?", "Darf ich diesen Kurs heute nach den Buchungsregeln buchen?"],
        "fr": ["Puis-je réserver ce cours aujourd'hui ?", "Les règles me permettent-elles de réserver ce cours aujourd'hui ?"],
        "es": ["¿Puedo reservar esta clase hoy?", "¿Las normas me permiten reservar esta clase hoy?"]})
    truth, margin = verdict(P, [ahead <= n_days, used < maxb], [grade_by(abs(ahead - n_days), "days"), g1])
    extra = {"en": ["Towel hire: €1.50", "Day pass for guests: €12.00"], "nl": ["Handdoek huren: € 1,50", "Dagpas voor gasten: € 12,00"],
             "de": ["Handtuchverleih: 1,50 €", "Tageskarte für Gäste: 12,00 €"], "fr": ["Location de serviette : 1,50 €", "Pass journée invité : 12,00 €"],
             "es": ["Alquiler de toalla: 1,50 €", "Pase de día para invitados: 12,00 €"]}[lang]
    return page(rng, lang, rules, details, extra), noul(q), truth, margin


POSITIONS = [dict(en="defender", nl="verdediger", de="Verteidiger", fr="défenseur", es="defensa"),
             dict(en="midfielder", nl="middenvelder", de="Mittelfeldspieler", fr="milieu de terrain", es="centrocampista"),
             dict(en="forward", nl="aanvaller", de="Stürmer", fr="attaquant", es="delantero")]
KEEPER = dict(en="goalkeeper", nl="keeper", de="Torhüter", fr="gardien de but", es="portero")


def league_playoff_eligible(rng, lang):
    P = Plan(rng, 2, waivable=0)
    pd, name = page_dates(rng, lang), rng.choice(NAMES)
    minm = rng.choice([4, 5, 6, 8, 10])
    g0 = dist(rng, P.lv[0], P.passes[0], "count")
    played = minm + g0 if P.passes[0] else max(0, minm - g0)
    deadline = today(rng)
    g1 = dist(rng, P.lv[1], P.passes[1], "days")
    reg = deadline + dt.timedelta(days=-g1 if P.passes[1] else g1)
    pos = (KEEPER if P.waiver else rng.choice(POSITIONS))[lang]
    k = rng.randint(1, 3)
    sus = ({"en": f"{k} more match{'es' if k > 1 else ''} to serve", "nl": f"nog {k} wedstrijd{'en' if k > 1 else ''}", "de": f"noch {k} Spiel{'e' if k > 1 else ''}",
            "fr": f"encore {k} match{'s' if k > 1 else ''}", "es": f"{k} partido{'s' if k > 1 else ''} por cumplir"} if P.block else
           {"en": "none", "nl": "geen", "de": "keine", "fr": "aucune", "es": "ninguna"})[lang]
    rules = {
        "en": [rng.choice(["Play-off eligibility", "Who may play in the play-offs"]),
               f"A player must have played at least {minm} regular-season matches. This minimum does not apply to goalkeepers.",
               f"The player must have been registered with the league on or before {pd(deadline)}.", "Players serving a suspension are not eligible."],
        "nl": [rng.choice(["Speelgerechtigdheid play-offs", "Wie mag in de play-offs spelen"]),
               f"Een speler moet minstens {minm} competitiewedstrijden hebben gespeeld. Dit minimum geldt niet voor keepers.",
               f"De speler moet uiterlijk op {pd(deadline)} bij de bond zijn ingeschreven.", "Geschorste spelers zijn niet speelgerechtigd."],
        "de": [rng.choice(["Spielberechtigung Play-offs", "Wer in den Play-offs spielen darf"]),
               f"Ein Spieler muss mindestens {minm} Ligaspiele bestritten haben. Für Torhüter gilt dieses Minimum nicht.",
               f"Der Spieler muss spätestens am {pd(deadline)} bei der Liga gemeldet worden sein.", "Gesperrte Spieler sind nicht spielberechtigt."],
        "fr": [rng.choice(["Qualification pour les play-offs", "Qui peut jouer les play-offs"]),
               f"Un joueur doit avoir disputé au moins {minm} matchs de saison régulière. Ce minimum ne s'applique pas aux gardiens de but.",
               f"Le joueur doit avoir été inscrit auprès de la ligue au plus tard le {pd(deadline)}.", "Les joueurs suspendus ne sont pas qualifiés."],
        "es": [rng.choice(["Elegibilidad para los play-offs", "Quién puede jugar los play-offs"]),
               f"Un jugador debe haber disputado al menos {minm} partidos de la fase regular. Este mínimo no se aplica a los porteros.",
               f"El jugador debe haber sido inscrito en la liga el {pd(deadline)} como muy tarde.", "Los jugadores sancionados no son elegibles."]}[lang]
    details = {
        "en": [f"Player: {name}", f"Position: {pos}", f"Regular-season matches played: {played}", f"Registered with the league: {pd(reg)}", f"Suspension: {sus}"],
        "nl": [f"Speler: {name}", f"Positie: {pos}", f"Gespeelde competitiewedstrijden: {played}", f"Ingeschreven bij de bond: {pd(reg)}", f"Schorsing: {sus}"],
        "de": [f"Spieler: {name}", f"Position: {pos}", f"Bestrittene Ligaspiele: {played}", f"Bei der Liga gemeldet: {pd(reg)}", f"Sperre: {sus}"],
        "fr": [f"Joueur : {name}", f"Poste : {pos}", f"Matchs de saison régulière disputés : {played}", f"Inscription auprès de la ligue : {pd(reg)}", f"Suspension : {sus}"],
        "es": [f"Jugador: {name}", f"Posición: {pos}", f"Partidos de fase regular disputados: {played}", f"Inscripción en la liga: {pd(reg)}", f"Sanción: {sus}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Is {n} eligible to play in the play-offs?", "May {n} play in the play-off matches?", "Do the rules allow {n} to play in the play-offs?"],
        "nl": ["Mag {n} in de play-offs spelen?", "Is {n} speelgerechtigd voor de play-offs?"],
        "de": ["Ist {n} für die Play-offs spielberechtigt?", "Darf {n} in den Play-offs spielen?"],
        "fr": ["Est-ce que {n} peut jouer les play-offs ?", "Le règlement autorise-t-il {n} à jouer les play-offs ?"],
        "es": ["¿Puede {n} jugar los play-offs?", "¿{n} cumple los requisitos para jugar los play-offs?"]}, n=name)
    truth, margin = verdict(P, [played >= minm, reg <= deadline], [grade_by(abs(played - minm), "count"), grade_by(g1, "days")])
    extra = {"en": ["Play-off tickets: €6.00, under 12 free"], "nl": ["Kaarten play-offs: € 6,00, tot 12 jaar gratis"], "de": ["Play-off-Tickets: 6,00 €, unter 12 frei"],
             "fr": ["Billets play-offs : 6,00 €, gratuit pour les moins de 12 ans"], "es": ["Entradas play-offs: 6,00 €, gratis para menores de 12"]}[lang]
    return page(rng, lang, rules, details, extra), noul(q), truth, margin


DISTRICTS = ["Noord", "Oost", "Zuidwest", "Centrum", "Westerpark", "Rivierenwijk", "Lindenhof", "Havenkwartier"]
PROJECTS_OK = [dict(en="street party", nl="straatfeest", de="Straßenfest", fr="fête de rue", es="fiesta vecinal"),
               dict(en="community garden", nl="buurtmoestuin", de="Gemeinschaftsgarten", fr="jardin partagé", es="huerto comunitario"),
               dict(en="music workshop", nl="muziekworkshop", de="Musikworkshop", fr="atelier de musique", es="taller de música"),
               dict(en="sports day", nl="sportdag", de="Sportfest", fr="journée sportive", es="jornada deportiva")]
PROJECTS_NO = [dict(en="commercial market", nl="commerciële markt", de="kommerzieller Markt", fr="marché commercial", es="mercado comercial"),
               dict(en="political event", nl="politieke bijeenkomst", de="politische Veranstaltung", fr="événement politique", es="acto político")]
SCHOOL = dict(en="school", nl="school", de="Schule", fr="école", es="colegio")
APPLICANTS = [dict(en="residents' association", nl="bewonersvereniging", de="Anwohnerverein", fr="association de riverains", es="asociación de vecinos"),
              dict(en="sports club", nl="sportclub", de="Sportverein", fr="club de sport", es="club deportivo")]


def grant_application_eligible(rng, lang):
    P = Plan(rng, 3, hard=(1,), waivable=1)
    pd = page_dates(rng, lang)
    cap = rng.choice([500, 750, 1000, 1500, 2500, 5000])
    asked = amt(rng, cap, P.passes[0], P.lv[0], 5)
    listed = rng.sample(DISTRICTS, 3)
    place = rng.choice(listed if P.passes[1] else [d for d in DISTRICTS if d not in listed])
    deadline = today(rng)
    g2 = dist(rng, P.lv[2], P.passes[2], "days")
    sent = deadline + dt.timedelta(days=-g2 if P.passes[2] else g2)
    who = (SCHOOL if P.waiver else rng.choice(APPLICANTS))[lang]
    what = rng.choice(PROJECTS_NO if P.block else PROJECTS_OK)[lang]
    m, names = (lambda x: money(x, lang, "€", 0)), ", ".join(listed)
    rules = {
        "en": [rng.choice(["Neighbourhood grant: conditions", "Conditions for the neighbourhood grant"]), f"You can apply for at most {m(cap)} per project.",
               f"The project must take place in one of these districts: {names}. Schools may apply for projects in any district.",
               f"Applications must be submitted on or before {pd(deadline)}.", "Commercial markets and political events are excluded."],
        "nl": [rng.choice(["Buurtsubsidie: voorwaarden", "Voorwaarden voor de buurtsubsidie"]), f"Je kunt maximaal {m(cap)} per project aanvragen.",
               f"Het project moet plaatsvinden in een van deze wijken: {names}. Scholen mogen voor projecten in elke wijk aanvragen.",
               f"Aanvragen moeten uiterlijk op {pd(deadline)} zijn ingediend.", "Commerciële markten en politieke bijeenkomsten zijn uitgesloten."],
        "de": [rng.choice(["Stadtteilförderung: Bedingungen", "Bedingungen der Stadtteilförderung"]), f"Pro Projekt können höchstens {m(cap)} beantragt werden.",
               f"Das Projekt muss in einem dieser Stadtteile stattfinden: {names}. Schulen können Projekte in jedem Stadtteil beantragen.",
               f"Anträge müssen spätestens am {pd(deadline)} eingereicht sein.", "Kommerzielle Märkte und politische Veranstaltungen sind ausgeschlossen."],
        "fr": [rng.choice(["Subvention de quartier : conditions", "Conditions de la subvention de quartier"]), f"Le montant demandé ne peut pas dépasser {m(cap)} par projet.",
               f"Le projet doit se dérouler dans l'un de ces quartiers : {names}. Les écoles peuvent déposer une demande pour un projet dans n'importe quel quartier.",
               f"Les demandes doivent être déposées au plus tard le {pd(deadline)}.", "Les marchés commerciaux et les événements politiques sont exclus."],
        "es": [rng.choice(["Subvención de barrio: condiciones", "Condiciones de la subvención de barrio"]), f"Se puede solicitar un máximo de {m(cap)} por proyecto.",
               f"El proyecto debe realizarse en uno de estos barrios: {names}. Los colegios pueden solicitarla para proyectos en cualquier barrio.",
               f"Las solicitudes deben presentarse el {pd(deadline)} como muy tarde.", "Quedan excluidos los mercados comerciales y los actos políticos."]}[lang]
    ref = f"{rng.randint(2026, 2027)}-{rng.randint(100, 999)}"
    details = {
        "en": [f"Application {ref}", f"Applicant: {who}", f"Project: {what}", f"Location: {place}", f"Amount requested: {m(asked)}", f"Submitted: {pd(sent)}"],
        "nl": [f"Aanvraag {ref}", f"Aanvrager: {who}", f"Project: {what}", f"Locatie: {place}", f"Aangevraagd bedrag: {m(asked)}", f"Ingediend: {pd(sent)}"],
        "de": [f"Antrag {ref}", f"Antragsteller: {who}", f"Projekt: {what}", f"Ort: {place}", f"Beantragter Betrag: {m(asked)}", f"Eingereicht: {pd(sent)}"],
        "fr": [f"Demande {ref}", f"Demandeur : {who}", f"Projet : {what}", f"Lieu : {place}", f"Montant demandé : {m(asked)}", f"Déposée le : {pd(sent)}"],
        "es": [f"Solicitud {ref}", f"Solicitante: {who}", f"Proyecto: {what}", f"Lugar: {place}", f"Importe solicitado: {m(asked)}", f"Presentada: {pd(sent)}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Does this application meet the conditions for the grant?", "Is this project eligible for the neighbourhood grant?",
               "Does this application satisfy the grant rules?"],
        "nl": ["Voldoet deze aanvraag aan de voorwaarden voor de subsidie?", "Komt dit project in aanmerking voor de buurtsubsidie?"],
        "de": ["Erfüllt dieser Antrag die Bedingungen der Förderung?", "Ist dieses Projekt nach den Bedingungen förderfähig?"],
        "fr": ["Cette demande remplit-elle les conditions de la subvention ?", "Ce projet est-il éligible à la subvention de quartier ?"],
        "es": ["¿Esta solicitud cumple las condiciones de la subvención?", "¿Este proyecto puede optar a la subvención de barrio?"]})
    truth, margin = verdict(P, [asked <= cap, place in listed, sent <= deadline], [rel_grade(asked, cap), "far", grade_by(g2, "days")])
    return page(rng, lang, rules, details), noul(q), truth, margin


JOBS = [dict(en="drilling", nl="boren", de="Bohren", fr="perçage", es="taladrar"),
        dict(en="sanding the floor", nl="vloer schuren", de="Boden schleifen", fr="ponçage du parquet", es="lijar el suelo"),
        dict(en="hammering", nl="timmeren", de="Hämmern", fr="travaux au marteau", es="martillear")]
URGENT = dict(en="urgent repair (burst water pipe)", nl="spoedreparatie (gesprongen waterleiding)", de="Notreparatur (Wasserrohrbruch)",
              fr="réparation urgente (rupture de canalisation)", es="reparación urgente (rotura de tubería)")


def noisy_work_allowed(rng, lang):
    P = Plan(rng, 2, waivable=0)
    if P.kind == "block" or not P.ok:
        P.waiver = False  # an urgent repair is allowed whatever else holds, so it only appears where the answer is yes
    cs, pd = clock_style(rng, lang), page_dates(rng, lang)
    wo, wc, so, sc = rng.choice([7, 8, 9]) * 60, rng.choice([18, 19, 20]) * 60, rng.choice([9, 10]) * 60, rng.choice([16, 17]) * 60
    t, holiday = today(rng), False
    if P.block and side(rng):
        while t.weekday() != 6:
            t += dt.timedelta(days=1)
    elif P.block:
        holiday = True
        while t.weekday() > 4:
            t += dt.timedelta(days=1)
    elif t.weekday() == 6:
        t += dt.timedelta(days=rng.randint(1, 6))
    lo, hi = (so, sc) if t.weekday() == 5 else (wo, wc)
    length = rng.choice([30, 45, 60, 90, 120])
    g0, g1 = dist(rng, P.lv[0], P.passes[0], "min"), dist(rng, P.lv[1], P.passes[1], "min")
    if not P.passes[0]:
        start = lo - g0
        end = hi + g1 if not P.passes[1] else start + length
    elif not P.passes[1] or P.focus == 1:
        end = hi + g1 if not P.passes[1] else hi - g1
        start = end - length
    else:
        start = lo + g0
        end = start + length
    if start < 5 * 60 or end > 23 * 60 + 55:
        raise Skip
    c = lambda m: clock(m, cs)
    job = (URGENT if P.waiver else rng.choice(JOBS))[lang]
    hol = {"en": " (public holiday)", "nl": " (feestdag)", "de": " (Feiertag)", "fr": " (jour férié)", "es": " (festivo)"}[lang] if holiday else ""
    rules = {
        "en": [rng.choice(["House rules: noise", "Building rules on noise"]),
               f"Noisy work (drilling, sanding, hammering) is allowed Monday to Friday between {c(wo)} and {c(wc)} and on Saturday between {c(so)} and {c(sc)}. "
               "The work must start and finish within these hours.",
               "No noisy work on Sundays and public holidays.", "Urgent repairs (for example a burst pipe) are allowed at any time on any day."],
        "nl": [rng.choice(["Huisregels: geluid", "Huisregels over geluidsoverlast"]),
               f"Lawaaiige klussen (boren, schuren, timmeren) mogen van maandag tot en met vrijdag tussen {c(wo)} en {c(wc)} uur en op zaterdag tussen {c(so)} en {c(sc)} uur. "
               "De klus moet binnen deze tijden beginnen en eindigen.",
               "Op zondag en op feestdagen zijn lawaaiige klussen niet toegestaan.", "Spoedreparaties (bijvoorbeeld een gesprongen leiding) mogen altijd, op elke dag."],
        "de": [rng.choice(["Hausordnung: Lärm", "Hausordnung – Ruhezeiten"]),
               f"Lärmende Arbeiten (Bohren, Schleifen, Hämmern) sind montags bis freitags zwischen {c(wo)} und {c(wc)} Uhr und samstags zwischen {c(so)} und {c(sc)} Uhr erlaubt. "
               "Die Arbeit muss innerhalb dieser Zeiten beginnen und enden.",
               "An Sonn- und Feiertagen sind lärmende Arbeiten nicht erlaubt.", "Notreparaturen (zum Beispiel ein Rohrbruch) sind jederzeit und an jedem Tag erlaubt."],
        "fr": [rng.choice(["Règlement intérieur : bruit", "Règlement de l'immeuble : nuisances sonores"]),
               f"Les travaux bruyants (perçage, ponçage, marteau) sont autorisés du lundi au vendredi entre {c(wo)} et {c(wc)} et le samedi entre {c(so)} et {c(sc)}. "
               "Les travaux doivent commencer et se terminer dans ces horaires.",
               "Aucun travail bruyant le dimanche et les jours fériés.",
               "Les réparations urgentes (par exemple une rupture de canalisation) sont autorisées à toute heure, tous les jours."],
        "es": [rng.choice(["Normas de la comunidad: ruido", "Normas del edificio sobre ruidos"]),
               f"Los trabajos ruidosos (taladrar, lijar, martillear) se permiten de lunes a viernes entre las {c(wo)} y las {c(wc)} y los sábados entre las {c(so)} y las {c(sc)}. "
               "El trabajo debe empezar y terminar dentro de ese horario.",
               "No se permiten trabajos ruidosos los domingos ni los festivos.",
               "Las reparaciones urgentes (por ejemplo, la rotura de una tubería) se permiten a cualquier hora y cualquier día."]}[lang]
    when = f"{weekday(t, lang)} {pd(t)}{hol}"
    details = {
        "en": ["Planned work", f"Job: {job}", f"Day: {when}", f"Time: {c(start)}–{c(end)}"],
        "nl": ["Geplande klus", f"Klus: {job}", f"Dag: {when}", f"Tijd: {c(start)}–{c(end)}"],
        "de": ["Geplante Arbeit", f"Arbeit: {job}", f"Tag: {when}", f"Zeit: {c(start)}–{c(end)}"],
        "fr": ["Travaux prévus", f"Travaux : {job}", f"Jour : {when}", f"Horaire : {c(start)}–{c(end)}"],
        "es": ["Trabajo previsto", f"Trabajo: {job}", f"Día: {when}", f"Horario: {c(start)}–{c(end)}"]}[lang]
    q = ask(rng, lang, {
        "en": ["May I do this work at the planned time?", "Do the house rules allow this work on the planned day and time?",
               "Is the planned work allowed under the house rules?"],
        "nl": ["Mag ik deze klus op het geplande tijdstip doen?", "Staan de huisregels deze klus op de geplande dag en tijd toe?"],
        "de": ["Darf ich diese Arbeit zur geplanten Zeit ausführen?", "Erlaubt die Hausordnung diese Arbeit am geplanten Tag zur geplanten Zeit?"],
        "fr": ["Puis-je faire ces travaux à l'horaire prévu ?", "Le règlement autorise-t-il ces travaux au jour et à l'horaire prévus ?"],
        "es": ["¿Puedo hacer este trabajo en el horario previsto?", "¿Las normas permiten este trabajo el día y a la hora previstos?"]})
    truth, margin = verdict(P, [start >= lo, end <= hi], [grade_by(abs(start - lo), "min"), grade_by(abs(end - hi), "min")])
    extra = {"en": ["Bicycles belong in the basement, not in the stairwell.", "Service charge 2026: €84.00 per month"],
             "nl": ["Fietsen horen in de kelder, niet in het trappenhuis.", "Servicekosten 2026: € 84,00 per maand"],
             "de": ["Fahrräder gehören in den Keller, nicht ins Treppenhaus.", "Nebenkosten 2026: 84,00 € pro Monat"],
             "fr": ["Les vélos se rangent au sous-sol, pas dans la cage d'escalier.", "Charges 2026 : 84,00 € par mois"],
             "es": ["Las bicicletas van en el sótano, no en la escalera.", "Gastos de comunidad 2026: 84,00 € al mes"]}[lang]
    return page(rng, lang, rules, details, extra), noul(q), truth, margin


WASTE_OK = [dict(en="sofa", nl="bank", de="Sofa", fr="canapé", es="sofá"), dict(en="mattress", nl="matras", de="Matratze", fr="matelas", es="colchón"),
            dict(en="bookcase", nl="boekenkast", de="Bücherregal", fr="bibliothèque", es="estantería"),
            dict(en="washing machine", nl="wasmachine", de="Waschmaschine", fr="lave-linge", es="lavadora"),
            dict(en="carpet", nl="vloerkleed", de="Teppich", fr="tapis", es="alfombra"), dict(en="table", nl="tafel", de="Tisch", fr="table", es="mesa")]
WASTE_NO = [dict(en="rubble", nl="bouwpuin", de="Bauschutt", fr="gravats", es="escombros"),
            dict(en="car tyres", nl="autobanden", de="Autoreifen", fr="pneus", es="neumáticos"),
            dict(en="paint tins", nl="verfblikken", de="Farbdosen", fr="pots de peinture", es="botes de pintura")]


def bulky_waste_pickup_accepted(rng, lang):
    P = Plan(rng, 2, waivable=0)
    t, pd = today(rng), page_dates(rng, lang)
    maxi, n_days = rng.choice([3, 4, 5, 6]), rng.choice([2, 3, 5, 7])
    g0 = dist(rng, P.lv[0], P.passes[0], "count")
    items = max(1, maxi - g0) if P.passes[0] else maxi + g0
    # "N days before": kept off the exact day, where readers count differently.
    g1 = dist(rng, P.lv[1], P.passes[1], "days", least=1)
    ahead = n_days + g1 if P.passes[1] else max(0, n_days - g1)
    day = t + dt.timedelta(days=ahead)
    shown = rng.sample(WASTE_OK, min(items, rng.randint(1, 3)))
    if P.block:
        shown[rng.randrange(len(shown))] = rng.choice(WASTE_NO)
    names, yn = ", ".join(w[lang] for w in shown), (YES if P.waiver else NO)[lang]
    rules = {
        "en": [rng.choice(["Bulky waste collection", "Bulky waste: collection on request"]),
               f"At most {maxi} items per collection. Households with a registered house move have no item limit.",
               f"Request the collection no later than {n_days} days before the collection date.",
               "Rubble, car tyres and paint tins are not collected: a request that includes any of them is refused."],
        "nl": [rng.choice(["Grofvuil ophalen", "Grofvuil: ophalen op afspraak"]),
               f"Maximaal {maxi} stuks per ophaalbeurt. Voor huishoudens met een gemelde verhuizing geldt geen maximum.",
               f"Vraag de ophaalbeurt uiterlijk {n_days} dagen vóór de ophaaldatum aan.",
               "Bouwpuin, autobanden en verfblikken halen we niet op: een aanvraag waar die in zitten, wordt geweigerd."],
        "de": [rng.choice(["Sperrmüllabholung", "Sperrmüll: Abholung auf Anmeldung"]),
               f"Höchstens {maxi} Teile pro Abholung. Für Haushalte mit gemeldetem Umzug gilt keine Obergrenze.",
               f"Die Abholung muss spätestens {n_days} Tage vor dem Abholtermin angemeldet werden.",
               "Bauschutt, Autoreifen und Farbdosen werden nicht abgeholt: Anmeldungen, die solche Teile enthalten, werden abgelehnt."],
        "fr": [rng.choice(["Collecte des encombrants", "Encombrants : collecte sur demande"]),
               f"{maxi} objets au maximum par collecte. Cette limite ne s'applique pas aux foyers ayant déclaré un déménagement.",
               f"La demande doit être faite au plus tard {n_days} jours avant la date de collecte.",
               "Les gravats, les pneus et les pots de peinture ne sont pas collectés : toute demande qui en contient est refusée."],
        "es": [rng.choice(["Recogida de voluminosos", "Voluminosos: recogida previa solicitud"]),
               f"Máximo {maxi} objetos por recogida. Este límite no se aplica a los hogares con una mudanza registrada.",
               f"La solicitud debe hacerse como muy tarde {n_days} días antes de la fecha de recogida.",
               "No se recogen escombros, neumáticos ni botes de pintura: las solicitudes que los incluyan se rechazan."]}[lang]
    details = {
        "en": ["Your request", f"Request date: {pd(t)} (today)", f"Collection date: {weekday(day, lang)} {pd(day)}", f"Number of items: {items}",
               f"Includes: {names}", f"House move registered: {yn}"],
        "nl": ["Je aanvraag", f"Datum aanvraag: {pd(t)} (vandaag)", f"Ophaaldatum: {weekday(day, lang)} {pd(day)}", f"Aantal stuks: {items}",
               f"Waaronder: {names}", f"Verhuizing gemeld: {yn}"],
        "de": ["Ihre Anmeldung", f"Datum der Anmeldung: {pd(t)} (heute)", f"Abholtermin: {weekday(day, lang)}, {pd(day)}", f"Anzahl Teile: {items}",
               f"Darunter: {names}", f"Umzug gemeldet: {yn}"],
        "fr": ["Votre demande", f"Date de la demande : {pd(t)} (aujourd'hui)", f"Date de collecte : {weekday(day, lang)} {pd(day)}", f"Nombre d'objets : {items}",
               f"Dont : {names}", f"Déménagement déclaré : {yn}"],
        "es": ["Tu solicitud", f"Fecha de la solicitud: {pd(t)} (hoy)", f"Fecha de recogida: {weekday(day, lang)} {pd(day)}", f"Número de objetos: {items}",
               f"Entre ellos: {names}", f"Mudanza registrada: {yn}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Will this collection request be accepted?", "Does my request meet the bulky waste rules?", "Does this request satisfy the collection rules?"],
        "nl": ["Wordt deze aanvraag geaccepteerd?", "Voldoet mijn aanvraag aan de grofvuilregels?"],
        "de": ["Wird diese Anmeldung angenommen?", "Erfüllt meine Anmeldung die Sperrmüllregeln?"],
        "fr": ["Cette demande de collecte sera-t-elle acceptée ?", "Ma demande respecte-t-elle les règles de collecte ?"],
        "es": ["¿Se aceptará esta solicitud de recogida?", "¿Mi solicitud cumple las normas de recogida?"]})
    truth, margin = verdict(P, [items <= maxi, ahead >= n_days], [grade_by(abs(items - maxi), "count"), grade_by(abs(ahead - n_days), "days")])
    extra = {"en": ["Recycling centre: Tuesday to Saturday", "Extra collection outside the schedule: €25.00"],
             "nl": ["Milieustraat: dinsdag tot en met zaterdag", "Extra ophaalbeurt buiten het rooster: € 25,00"],
             "de": ["Wertstoffhof: Dienstag bis Samstag", "Zusätzliche Abholung außerhalb des Plans: 25,00 €"],
             "fr": ["Déchetterie : du mardi au samedi", "Collecte supplémentaire hors calendrier : 25,00 €"],
             "es": ["Punto limpio: de martes a sábado", "Recogida extra fuera de calendario: 25,00 €"]}[lang]
    return page(rng, lang, rules, details, extra), noul(q), truth, margin


ROOMS = ["Atlas", "Orion", "Studio 2", "Loft", "Kyoto", "Havana", "Room B"]
DAY_PASS = dict(en="Day pass", nl="Dagpas", de="Tagespass", fr="Pass journée", es="Pase de día")


def meeting_room_booking_valid(rng, lang):
    P = Plan(rng, 2, waivable=0)
    if P.block:
        P.waiver = False  # one plan per member: a Day pass is not a Team plan
    cs = clock_style(rng, lang)
    maxh, close = rng.choice([2, 3, 4]), rng.choice([18, 19, 20, 21]) * 60
    g0, g1 = dist(rng, P.lv[0], P.passes[0], "quarter"), dist(rng, P.lv[1], P.passes[1], "quarter")
    length = maxh * 60 + (-g0 if P.passes[0] else g0)
    end = close + (-g1 if P.passes[1] else g1)
    start = end - length
    if length < 30 or start < 8 * 60 or end > 23 * 60 + 45:
        raise Skip
    c = lambda m: clock(m, cs)
    tier = "Team" if P.waiver else (DAY_PASS[lang] if P.block else rng.choice(["Flex", "Fixed desk"]))
    room, day = rng.choice(ROOMS), DAYS[lang][rng.randrange(5)]
    rules = {
        "en": [rng.choice(["Meeting rooms", "Meeting room rules"]), f"A booking may last at most {maxh} hours. Members on the Team plan have no time limit.",
               f"Every booking must end by {c(close)} at the latest.", "Day pass holders cannot book meeting rooms."],
        "nl": [rng.choice(["Vergaderruimtes", "Regels voor vergaderruimtes"]),
               f"Een reservering duurt maximaal {maxh} uur. Voor leden met het Team-abonnement geldt geen tijdslimiet.",
               f"Elke reservering moet uiterlijk om {c(close)} uur eindigen.", "Met een Dagpas kun je geen vergaderruimte reserveren."],
        "de": [rng.choice(["Besprechungsräume", "Regeln für Besprechungsräume"]),
               f"Eine Buchung darf höchstens {maxh} Stunden dauern. Für Mitglieder im Team-Tarif gilt keine Zeitbegrenzung.",
               f"Jede Buchung muss spätestens um {c(close)} Uhr enden.", "Mit einem Tagespass können keine Besprechungsräume gebucht werden."],
        "fr": [rng.choice(["Salles de réunion", "Règles des salles de réunion"]),
               f"Une réservation dure {maxh} heures au maximum. Les membres de la formule Team n'ont pas de limite de durée.",
               f"Toute réservation doit se terminer à {c(close)} au plus tard.", "Les titulaires d'un Pass journée ne peuvent pas réserver de salle de réunion."],
        "es": [rng.choice(["Salas de reuniones", "Normas de las salas de reuniones"]),
               f"Una reserva puede durar como máximo {maxh} horas. Los miembros del plan Team no tienen límite de tiempo.",
               f"Todas las reservas deben terminar a las {c(close)} como muy tarde.", "Con un Pase de día no se pueden reservar salas de reuniones."]}[lang]
    details = {
        "en": ["Your booking", f"Room: {room}", f"When: {day}, {c(start)}–{c(end)}", f"Your plan: {tier}"],
        "nl": ["Je reservering", f"Ruimte: {room}", f"Wanneer: {day}, {c(start)}–{c(end)}", f"Jouw abonnement: {tier}"],
        "de": ["Ihre Buchung", f"Raum: {room}", f"Wann: {day}, {c(start)}–{c(end)}", f"Ihr Tarif: {tier}"],
        "fr": ["Votre réservation", f"Salle : {room}", f"Quand : {day}, {c(start)}–{c(end)}", f"Votre formule : {tier}"],
        "es": ["Tu reserva", f"Sala: {room}", f"Cuándo: {day}, {c(start)}–{c(end)}", f"Tu plan: {tier}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Is this booking allowed under the meeting room rules?", "Can I book the room for this time slot?", "Does this booking satisfy the rules?"],
        "nl": ["Is deze reservering toegestaan volgens de regels?", "Kan ik de ruimte voor dit tijdvak reserveren?"],
        "de": ["Ist diese Buchung nach den Regeln zulässig?", "Kann ich den Raum für dieses Zeitfenster buchen?"],
        "fr": ["Cette réservation est-elle conforme aux règles ?", "Puis-je réserver la salle pour ce créneau ?"],
        "es": ["¿Esta reserva cumple las normas?", "¿Puedo reservar la sala para esta franja?"]})
    truth, margin = verdict(P, [length <= maxh * 60, end <= close], [grade_by(g0, "quarter"), grade_by(g1, "quarter")])
    extra = {"en": ["Coffee and tea are included.", "Room rate: 2 credits per hour"], "nl": ["Koffie en thee zijn inbegrepen.", "Tarief: 2 credits per uur"],
             "de": ["Kaffee und Tee sind inklusive.", "Preis: 2 Credits pro Stunde"], "fr": ["Café et thé inclus.", "Tarif : 2 crédits par heure"],
             "es": ["Café y té incluidos.", "Tarifa: 2 créditos por hora"]}[lang]
    return page(rng, lang, rules, details, extra), noul(q), truth, margin


ROLES = [dict(en="kitchen", nl="keuken", de="Küche", fr="cuisine", es="cocina"), dict(en="reception desk", nl="receptie", de="Empfang", fr="accueil", es="recepción"),
         dict(en="cash desk", nl="kassa", de="Kasse", fr="caisse", es="caja")]
SETUP = dict(en="set-up crew", nl="opbouwploeg", de="Aufbauteam", fr="équipe de montage", es="equipo de montaje")


def volunteer_shift_signup_allowed(rng, lang):
    P = Plan(rng, 2, hard=(1,), waivable=1)
    pd, name = page_dates(rng, lang), rng.choice(NAMES)
    cap, length = rng.choice([12, 16, 20, 24]), rng.choice([3, 4, 5, 6])
    g0 = dist(rng, P.lv[0], P.passes[0], "count")
    planned = cap + (-g0 if P.passes[0] else g0) - length
    if planned < 0:
        raise Skip
    day = today(rng)
    role = (SETUP if P.waiver else rng.choice(ROLES))[lang]
    done, other = (YES if P.passes[1] else NO)[lang], (YES if P.block else NO)[lang]
    rules = {
        "en": [rng.choice(["Signing up for shifts", "Shift sign-up rules"]), f"Volunteers may work at most {cap} hours per calendar month, the new shift included.",
               "You need to have completed the induction session, except for shifts with the set-up crew.", "You cannot sign up for two shifts on the same day."],
        "nl": [rng.choice(["Inschrijven voor diensten", "Regels voor het inschrijven"]),
               f"Vrijwilligers mogen maximaal {cap} uur per kalendermaand werken, de nieuwe dienst meegerekend.",
               "Je moet de introductiebijeenkomst hebben gevolgd, behalve voor diensten bij de opbouwploeg.",
               "Je kunt je niet voor twee diensten op dezelfde dag inschrijven."],
        "de": [rng.choice(["Für Schichten anmelden", "Regeln für die Schichtanmeldung"]),
               f"Ehrenamtliche dürfen höchstens {cap} Stunden pro Kalendermonat arbeiten, die neue Schicht eingerechnet.",
               "Die Einführungsveranstaltung muss absolviert sein, außer für Schichten im Aufbauteam.",
               "Man kann sich nicht für zwei Schichten am selben Tag anmelden."],
        "fr": [rng.choice(["S'inscrire à un créneau", "Règles d'inscription aux créneaux"]),
               f"Les bénévoles peuvent travailler au maximum {cap} heures par mois civil, nouveau créneau compris.",
               "Il faut avoir suivi la séance d'accueil, sauf pour les créneaux de l'équipe de montage.",
               "Il n'est pas possible de s'inscrire à deux créneaux le même jour."],
        "es": [rng.choice(["Apuntarse a un turno", "Normas para apuntarse a turnos"]),
               f"Los voluntarios pueden trabajar como máximo {cap} horas por mes natural, incluido el nuevo turno.",
               "Hay que haber hecho la sesión de acogida, salvo para los turnos del equipo de montaje.",
               "No es posible apuntarse a dos turnos el mismo día."]}[lang]
    details = {
        "en": [f"Volunteer: {name}", f"Shift: {role}, {weekday(day, lang)} {pd(day)}, {length} hours", f"Hours already planned that month: {planned}",
               f"Induction session completed: {done}", f"Other shift on that day: {other}"],
        "nl": [f"Vrijwilliger: {name}", f"Dienst: {role}, {weekday(day, lang)} {pd(day)}, {length} uur", f"Al ingeplande uren die maand: {planned}",
               f"Introductiebijeenkomst gevolgd: {done}", f"Andere dienst op die dag: {other}"],
        "de": [f"Ehrenamtliche Person: {name}", f"Schicht: {role}, {weekday(day, lang)}, {pd(day)}, {length} Stunden", f"Bereits geplante Stunden in dem Monat: {planned}",
               f"Einführungsveranstaltung absolviert: {done}", f"Andere Schicht an dem Tag: {other}"],
        "fr": [f"Bénévole : {name}", f"Créneau : {role}, {weekday(day, lang)} {pd(day)}, {length} heures", f"Heures déjà prévues ce mois-là : {planned}",
               f"Séance d'accueil suivie : {done}", f"Autre créneau ce jour-là : {other}"],
        "es": [f"Voluntario/a: {name}", f"Turno: {role}, {weekday(day, lang)} {pd(day)}, {length} horas", f"Horas ya previstas ese mes: {planned}",
               f"Sesión de acogida realizada: {done}", f"Otro turno ese día: {other}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Can {n} sign up for this shift?", "Is {n} allowed to take this shift?", "Do the sign-up rules let {n} take this shift?"],
        "nl": ["Kan {n} zich voor deze dienst inschrijven?", "Mag {n} deze dienst volgens de regels draaien?"],
        "de": ["Kann sich {n} für diese Schicht anmelden?", "Darf {n} diese Schicht nach den Regeln übernehmen?"],
        "fr": ["Est-ce que {n} peut s'inscrire à ce créneau ?", "Le règlement permet-il à {n} de prendre ce créneau ?"],
        "es": ["¿Puede {n} apuntarse a este turno?", "¿Las normas permiten que {n} haga este turno?"]}, n=name)
    truth, margin = verdict(P, [planned + length <= cap, P.passes[1]], [grade_by(g0, "count"), "far"])
    return page(rng, lang, rules, details), noul(q), truth, margin


LABS = ["B2.14", "Chem 3", "Laser lab", "Wet lab 1", "C0.07", "Bio 2"]


def lab_access_allowed(rng, lang):
    P = Plan(rng, 2, waivable=1)
    t, pd, name = today(rng), page_dates(rng, lang), rng.choice(NAMES)
    months = rng.choice([12, 24])
    g0 = dist(rng, P.lv[0], P.passes[0], "days", least=1)  # never on the expiry date itself
    done = add_months(t + dt.timedelta(days=g0 if P.passes[0] else -g0), -months)
    expiry = add_months(done, months)
    if expiry == t:
        raise Skip
    if P.passes[1]:
        hazard = rng.randint(1, 4)
        level = hazard if P.lv[1] == "near" else rng.randint(hazard, 4)
    else:
        hazard = rng.randint(2, 4)
        level = hazard - 1 if P.lv[1] == "near" else rng.randint(1, hazard - 1)
    gap = abs(level - hazard) + (1 if level >= hazard else 0)
    g1 = "near" if gap == 1 else ("mid" if gap == 2 else "far")
    span = rng.randint(1, 6)
    if P.block:
        shut = t - dt.timedelta(days=rng.randint(0, span))
    else:
        shut = t + dt.timedelta(days=rng.randint(3, 20)) if side(rng) else t - dt.timedelta(days=span + rng.randint(3, 20))
    reopen = shut + dt.timedelta(days=span)
    lab, sup = rng.choice(LABS), (YES if P.waiver else NO)[lang]
    rules = {
        "en": [rng.choice(["Lab access rules", "Rules for entering the labs"]),
               f"Your safety training must still be valid: it expires {months} months after the date you completed it.",
               "Your clearance level must be at least the hazard level of the lab. Under direct supervision of a lab manager you may enter a lab of any level.",
               f"Closed for maintenance from {pd(shut)} up to and including {pd(reopen)}: no access for anyone."],
        "nl": [rng.choice(["Toegangsregels lab", "Regels voor toegang tot de labs"]),
               f"Je veiligheidstraining moet nog geldig zijn: die verloopt {months} maanden na de datum waarop je hem hebt afgerond.",
               "Je bevoegdheidsniveau moet minstens gelijk zijn aan het risiconiveau van het lab. Onder direct toezicht van een labmanager mag je een lab van elk niveau in.",
               f"Gesloten wegens onderhoud van {pd(shut)} tot en met {pd(reopen)}: geen toegang voor wie dan ook."],
        "de": [rng.choice(["Zugangsregeln Labor", "Regeln für den Laborzutritt"]),
               f"Die Sicherheitsunterweisung muss noch gültig sein: Sie läuft {months} Monate nach dem Datum der Teilnahme ab.",
               "Die Freigabestufe muss mindestens der Gefahrenstufe des Labors entsprechen. Unter direkter Aufsicht einer Laborleitung ist der Zutritt zu Laboren jeder Stufe erlaubt.",
               f"Wegen Wartung geschlossen vom {pd(shut)} bis einschließlich {pd(reopen)}: kein Zutritt, für niemanden."],
        "fr": [rng.choice(["Règles d'accès au laboratoire", "Accès aux laboratoires : règles"]),
               f"La formation sécurité doit être encore valable : elle expire {months} mois après la date à laquelle elle a été suivie.",
               "Le niveau d'habilitation doit être au moins égal au niveau de risque du laboratoire. Sous la supervision directe d'un responsable de laboratoire, "
               "l'accès est autorisé quel que soit le niveau.",
               f"Fermé pour maintenance du {pd(shut)} au {pd(reopen)} inclus : aucun accès, pour personne."],
        "es": [rng.choice(["Normas de acceso al laboratorio", "Acceso a los laboratorios: normas"]),
               f"La formación de seguridad debe seguir vigente: caduca {months} meses después de la fecha en que se completó.",
               "El nivel de autorización debe ser al menos igual al nivel de riesgo del laboratorio. Bajo la supervisión directa de un responsable de laboratorio "
               "se puede entrar en laboratorios de cualquier nivel.",
               f"Cerrado por mantenimiento del {pd(shut)} al {pd(reopen)} inclusive: sin acceso para nadie."]}[lang]
    details = {
        "en": [f"User: {name}", f"Safety training completed: {pd(done)}", f"Clearance level: {level}", f"Lab: {lab} (hazard level {hazard})",
               f"Supervised by a lab manager today: {sup}", f"{TODAY[lang]}: {pd(t)}"],
        "nl": [f"Gebruiker: {name}", f"Veiligheidstraining afgerond: {pd(done)}", f"Bevoegdheidsniveau: {level}", f"Lab: {lab} (risiconiveau {hazard})",
               f"Vandaag onder toezicht van een labmanager: {sup}", f"{TODAY[lang]}: {pd(t)}"],
        "de": [f"Nutzer/in: {name}", f"Sicherheitsunterweisung absolviert: {pd(done)}", f"Freigabestufe: {level}", f"Labor: {lab} (Gefahrenstufe {hazard})",
               f"Heute unter Aufsicht einer Laborleitung: {sup}", f"{TODAY[lang]}: {pd(t)}"],
        "fr": [f"Utilisateur : {name}", f"Formation sécurité suivie le : {pd(done)}", f"Niveau d'habilitation : {level}", f"Laboratoire : {lab} (niveau de risque {hazard})",
               f"Supervision par un responsable aujourd'hui : {sup}", f"{TODAY[lang]} : {pd(t)}"],
        "es": [f"Usuario/a: {name}", f"Formación de seguridad completada: {pd(done)}", f"Nivel de autorización: {level}", f"Laboratorio: {lab} (nivel de riesgo {hazard})",
               f"Supervisión de un responsable hoy: {sup}", f"{TODAY[lang]}: {pd(t)}"]}[lang]
    q = ask(rng, lang, {
        "en": ["May {n} enter this lab today?", "Is {n} allowed to work in this lab today?", "Do the access rules let {n} into this lab today?"],
        "nl": ["Mag {n} vandaag dit lab in?", "Heeft {n} vandaag volgens de regels toegang tot dit lab?"],
        "de": ["Darf {n} heute dieses Labor betreten?", "Hat {n} heute nach den Regeln Zutritt zu diesem Labor?"],
        "fr": ["Est-ce que {n} peut entrer dans ce laboratoire aujourd'hui ?", "Les règles autorisent-elles {n} à accéder à ce laboratoire aujourd'hui ?"],
        "es": ["¿Puede {n} entrar hoy en este laboratorio?", "¿Las normas permiten a {n} acceder hoy a este laboratorio?"]}, n=name)
    truth, margin = verdict(P, [t < expiry, level >= hazard], [grade_by(abs((expiry - t).days), "days"), g1])
    if (shut <= t <= reopen) != P.block:
        raise Skip
    return page(rng, lang, rules, details), noul(q), truth, margin


DOMAINS = ["northwind.example", "acme-labs.example", "brightside.example", "delta-ops.example", "fjordworks.example"]
OTHER_DOMAINS = ["gmail.com", "outlook.com", "partner-agency.example", "freelance.example", "studio-kite.example"]
USERS = ["j.devries", "m.schmidt", "a.martin", "l.garcia", "s.jones", "t.bakker", "e.rossi", "n.dubois"]


def licence_seat_assignable(rng, lang):
    P = Plan(rng, 2, hard=(1,), waivable=1)
    total = rng.choice([5, 10, 15, 20, 25, 50])
    used, g0 = under_cap(rng, total, P.passes[0], P.lv[0])
    allowed = rng.sample(DOMAINS, 2)
    domain = rng.choice(allowed if P.passes[1] else OTHER_DOMAINS)
    email, names, inv = f"{rng.choice(USERS)}@{domain}", ", ".join(allowed), f"{rng.randint(2026, 2027)}-{rng.randint(100, 999)}"
    rules = {
        "en": [rng.choice(["Seats and invitations", "Adding users"]), "Every new user needs a free seat.",
               f"Members must have an email address at one of the approved domains: {names}. Guest accounts may use any email domain; they still take up a seat.",
               "While an invoice is overdue, no new users can be added."],
        "nl": [rng.choice(["Licenties en uitnodigingen", "Gebruikers toevoegen"]), "Elke nieuwe gebruiker heeft een vrije licentie nodig.",
               f"Leden moeten een e-mailadres hebben op een van de goedgekeurde domeinen: {names}. Gastaccounts mogen elk e-maildomein gebruiken; ze nemen wel een licentie in.",
               "Zolang een factuur over de vervaldatum is, kunnen er geen nieuwe gebruikers worden toegevoegd."],
        "de": [rng.choice(["Lizenzen und Einladungen", "Nutzer hinzufügen"]), "Jeder neue Nutzer benötigt eine freie Lizenz.",
               f"Mitglieder brauchen eine E-Mail-Adresse einer der zugelassenen Domains: {names}. Gastkonten dürfen jede E-Mail-Domain verwenden; sie belegen trotzdem eine Lizenz.",
               "Solange eine Rechnung überfällig ist, können keine neuen Nutzer hinzugefügt werden."],
        "fr": [rng.choice(["Licences et invitations", "Ajouter des utilisateurs"]), "Chaque nouvel utilisateur a besoin d'une licence libre.",
               f"Les membres doivent avoir une adresse e-mail sur l'un des domaines approuvés : {names}. Les comptes invités peuvent utiliser n'importe quel domaine ; "
               "ils occupent tout de même une licence.",
               "Tant qu'une facture est en retard de paiement, aucun nouvel utilisateur ne peut être ajouté."],
        "es": [rng.choice(["Licencias e invitaciones", "Añadir usuarios"]), "Cada usuario nuevo necesita una licencia libre.",
               f"Los miembros deben tener una dirección de correo de uno de los dominios aprobados: {names}. Las cuentas de invitado pueden usar cualquier dominio; "
               "aun así ocupan una licencia.",
               "Mientras haya una factura vencida, no se pueden añadir usuarios nuevos."]}[lang]
    kind = ({"en": "guest", "nl": "gast", "de": "Gast", "fr": "invité", "es": "invitado"} if P.waiver else
            {"en": "member", "nl": "lid", "de": "Mitglied", "fr": "membre", "es": "miembro"})[lang]
    bill = ({"en": f"invoice {inv} overdue", "nl": f"factuur {inv} over de vervaldatum", "de": f"Rechnung {inv} überfällig",
             "fr": f"facture {inv} en retard de paiement", "es": f"factura {inv} vencida"} if P.block else
            {"en": "all invoices paid", "nl": "alle facturen betaald", "de": "alle Rechnungen bezahlt", "fr": "toutes les factures sont payées",
             "es": "todas las facturas pagadas"})[lang]
    details = {
        "en": ["Workspace admin", f"Seats in use: {used} of {total}", f"Invite: {email}", f"Account type: {kind}", f"Billing: {bill}"],
        "nl": ["Beheer werkruimte", f"Licenties in gebruik: {used} van {total}", f"Uitnodigen: {email}", f"Accounttype: {kind}", f"Facturatie: {bill}"],
        "de": ["Workspace-Verwaltung", f"Belegte Lizenzen: {used} von {total}", f"Einladen: {email}", f"Kontotyp: {kind}", f"Abrechnung: {bill}"],
        "fr": ["Administration de l'espace", f"Licences utilisées : {used} sur {total}", f"Inviter : {email}", f"Type de compte : {kind}", f"Facturation : {bill}"],
        "es": ["Administración del espacio", f"Licencias en uso: {used} de {total}", f"Invitar: {email}", f"Tipo de cuenta: {kind}", f"Facturación: {bill}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Can I add this user to the workspace now?", "Do the rules allow this invitation?", "Can this person be given a seat right now?"],
        "nl": ["Kan ik deze gebruiker nu aan de werkruimte toevoegen?", "Is deze uitnodiging volgens de regels toegestaan?"],
        "de": ["Kann ich diesen Nutzer jetzt zum Workspace hinzufügen?", "Ist diese Einladung nach den Regeln zulässig?"],
        "fr": ["Puis-je ajouter cet utilisateur à l'espace maintenant ?", "Cette invitation est-elle autorisée par les règles ?"],
        "es": ["¿Puedo añadir a este usuario al espacio ahora?", "¿Las normas permiten esta invitación?"]})
    truth, margin = verdict(P, [used < total, domain in allowed], [g0, "far"])
    extra = {"en": ["Price per seat: €9.00 per month", "Next renewal: in 23 days"], "nl": ["Prijs per licentie: € 9,00 per maand", "Volgende verlenging: over 23 dagen"],
             "de": ["Preis pro Lizenz: 9,00 € pro Monat", "Nächste Verlängerung: in 23 Tagen"], "fr": ["Prix par licence : 9,00 € par mois", "Prochain renouvellement : dans 23 jours"],
             "es": ["Precio por licencia: 9,00 € al mes", "Próxima renovación: dentro de 23 días"]}[lang]
    return page(rng, lang, rules, details, extra), noul(q), truth, margin


CATS_OK = [dict(en="furniture", nl="meubels", de="Möbel", fr="meubles", es="muebles"), dict(en="bicycles", nl="fietsen", de="Fahrräder", fr="vélos", es="bicicletas"),
           dict(en="electronics", nl="elektronica", de="Elektronik", fr="électronique", es="electrónica"),
           dict(en="musical instruments", nl="muziekinstrumenten", de="Musikinstrumente", fr="instruments de musique", es="instrumentos musicales")]
CATS_NO = [dict(en="event tickets", nl="evenemententickets", de="Veranstaltungstickets", fr="billets de spectacle", es="entradas para eventos"),
           dict(en="medicines", nl="medicijnen", de="Medikamente", fr="médicaments", es="medicamentos"),
           dict(en="live animals", nl="levende dieren", de="lebende Tiere", fr="animaux vivants", es="animales vivos")]


def classifieds_listing_accepted(rng, lang):
    P = Plan(rng, 2, waivable=1)
    minp, cap = rng.choice([2, 3, 4]), rng.choice([500, 1000, 2000, 3000])
    g0 = dist(rng, P.lv[0], P.passes[0], "count")
    photos = minp + g0 if P.passes[0] else max(0, minp - g0)
    price = amt(rng, cap, P.passes[1], P.lv[1], 5)
    cat = rng.choice(CATS_NO if P.block else CATS_OK)[lang]
    m = lambda x: money(x, lang, "€", 0)
    banned = ", ".join(c[lang] for c in CATS_NO)
    rules = {
        "en": [rng.choice(["Posting rules", "Rules for listings"]), f"A listing needs at least {minp} photos.",
               f"The asking price may not exceed {m(cap)}. This price limit does not apply to verified sellers.",
               f"Listings in these categories are not accepted: {banned}."],
        "nl": [rng.choice(["Plaatsingsregels", "Regels voor advertenties"]), f"Een advertentie heeft minstens {minp} foto's nodig.",
               f"De vraagprijs mag niet hoger zijn dan {m(cap)}. Deze prijslimiet geldt niet voor geverifieerde verkopers.",
               f"Advertenties in deze categorieën worden niet geaccepteerd: {banned}."],
        "de": [rng.choice(["Regeln für Anzeigen", "Anzeigenrichtlinien"]), f"Eine Anzeige braucht mindestens {minp} Fotos.",
               f"Der Preis darf {m(cap)} nicht übersteigen. Diese Preisgrenze gilt nicht für verifizierte Verkäufer.",
               f"Anzeigen in diesen Kategorien werden nicht angenommen: {banned}."],
        "fr": [rng.choice(["Règles de publication", "Règles pour les annonces"]), f"Une annonce doit comporter au moins {minp} photos.",
               f"Le prix demandé ne peut pas dépasser {m(cap)}. Ce plafond ne s'applique pas aux vendeurs vérifiés.",
               f"Les annonces de ces catégories ne sont pas acceptées : {banned}."],
        "es": [rng.choice(["Normas de publicación", "Normas para los anuncios"]), f"Un anuncio necesita al menos {minp} fotos.",
               f"El precio no puede superar los {m(cap)}. Este límite no se aplica a los vendedores verificados.",
               f"No se aceptan anuncios de estas categorías: {banned}."]}[lang]
    status = ({"en": "verified", "nl": "geverifieerd", "de": "verifiziert", "fr": "vérifié", "es": "verificado"} if P.waiver else
              {"en": "not verified", "nl": "niet geverifieerd", "de": "nicht verifiziert", "fr": "non vérifié", "es": "no verificado"})[lang]
    details = {
        "en": ["New listing (draft)", f"Category: {cat}", f"Asking price: {m(price)}", f"Photos uploaded: {photos}", f"Seller status: {status}"],
        "nl": ["Nieuwe advertentie (concept)", f"Categorie: {cat}", f"Vraagprijs: {m(price)}", f"Geüploade foto's: {photos}", f"Verkopersstatus: {status}"],
        "de": ["Neue Anzeige (Entwurf)", f"Kategorie: {cat}", f"Preis: {m(price)}", f"Hochgeladene Fotos: {photos}", f"Verkäuferstatus: {status}"],
        "fr": ["Nouvelle annonce (brouillon)", f"Catégorie : {cat}", f"Prix demandé : {m(price)}", f"Photos ajoutées : {photos}", f"Statut du vendeur : {status}"],
        "es": ["Nuevo anuncio (borrador)", f"Categoría: {cat}", f"Precio: {m(price)}", f"Fotos subidas: {photos}", f"Estado del vendedor: {status}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Will this listing be accepted as it is?", "Does my listing comply with the posting rules?", "Does this draft satisfy the rules for listings?"],
        "nl": ["Wordt deze advertentie zo geaccepteerd?", "Voldoet mijn advertentie aan de plaatsingsregels?"],
        "de": ["Wird diese Anzeige so angenommen?", "Entspricht meine Anzeige den Regeln?"],
        "fr": ["Cette annonce sera-t-elle acceptée telle quelle ?", "Mon annonce respecte-t-elle les règles de publication ?"],
        "es": ["¿Se aceptará este anuncio tal como está?", "¿Mi anuncio cumple las normas de publicación?"]})
    truth, margin = verdict(P, [photos >= minp, price <= cap], [grade_by(abs(photos - minp), "count"), rel_grade(price, cap)])
    extra = {"en": ["Highlight your listing for 7 days: €2.50", "Listings stay online for 60 days."],
             "nl": ["Zet je advertentie 7 dagen bovenaan: € 2,50", "Advertenties blijven 60 dagen online."],
             "de": ["Anzeige 7 Tage hervorheben: 2,50 €", "Anzeigen bleiben 60 Tage online."],
             "fr": ["Mettez votre annonce en avant pendant 7 jours : 2,50 €", "Les annonces restent en ligne 60 jours."],
             "es": ["Destaca tu anuncio 7 días: 2,50 €", "Los anuncios permanecen 60 días en línea."]}[lang]
    return page(rng, lang, rules, details, extra), noul(q), truth, margin


ZONES_OK = [dict(en="north meadow", nl="noordelijke weide", de="Nordwiese", fr="prairie nord", es="pradera norte"),
            dict(en="sports field", nl="sportveld", de="Sportplatz", fr="terrain de sport", es="campo de deportes"),
            dict(en="hilltop", nl="heuveltop", de="Hügelkuppe", fr="sommet de la colline", es="cima de la colina")]
ZONES_NO = [dict(en="playground", nl="speeltuin", de="Spielplatz", fr="aire de jeux", es="zona de juegos"),
            dict(en="lake shore", nl="oever van het meer", de="Seeufer", fr="rive du lac", es="orilla del lago"),
            dict(en="car park", nl="parkeerterrein", de="Parkplatz", fr="parking", es="aparcamiento")]
DRONES = ["Mini 4", "Air 3", "Skylark X", "Falcon 2", "Pixy", "Hover One"]


def drone_flight_allowed(rng, lang):
    P = Plan(rng, 2, waivable=0)
    maxg, maxalt = rng.choice([250, 500, 900]), rng.choice([30, 50, 60, 120])
    grams, height = int(amt(rng, maxg, P.passes[0], P.lv[0])), int(amt(rng, maxalt, P.passes[1], P.lv[1]))
    zone = rng.choice(ZONES_NO if P.block else ZONES_OK)[lang]
    model, permit = rng.choice(DRONES), (YES if P.waiver else NO)[lang]
    rules = {
        "en": [rng.choice(["Drone rules in the park", "Flying drones in the park"]),
               f"Drones may weigh at most {maxg} g. This weight limit does not apply to pilots with a park permit.",
               f"Maximum flying height: {maxalt} m.", "No flying over the playground, the lake shore or the car park."],
        "nl": [rng.choice(["Droneregels in het park", "Vliegen met drones in het park"]),
               f"Drones mogen maximaal {maxg} g wegen. Deze gewichtslimiet geldt niet voor piloten met een parkvergunning.",
               f"Maximale vlieghoogte: {maxalt} m.", "Niet vliegen boven de speeltuin, de oever van het meer of het parkeerterrein."],
        "de": [rng.choice(["Drohnenregeln im Park", "Drohnenflüge im Park"]),
               f"Drohnen dürfen höchstens {maxg} g wiegen. Diese Gewichtsgrenze gilt nicht für Piloten mit Parkgenehmigung.",
               f"Maximale Flughöhe: {maxalt} m.", "Kein Flug über dem Spielplatz, dem Seeufer oder dem Parkplatz."],
        "fr": [rng.choice(["Règles pour les drones dans le parc", "Drones dans le parc"]),
               f"Les drones ne doivent pas peser plus de {maxg} g. Cette limite de poids ne s'applique pas aux pilotes titulaires d'un permis du parc.",
               f"Hauteur de vol maximale : {maxalt} m.", "Interdiction de survoler l'aire de jeux, la rive du lac et le parking."],
        "es": [rng.choice(["Normas para drones en el parque", "Drones en el parque"]),
               f"Los drones pueden pesar como máximo {maxg} g. Este límite de peso no se aplica a los pilotos con permiso del parque.",
               f"Altura máxima de vuelo: {maxalt} m.", "Prohibido volar sobre la zona de juegos, la orilla del lago y el aparcamiento."]}[lang]
    details = {
        "en": ["My flight plan", f"Drone: {model}, {grams} g", f"Planned height: {height} m", f"Area: {zone}", f"Park permit: {permit}"],
        "nl": ["Mijn vliegplan", f"Drone: {model}, {grams} g", f"Geplande hoogte: {height} m", f"Gebied: {zone}", f"Parkvergunning: {permit}"],
        "de": ["Mein Flugplan", f"Drohne: {model}, {grams} g", f"Geplante Höhe: {height} m", f"Bereich: {zone}", f"Parkgenehmigung: {permit}"],
        "fr": ["Mon plan de vol", f"Drone : {model}, {grams} g", f"Hauteur prévue : {height} m", f"Zone : {zone}", f"Permis du parc : {permit}"],
        "es": ["Mi plan de vuelo", f"Dron: {model}, {grams} g", f"Altura prevista: {height} m", f"Zona: {zone}", f"Permiso del parque: {permit}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Am I allowed to make this flight?", "Does my planned flight comply with the park rules?", "May I fly my drone as planned?"],
        "nl": ["Mag ik deze vlucht maken?", "Voldoet mijn geplande vlucht aan de parkregels?"],
        "de": ["Darf ich diesen Flug durchführen?", "Entspricht mein geplanter Flug den Parkregeln?"],
        "fr": ["Ai-je le droit d'effectuer ce vol ?", "Mon vol prévu respecte-t-il les règles du parc ?"],
        "es": ["¿Puedo hacer este vuelo?", "¿Mi vuelo previsto cumple las normas del parque?"]})
    truth, margin = verdict(P, [grams <= maxg, height <= maxalt], [rel_grade(grams, maxg), rel_grade(height, maxalt)])
    extra = {"en": ["Dogs must be kept on a lead.", "Park café: coffee €2.80"], "nl": ["Honden moeten aangelijnd zijn.", "Parkcafé: koffie € 2,80"],
             "de": ["Hunde sind an der Leine zu führen.", "Parkcafé: Kaffee 2,80 €"], "fr": ["Les chiens doivent être tenus en laisse.", "Café du parc : café 2,80 €"],
             "es": ["Los perros deben ir con correa.", "Cafetería del parque: café 2,80 €"]}[lang]
    return page(rng, lang, rules, details, extra), noul(q), truth, margin


# -------------------------------------------------------------------------------------------- held-out families

def blood_donation_allowed(rng, lang):
    P = Plan(rng, 2, waivable=0)
    t, pd, name = today(rng), page_dates(rng, lang), rng.choice(NAMES)
    wait = rng.choice([56, 84])
    g0 = dist(rng, P.lv[0], P.passes[0], "days", least=1)  # off the exact day: "between" can be counted two ways
    since = wait + g0 if P.passes[0] else wait - g0
    last = t - dt.timedelta(days=since)
    kg = amt(rng, 50, not P.passes[1], P.lv[1], 0.5)
    if not P.passes[1]:
        kg = max(kg, 40.0)
    plasma, tattoo = P.waiver, (YES if P.block else NO)[lang]
    rules = {
        "en": [rng.choice(["Donation rules", "Before you donate"]),
               f"There must be at least {wait} days between two whole-blood donations. This waiting period does not apply to plasma donations.",
               "You must weigh at least 50 kg.", "No donation within 4 months after a tattoo or piercing."],
        "nl": [rng.choice(["Regels voor doneren", "Voordat je doneert"]),
               f"Tussen twee volbloeddonaties moeten minstens {wait} dagen zitten. Deze wachttijd geldt niet voor plasmadonaties.",
               "Je moet minstens 50 kg wegen.", "Niet doneren binnen 4 maanden na een tatoeage of piercing."],
        "de": [rng.choice(["Regeln für die Spende", "Vor der Spende"]),
               f"Zwischen zwei Vollblutspenden müssen mindestens {wait} Tage liegen. Diese Wartezeit gilt nicht für Plasmaspenden.",
               "Das Körpergewicht muss mindestens 50 kg betragen.", "Keine Spende innerhalb von 4 Monaten nach einer Tätowierung oder einem Piercing."],
        "fr": [rng.choice(["Règles du don", "Avant de donner"]),
               f"Il faut au moins {wait} jours entre deux dons de sang total. Ce délai ne s'applique pas aux dons de plasma.",
               "Il faut peser au moins 50 kg.", "Pas de don dans les 4 mois qui suivent un tatouage ou un piercing."],
        "es": [rng.choice(["Normas de donación", "Antes de donar"]),
               f"Deben pasar al menos {wait} días entre dos donaciones de sangre total. Esta espera no se aplica a las donaciones de plasma.",
               "Hay que pesar al menos 50 kg.", "No se puede donar en los 4 meses siguientes a un tatuaje o piercing."]}[lang]
    kind = ({"en": "plasma", "nl": "plasma", "de": "Plasma", "fr": "plasma", "es": "plasma"} if plasma else
            {"en": "whole blood", "nl": "volbloed", "de": "Vollblut", "fr": "sang total", "es": "sangre total"})[lang]
    w = num(kg, lang, 1)
    details = {
        "en": [f"Donor: {name}", f"Last whole-blood donation: {pd(last)}", f"Weight: {w} kg", f"Donation type today: {kind}",
               f"Tattoo or piercing in the last 4 months: {tattoo}", f"{TODAY[lang]}: {pd(t)}"],
        "nl": [f"Donor: {name}", f"Laatste volbloeddonatie: {pd(last)}", f"Gewicht: {w} kg", f"Soort donatie vandaag: {kind}",
               f"Tatoeage of piercing in de afgelopen 4 maanden: {tattoo}", f"{TODAY[lang]}: {pd(t)}"],
        "de": [f"Spender/in: {name}", f"Letzte Vollblutspende: {pd(last)}", f"Gewicht: {w} kg", f"Art der Spende heute: {kind}",
               f"Tätowierung oder Piercing in den letzten 4 Monaten: {tattoo}", f"{TODAY[lang]}: {pd(t)}"],
        "fr": [f"Donneur : {name}", f"Dernier don de sang total : {pd(last)}", f"Poids : {w} kg", f"Type de don aujourd'hui : {kind}",
               f"Tatouage ou piercing au cours des 4 derniers mois : {tattoo}", f"{TODAY[lang]} : {pd(t)}"],
        "es": [f"Donante: {name}", f"Última donación de sangre total: {pd(last)}", f"Peso: {w} kg", f"Tipo de donación hoy: {kind}",
               f"Tatuaje o piercing en los últimos 4 meses: {tattoo}", f"{TODAY[lang]}: {pd(t)}"]}[lang]
    q = ask(rng, lang, {
        "en": ["May {n} donate today?", "Do the rules allow {n} to give this donation today?"],
        "nl": ["Mag {n} vandaag doneren?", "Mag {n} volgens de regels vandaag deze donatie doen?"],
        "de": ["Darf {n} heute spenden?", "Darf {n} nach den Regeln heute diese Spende leisten?"],
        "fr": ["Est-ce que {n} peut donner aujourd'hui ?", "Les règles permettent-elles à {n} de faire ce don aujourd'hui ?"],
        "es": ["¿Puede {n} donar hoy?", "¿Las normas permiten a {n} hacer esta donación hoy?"]}, n=name)
    truth, margin = verdict(P, [since >= wait, kg >= 50], [grade_by(g0, "days"), rel_grade(kg, 50)])
    return page(rng, lang, rules, details), noul(q), truth, margin


PHOTO_CATS = [dict(en="Nature", nl="Natuur", de="Natur", fr="Nature", es="Naturaleza"), dict(en="Street", nl="Straat", de="Straße", fr="Rue", es="Calle"),
              dict(en="Portrait", nl="Portret", de="Porträt", fr="Portrait", es="Retrato")]
ARCHIVE = dict(en="Archive", nl="Archief", de="Archiv", fr="Archives", es="Archivo")


def photo_contest_entry_valid(rng, lang):
    P = Plan(rng, 2, waivable=1)
    pd = page_dates(rng, lang)
    minpx = rng.choice([2000, 3000, 4000])
    px = int(amt(rng, minpx, not P.passes[0], P.lv[0]))
    since = today(rng)
    g1 = dist(rng, P.lv[1], P.passes[1], "days")
    taken = since + dt.timedelta(days=g1 if P.passes[1] else -g1)
    cat, won = (ARCHIVE if P.waiver else rng.choice(PHOTO_CATS))[lang], (YES if P.block else NO)[lang]
    short = round(px * rng.choice([2 / 3, 3 / 4, 9 / 16]))
    size = f"{px} × {short} px" if rng.random() < 0.7 else f"{short} × {px} px"
    file = f"IMG_{rng.randint(1000, 9999)}.jpg"
    rules = {
        "en": [rng.choice(["Photo contest: entry rules", "Entry rules"]), f"The long edge of the image must be at least {minpx} pixels.",
               f"The photo must have been taken on or after {pd(since)}. In the Archive category, photos of any date are welcome.",
               "Photos that have already won a prize in another contest are excluded."],
        "nl": [rng.choice(["Fotowedstrijd: deelnameregels", "Deelnameregels"]), f"De lange zijde van de afbeelding moet minstens {minpx} pixels zijn.",
               f"De foto moet op of na {pd(since)} zijn gemaakt. In de categorie Archief zijn foto's van elke datum welkom.",
               "Foto's die al een prijs hebben gewonnen in een andere wedstrijd, zijn uitgesloten."],
        "de": [rng.choice(["Fotowettbewerb: Teilnahmeregeln", "Teilnahmeregeln"]), f"Die lange Bildkante muss mindestens {minpx} Pixel haben.",
               f"Das Foto muss am oder nach dem {pd(since)} aufgenommen worden sein. In der Kategorie Archiv sind Fotos jedes Datums willkommen.",
               "Fotos, die bereits in einem anderen Wettbewerb einen Preis gewonnen haben, sind ausgeschlossen."],
        "fr": [rng.choice(["Concours photo : règlement", "Règlement du concours"]), f"Le grand côté de l'image doit mesurer au moins {minpx} pixels.",
               f"La photo doit avoir été prise le {pd(since)} ou après. Dans la catégorie Archives, les photos de toute date sont acceptées.",
               "Les photos déjà primées dans un autre concours sont exclues."],
        "es": [rng.choice(["Concurso de fotografía: bases", "Bases del concurso"]), f"El lado largo de la imagen debe medir al menos {minpx} píxeles.",
               f"La foto debe haberse tomado el {pd(since)} o después. En la categoría Archivo se aceptan fotos de cualquier fecha.",
               "Quedan excluidas las fotos ya premiadas en otro concurso."]}[lang]
    details = {
        "en": ["My entry", f"File: {file}, {size}", f"Taken on: {pd(taken)}", f"Category: {cat}", f"Already won a prize in another contest: {won}"],
        "nl": ["Mijn inzending", f"Bestand: {file}, {size}", f"Gemaakt op: {pd(taken)}", f"Categorie: {cat}", f"Al een prijs gewonnen in een andere wedstrijd: {won}"],
        "de": ["Meine Einreichung", f"Datei: {file}, {size}", f"Aufgenommen am: {pd(taken)}", f"Kategorie: {cat}", f"Bereits in einem anderen Wettbewerb prämiert: {won}"],
        "fr": ["Ma participation", f"Fichier : {file}, {size}", f"Prise le : {pd(taken)}", f"Catégorie : {cat}", f"Déjà primée dans un autre concours : {won}"],
        "es": ["Mi participación", f"Archivo: {file}, {size}", f"Tomada el: {pd(taken)}", f"Categoría: {cat}", f"Ya premiada en otro concurso: {won}"]}[lang]
    q = ask(rng, lang, {
        "en": ["Is this photo a valid entry?", "Can I enter this photo in the contest?"],
        "nl": ["Is deze foto een geldige inzending?", "Kan ik met deze foto meedoen aan de wedstrijd?"],
        "de": ["Ist dieses Foto eine gültige Einreichung?", "Kann ich mit diesem Foto am Wettbewerb teilnehmen?"],
        "fr": ["Cette photo est-elle une participation valable ?", "Puis-je présenter cette photo au concours ?"],
        "es": ["¿Esta foto es una participación válida?", "¿Puedo presentar esta foto al concurso?"]})
    truth, margin = verdict(P, [px >= minpx, taken >= since], [rel_grade(px, minpx), grade_by(g1, "days")])
    extra = {"en": ["First prize: a €500 voucher"], "nl": ["Hoofdprijs: een cadeaubon van € 500"],
             "de": ["Hauptpreis: ein Gutschein über 500 €"], "fr": ["Premier prix : un bon d'achat de 500 €"],
             "es": ["Primer premio: un vale de 500 €"]}[lang]
    return page(rng, lang, rules, details, extra), noul(q), truth, margin


FIRES = [dict(en="a wood campfire", nl="een kampvuur op hout", de="ein Lagerfeuer (Holz)", fr="un feu de camp au bois", es="una hoguera de leña"),
         dict(en="a charcoal barbecue", nl="een houtskoolbarbecue", de="einen Holzkohlegrill", fr="un barbecue au charbon de bois", es="una barbacoa de carbón")]
GAS = dict(en="a gas barbecue", nl="een gasbarbecue", de="einen Gasgrill", fr="un barbecue à gaz", es="una barbacoa de gas")


def campsite_fire_allowed_now(rng, lang):
    P = Plan(rng, 2, waivable=0)
    cs = clock_style(rng, lang)
    maxw = rng.choice([20, 25, 30, 40])
    wind = int(amt(rng, maxw, P.passes[0], P.lv[0]))
    lo, hi = rng.choice([16, 17, 18]) * 60, rng.choice([22, 23]) * 60
    g1 = dist(rng, P.lv[1], P.passes[1], "min", least=1)  # never exactly on the first or last minute
    if side(rng):
        now = lo + g1 if P.passes[1] else lo - g1
    else:
        now = hi - g1 if P.passes[1] else hi + g1
    if now >= 24 * 60 or now < 6 * 60:
        raise Skip
    limit = rng.choice([3, 4])
    danger = rng.randint(limit, 5) if P.block else rng.randint(1, limit - 1)
    kind = (GAS if P.waiver else rng.choice(FIRES))[lang]
    c = lambda m: clock(m, cs)
    rules = {
        "en": [rng.choice(["Fires and barbecues", "Campsite rules: fires and barbecues"]), f"Open fires and barbecues are allowed only between {c(lo)} and {c(hi)}.",
               f"Not when the wind is stronger than {maxw} km/h. This wind limit does not apply to gas barbecues.",
               f"At fire danger level {limit} or higher, all fires and barbecues are banned, gas included."],
        "nl": [rng.choice(["Vuur en barbecues", "Campingregels: vuur en barbecues"]), f"Open vuur en barbecues zijn alleen toegestaan tussen {c(lo)} en {c(hi)} uur.",
               f"Niet bij een windsnelheid boven {maxw} km/u. Deze windlimiet geldt niet voor gasbarbecues.",
               f"Bij brandgevaarniveau {limit} of hoger zijn alle vuren en barbecues verboden, ook op gas."],
        "de": [rng.choice(["Feuer und Grillen", "Platzordnung: Feuer und Grillen"]), f"Offenes Feuer und Grills sind nur zwischen {c(lo)} und {c(hi)} Uhr erlaubt.",
               f"Nicht bei Wind über {maxw} km/h. Diese Windgrenze gilt nicht für Gasgrills.",
               f"Ab Waldbrandstufe {limit} sind alle Feuer und Grills verboten, auch Gasgrills."],
        "fr": [rng.choice(["Feux et barbecues", "Règlement du camping : feux et barbecues"]), f"Les feux et les barbecues ne sont autorisés qu'entre {c(lo)} et {c(hi)}.",
               f"Interdits lorsque le vent dépasse {maxw} km/h. Cette limite de vent ne s'applique pas aux barbecues à gaz.",
               f"À partir du niveau de risque d'incendie {limit}, tous les feux et barbecues sont interdits, y compris au gaz."],
        "es": [rng.choice(["Fuego y barbacoas", "Normas del camping: fuego y barbacoas"]), f"Las hogueras y barbacoas solo se permiten entre las {c(lo)} y las {c(hi)}.",
               f"No se permiten con viento superior a {maxw} km/h. Este límite de viento no se aplica a las barbacoas de gas.",
               f"Con nivel de riesgo de incendio {limit} o superior se prohíben todas las hogueras y barbacoas, también las de gas."]}[lang]
    details = {
        "en": ["Right now at the campsite", f"{NOW[lang]}: {c(now)}", f"Wind: {wind} km/h", f"Fire danger level today: {danger}", f"I want to use: {kind}"],
        "nl": ["Nu op de camping", f"{NOW[lang]}: {c(now)}", f"Wind: {wind} km/u", f"Brandgevaarniveau vandaag: {danger}", f"Ik wil gebruiken: {kind}"],
        "de": ["Aktuell auf dem Campingplatz", f"{NOW[lang]}: {c(now)}", f"Wind: {wind} km/h", f"Waldbrandstufe heute: {danger}", f"Ich möchte benutzen: {kind}"],
        "fr": ["En ce moment au camping", f"{NOW[lang]} : {c(now)}", f"Vent : {wind} km/h", f"Niveau de risque d'incendie aujourd'hui : {danger}", f"Je veux utiliser : {kind}"],
        "es": ["Ahora mismo en el camping", f"{NOW[lang]}: {c(now)}", f"Viento: {wind} km/h", f"Nivel de riesgo de incendio hoy: {danger}", f"Quiero usar: {kind}"]}[lang]
    q = ask(rng, lang, {
        "en": ["May I light my fire or barbecue right now?", "Is what I want to use allowed right now under the campsite rules?"],
        "nl": ["Mag ik mijn vuur of barbecue nu aansteken?", "Is wat ik wil gebruiken nu toegestaan volgens de campingregels?"],
        "de": ["Darf ich mein Feuer oder meinen Grill jetzt anzünden?", "Ist das, was ich benutzen möchte, nach der Platzordnung jetzt erlaubt?"],
        "fr": ["Puis-je allumer mon feu ou mon barbecue maintenant ?", "Ce que je veux utiliser est-il autorisé en ce moment par le règlement du camping ?"],
        "es": ["¿Puedo encender mi hoguera o barbacoa ahora?", "¿Lo que quiero usar está permitido ahora según las normas del camping?"]})
    truth, margin = verdict(P, [wind <= maxw, lo <= now <= hi], [rel_grade(wind, maxw), grade_by(g1, "min")])
    extra = {"en": ["Firewood: €6.00 per bag at reception", "Quiet on the campsite from midnight."], "nl": ["Brandhout: € 6,00 per zak bij de receptie", "Vanaf middernacht stilte op de camping."],
             "de": ["Brennholz: 6,00 € pro Sack an der Rezeption", "Ab Mitternacht Nachtruhe auf dem Platz."], "fr": ["Bois de chauffage : 6,00 € le sac à l'accueil", "Silence sur le camping à partir de minuit."],
             "es": ["Leña: 6,00 € el saco en recepción", "Silencio en el camping a partir de medianoche."]}[lang]
    return page(rng, lang, rules, details, extra), noul(q), truth, margin


DOG = dict(en="dog", nl="hond", de="Hund", fr="chien", es="perro")
PETS_NO = [dict(en="snake", nl="slang", de="Schlange", fr="serpent", es="serpiente"), dict(en="mini pig", nl="minivarken", de="Minischwein", fr="cochon nain", es="cerdo enano"),
           dict(en="goat", nl="geit", de="Ziege", fr="chèvre", es="cabra")]


def tenant_pet_allowed(rng, lang):
    P = Plan(rng, 2, waivable=1)
    if P.block:
        P.waiver = False  # an assistance dog is a dog
    name = rng.choice(NAMES)
    maxp, maxkg = rng.choice([2, 3]), rng.choice([10, 15, 20, 25])
    pets, g0 = under_cap(rng, maxp, P.passes[0], P.lv[0])
    kg = amt(rng, maxkg, P.passes[1], P.lv[1], 0.5)
    animal, helper = (rng.choice(PETS_NO) if P.block else DOG)[lang], (YES if P.waiver else NO)[lang]
    w = num(kg, lang, 1)
    rules = {
        "en": [rng.choice(["House rules: pets", "Pets in the building"]), f"At most {maxp} pets per apartment.",
               f"A pet may weigh at most {maxkg} kg. This weight limit does not apply to registered assistance dogs.", "Snakes, pigs and goats are not allowed."],
        "nl": [rng.choice(["Huisregels: huisdieren", "Huisdieren in het gebouw"]), f"Maximaal {maxp} huisdieren per appartement.",
               f"Een huisdier mag maximaal {maxkg} kg wegen. Deze gewichtslimiet geldt niet voor geregistreerde assistentiehonden.",
               "Slangen, varkens en geiten zijn niet toegestaan."],
        "de": [rng.choice(["Hausordnung: Haustiere", "Haustiere im Haus"]), f"Höchstens {maxp} Haustiere pro Wohnung.",
               f"Ein Haustier darf höchstens {maxkg} kg wiegen. Diese Gewichtsgrenze gilt nicht für registrierte Assistenzhunde.",
               "Schlangen, Schweine und Ziegen sind nicht erlaubt."],
        "fr": [rng.choice(["Règlement intérieur : animaux", "Animaux dans l'immeuble"]), f"{maxp} animaux au maximum par appartement.",
               f"Un animal ne doit pas peser plus de {maxkg} kg. Cette limite de poids ne s'applique pas aux chiens d'assistance enregistrés.",
               "Les serpents, les cochons et les chèvres ne sont pas autorisés."],
        "es": [rng.choice(["Normas de la comunidad: mascotas", "Mascotas en el edificio"]), f"Máximo {maxp} mascotas por vivienda.",
               f"Una mascota puede pesar como máximo {maxkg} kg. Este límite de peso no se aplica a los perros de asistencia registrados.",
               "No se admiten serpientes, cerdos ni cabras."]}[lang]
    details = {
        "en": [f"Tenant: {name}", f"Pets already in the apartment: {pets}", f"New pet: {animal}, {w} kg", f"Registered assistance dog: {helper}"],
        "nl": [f"Huurder: {name}", f"Huisdieren al in het appartement: {pets}", f"Nieuw huisdier: {animal}, {w} kg", f"Geregistreerde assistentiehond: {helper}"],
        "de": [f"Mieter/in: {name}", f"Bereits in der Wohnung lebende Haustiere: {pets}", f"Neues Haustier: {animal}, {w} kg", f"Registrierter Assistenzhund: {helper}"],
        "fr": [f"Locataire : {name}", f"Animaux déjà présents dans l'appartement : {pets}", f"Nouvel animal : {animal}, {w} kg", f"Chien d'assistance enregistré : {helper}"],
        "es": [f"Inquilino/a: {name}", f"Mascotas que ya viven en la vivienda: {pets}", f"Nueva mascota: {animal}, {w} kg", f"Perro de asistencia registrado: {helper}"]}[lang]
    q = ask(rng, lang, {
        "en": ["May {n} keep this new pet in the apartment?", "Do the house rules allow {n} to add this pet?"],
        "nl": ["Mag {n} dit nieuwe huisdier in het appartement houden?", "Staan de huisregels toe dat {n} dit huisdier erbij neemt?"],
        "de": ["Darf {n} dieses neue Haustier in der Wohnung halten?", "Erlaubt die Hausordnung {n}, dieses Haustier zusätzlich aufzunehmen?"],
        "fr": ["Est-ce que {n} peut garder ce nouvel animal dans l'appartement ?", "Le règlement autorise-t-il {n} à accueillir cet animal en plus ?"],
        "es": ["¿Puede {n} tener esta nueva mascota en la vivienda?", "¿Las normas permiten a {n} añadir esta mascota?"]}, n=name)
    truth, margin = verdict(P, [pets < maxp, kg <= maxkg], [g0, rel_grade(kg, maxkg)])
    extra = {"en": ["Pet deposit: €150.00, refunded when you move out"], "nl": ["Borg voor huisdieren: € 150,00, terug bij vertrek"],
             "de": ["Haustierkaution: 150,00 €, Rückzahlung beim Auszug"], "fr": ["Dépôt de garantie pour animaux : 150,00 €, restitué au départ"],
             "es": ["Fianza por mascota: 150,00 €, se devuelve al dejar la vivienda"]}[lang]
    return page(rng, lang, rules, details, extra), noul(q), truth, margin


TRAINED = [library_renewal_allowed, gym_class_booking_allowed, league_playoff_eligible, grant_application_eligible, noisy_work_allowed,
           bulky_waste_pickup_accepted, meeting_room_booking_valid, volunteer_shift_signup_allowed, lab_access_allowed, licence_seat_assignable,
           classifieds_listing_accepted, drone_flight_allowed]
HELD_OUT = [blood_donation_allowed, photo_contest_entry_valid, campsite_fire_allowed_now, tenant_pet_allowed]


def even(fn, slack: int = 3):
    """Redraw so that neither answer ever gets more than `slack` rows ahead: each family is half yes, half no at any
    size, without changing what a yes or a no row looks like."""
    lead = 0

    def draw(rng, lang):
        nonlocal lead
        while True:
            row = fn(rng, lang)
            step = 1 if row[2] == "true" else -1
            if abs(lead + step) <= slack:
                lead += step
                return row
    draw.__name__ = fn.__name__
    return draw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path)
    parser.add_argument("--per-family", type=int, default=1000)
    parser.add_argument("--test-per-family", type=int, default=40)
    parser.add_argument("--held-out-per-family", type=int, default=120)
    args = parser.parse_args()
    # Each family draws its test rows and its training rows from streams of its own, so the test set stays the same
    # whatever the training size (and teachers' labels of it stay valid).
    train, test = [], []
    for fn in TRAINED:
        rows = bn.rows_for(even(safe(fn)), args.test_per_family, random.Random(f"2028-test-{fn.__name__}"), "family-seen")
        texts = {r["request"]["state"] for r in rows}
        test += rows
        train += [r for r in bn.rows_for(even(safe(fn)), args.per_family, random.Random(f"2028-train-{fn.__name__}"), "train") if r["request"]["state"] not in texts]
    for fn in HELD_OUT:
        test += bn.rows_for(even(safe(fn)), args.held_out_per_family, random.Random(f"2028-test-{fn.__name__}"), "family-unseen")
    for r in train + test:
        r["meta"]["source"] = "conditions"
    for path, rows in ((args.out, train), (args.test, test)):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
        if args.truth_dir:
            (args.truth_dir / f"{path.stem}.truth.jsonl").write_text(
                "".join(json.dumps(bn.truth_row(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    for name, rows in (("train", train), ("test", test)):
        c = Counter((r["meta"]["family"], r["meta"]["truth"]) for r in rows)
        m = Counter((r["meta"]["family"], r["meta"]["margin"]) for r in rows)
        print(f"{name}: {len(rows)} rows, margins {dict(Counter(r['meta']['margin'] for r in rows))}")
        for fam in sorted({f for f, _ in c}):
            parts = {t: n for (f, t), n in sorted(c.items()) if f == fam}
            margins = {k: n for (f, k), n in sorted(m.items()) if f == fam}
            print(f"  {fam:<34} {parts}  {margins}")


if __name__ == "__main__":
    main()
