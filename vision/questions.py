# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""System One questions about pictures, worded the ways a caller would word them, with the answer known.

Whoever asks (Atelier, the companion through a tool, a pack) writes the options itself, so the same thing arrives in
many shapes. For a photo of a dog:

    choice   options as bare names ("dog", "cat"), with their article ("a dog"), as sentences ("The photo shows a
             dog."), or with the name as the option's key and no description at all
    noul     with what a yes and a no look like ("The photo shows a dog." / "The photo shows something else."), with
             only the yes, or as the bare question ("Is there a dog in this photo?")

in English, Dutch, German, French or Spanish, whatever the picture's own language. Every list of wordings ends with
one that is never trained on (UNSEEN): held-out questions are also asked that way, to see whether a wording nobody
showed it changes the answer.

A family is one kind of picture with its own things to name: photos (vision/classes.py), albums, buttons
(vision/icons.py), documents and pages (vision/synth.py). The templates were written for this project by Claude
(Anthropic), CC BY 4.0.
"""

from __future__ import annotations

import random
import re

LANGS = ["en", "nl", "de", "fr", "es"]
LANG_WEIGHTS = [0.4, 0.2, 0.15, 0.13, 0.12]  # as step/synth.py
UNSEEN = -1  # the last wording of every list is held out

# {x}: the thing with its article ("a dog"); {name}: without ("dog"). German phrases are nominative, so the German
# templates keep the thing in a place where the nominative is right.
SAY = {
    "photo": {
        "en": ["The photo shows {x}.", "A photo of {x}.", "In the picture: {x}.", "You can see {x} here."],
        "nl": ["De foto toont {x}.", "Een foto van {x}.", "Op de foto: {x}.", "Je ziet hier {x}."],
        "de": ["Auf dem Foto: {x}.", "Zu sehen: {x}.", "Das Bild zeigt: {x}.", "Hier ist zu sehen: {x}."],
        "fr": ["La photo montre {x}.", "Sur la photo : {x}.", "On voit {x}.", "Ici, il y a {x}."],
        "es": ["La foto muestra {x}.", "En la foto: {x}.", "Se ve {x}.", "Aquí aparece {x}."],
    },
    "album": {
        "en": ["{name}", "A photo for the album “{name}”.", "It belongs with: {name}.", "Photos like this go under “{name}”."],
        "nl": ["{name}", "Een foto voor het album “{name}”.", "Het hoort bij: {name}.", "Zulke foto's horen onder “{name}”."],
        "de": ["{name}", "Ein Foto für das Album „{name}“.", "Es gehört zu: {name}.", "Solche Fotos kommen unter „{name}“."],
        "fr": ["{name}", "Une photo pour l'album « {name} ».", "Elle va avec : {name}.", "Ce genre de photo se range sous « {name} »."],
        "es": ["{name}", "Una foto para el álbum «{name}».", "Va con: {name}.", "Fotos así van en «{name}»."],
    },
    "control": {
        "en": ["A button for: {name}.", "An icon that means {name}.", "The {name} button.", "This control is for {name}."],
        "nl": ["Een knop voor: {name}.", "Een icoon dat {name} betekent.", "De knop {name}.", "Deze knop is voor {name}."],
        "de": ["Eine Schaltfläche für: {name}.", "Ein Symbol, das {name} bedeutet.", "Die Schaltfläche {name}.", "Dieses Bedienelement ist für {name}."],
        "fr": ["Un bouton pour : {name}.", "Une icône qui signifie {name}.", "Le bouton {name}.", "Cette commande sert à : {name}."],
        "es": ["Un botón para: {name}.", "Un icono que significa {name}.", "El botón de {name}.", "Este control sirve para {name}."],
    },
    "document": {
        "en": ["The picture shows {x}.", "This is {x}.", "A picture of {x}.", "What you see here is {x}."],
        "nl": ["De afbeelding toont {x}.", "Dit is {x}.", "Een afbeelding van {x}.", "Wat je hier ziet is {x}."],
        "de": ["Das Bild zeigt: {x}.", "Das ist {x}.", "Zu sehen: {x}.", "Was hier zu sehen ist: {x}."],
        "fr": ["L'image montre {x}.", "C'est {x}.", "Sur l'image : {x}.", "Ce que l'on voit ici : {x}."],
        "es": ["La imagen muestra {x}.", "Esto es {x}.", "En la imagen: {x}.", "Lo que se ve aquí es {x}."],
    },
}
# What a "no" looks like, for a noul's "false".
OTHER = {
    "photo": {
        "en": ["The photo shows something else.", "No {name} in the photo.", "Anything else.", "There is no {name} to be seen."],
        "nl": ["De foto toont iets anders.", "Geen {name} op de foto.", "Iets anders.", "Er is geen {name} te zien."],
        "de": ["Das Foto zeigt etwas anderes.", "Nicht auf dem Foto: {name}.", "Etwas anderes.", "Zu sehen ist etwas anderes als: {name}."],
        "fr": ["La photo montre autre chose.", "Pas de {name} sur la photo.", "Autre chose.", "On ne voit pas de {name}."],
        "es": ["La foto muestra otra cosa.", "No hay {name} en la foto.", "Otra cosa.", "No se ve {name}."],
    },
    "control": {
        "en": ["The button does something else.", "Not a button for {name}.", "Anything else.", "A control for something other than {name}."],
        "nl": ["De knop doet iets anders.", "Geen knop voor {name}.", "Iets anders.", "Een knop voor iets anders dan {name}."],
        "de": ["Die Schaltfläche macht etwas anderes.", "Keine Schaltfläche für {name}.", "Etwas anderes.", "Ein Bedienelement für etwas anderes als {name}."],
        "fr": ["Le bouton fait autre chose.", "Pas un bouton pour {name}.", "Autre chose.", "Une commande pour autre chose que {name}."],
        "es": ["El botón hace otra cosa.", "No es un botón para {name}.", "Otra cosa.", "Un control para algo distinto de {name}."],
    },
    "document": {
        "en": ["The picture shows something else.", "This is not {x}.", "Anything else.", "Something other than {x}."],
        "nl": ["De afbeelding toont iets anders.", "Dit is niet {x}.", "Iets anders.", "Iets anders dan {x}."],
        "de": ["Das Bild zeigt etwas anderes.", "Das ist nicht {x}.", "Etwas anderes.", "Etwas anderes als {x}."],
        "fr": ["L'image montre autre chose.", "Ce n'est pas {x}.", "Autre chose.", "Autre chose que {x}."],
        "es": ["La imagen muestra otra cosa.", "Esto no es {x}.", "Otra cosa.", "Algo distinto de {x}."],
    },
}
OTHER["album"] = OTHER["photo"]
# The bare question, for a noul with no criteria.
ASK = {
    "photo": {
        "en": ["Is there {x} in this photo?", "Does this photo show {x}?", "Can you see {x} in this picture?", "Is {x} visible here?"],
        "nl": ["Staat er {x} op deze foto?", "Toont deze foto {x}?", "Zie je {x} op deze afbeelding?", "Is hier {x} te zien?"],
        "de": ["Ist auf diesem Foto {x} zu sehen?", "Ist {x} auf dem Bild?", "Sieht man hier {x}?", "Ist hier {x} abgebildet?"],
        "fr": ["Y a-t-il {x} sur cette photo ?", "Voit-on {x} sur cette photo ?", "Cette photo montre-t-elle {x} ?", "Est-ce que {x} apparaît ici ?"],
        "es": ["¿Hay {x} en esta foto?", "¿Se ve {x} en esta foto?", "¿Esta foto muestra {x}?", "¿Aparece {x} aquí?"],
    },
    "album": {
        "en": ["Does this photo belong in the album “{name}”?", "Is this a photo for “{name}”?", "Would you file this under “{name}”?", "Does “{name}” fit this photo?"],
        "nl": ["Hoort deze foto in het album “{name}”?", "Is dit een foto voor “{name}”?", "Zou je dit onder “{name}” zetten?", "Past “{name}” bij deze foto?"],
        "de": ["Gehört dieses Foto ins Album „{name}“?", "Ist das ein Foto für „{name}“?", "Würdest du das unter „{name}“ ablegen?", "Passt „{name}“ zu diesem Foto?"],
        "fr": ["Cette photo va-t-elle dans l'album « {name} » ?", "Est-ce une photo pour « {name} » ?", "La classeriez-vous sous « {name} » ?", "« {name} » convient-il à cette photo ?"],
        "es": ["¿Esta foto va en el álbum «{name}»?", "¿Es una foto para «{name}»?", "¿La pondrías en «{name}»?", "¿Encaja «{name}» con esta foto?"],
    },
    "control": {
        "en": ["Is this the {name} button?", "Is this a button for {name}?", "Does this icon mean {name}?", "Would you click this for {name}?"],
        "nl": ["Is dit de knop {name}?", "Is dit een knop voor {name}?", "Betekent dit icoon {name}?", "Zou je hierop klikken voor {name}?"],
        "de": ["Ist das die Schaltfläche {name}?", "Ist das eine Schaltfläche für {name}?", "Bedeutet dieses Symbol {name}?", "Würdest du hier für {name} klicken?"],
        "fr": ["Est-ce le bouton {name} ?", "Est-ce un bouton pour {name} ?", "Cette icône signifie-t-elle {name} ?", "Cliqueriez-vous ici pour {name} ?"],
        "es": ["¿Es este el botón de {name}?", "¿Es un botón para {name}?", "¿Este icono significa {name}?", "¿Harías clic aquí para {name}?"],
    },
    "document": {
        "en": ["Is this {x}?", "Is this a picture of {x}?", "Does the picture show {x}?", "Are we looking at {x}?"],
        "nl": ["Is dit {x}?", "Is dit een afbeelding van {x}?", "Toont de afbeelding {x}?", "Kijken we naar {x}?"],
        "de": ["Ist das {x}?", "Ist auf dem Bild {x} zu sehen?", "Ist hier {x} abgebildet?", "Sehen wir hier {x}?"],
        "fr": ["Est-ce {x} ?", "Est-ce une image de {x} ?", "L'image montre-t-elle {x} ?", "Regarde-t-on {x} ?"],
        "es": ["¿Es esto {x}?", "¿Es una imagen de {x}?", "¿La imagen muestra {x}?", "¿Estamos viendo {x}?"],
    },
}
# What a choice asks. The model never reads this for a choice; it is here because a real question has it.
CHOOSE = {
    "photo": {"en": ["What is in this photo?", "Which of these does the photo show?"], "nl": ["Wat staat er op deze foto?", "Welke hiervan toont de foto?"],
              "de": ["Was ist auf diesem Foto zu sehen?", "Was davon zeigt das Foto?"], "fr": ["Que voit-on sur cette photo ?", "Lequel de ces éléments la photo montre-t-elle ?"],
              "es": ["¿Qué hay en esta foto?", "¿Cuál de estos muestra la foto?"]},
    "album": {"en": ["Which album does this photo belong in?"], "nl": ["In welk album hoort deze foto?"], "de": ["In welches Album gehört dieses Foto?"],
              "fr": ["Dans quel album cette photo va-t-elle ?"], "es": ["¿En qué álbum va esta foto?"]},
    "control": {"en": ["What does this button do?", "What is this control for?"], "nl": ["Wat doet deze knop?", "Waar is deze knop voor?"],
                "de": ["Was macht diese Schaltfläche?", "Wofür ist dieses Bedienelement?"], "fr": ["Que fait ce bouton ?", "À quoi sert cette commande ?"],
                "es": ["¿Qué hace este botón?", "¿Para qué sirve este control?"]},
    "document": {"en": ["What is this a picture of?", "What kind of picture is this?"], "nl": ["Waar is dit een afbeelding van?", "Wat voor afbeelding is dit?"],
                 "de": ["Was zeigt dieses Bild?", "Was für ein Bild ist das?"], "fr": ["De quoi est-ce une image ?", "Quel genre d'image est-ce ?"],
                 "es": ["¿De qué es esta imagen?", "¿Qué tipo de imagen es esta?"]},
    "page": {"en": ["What is going on on this page?", "What state is this page in?"], "nl": ["Wat is er aan de hand op deze pagina?", "In welke toestand is deze pagina?"],
             "de": ["Was ist auf dieser Seite los?", "In welchem Zustand ist diese Seite?"], "fr": ["Que se passe-t-il sur cette page ?", "Dans quel état est cette page ?"],
             "es": ["¿Qué pasa en esta página?", "¿En qué estado está esta página?"]},
}
# A page's states are sentences already (vision/synth.py PAGE_WORDS): each state has its own ways to say it, to deny it
# and to ask about it, so the page family has no templates here.

LETTERS = "abcdefghijklmnop"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "x"


def pick(rng: random.Random, options: list[str], unseen: bool) -> tuple[str, bool]:
    """One wording: among the trained ones, or (unseen) the held-out one half of the time."""
    if unseen and rng.random() < 0.5:
        return options[UNSEEN], True
    return rng.choice(options[:UNSEEN]), False


class Family:
    """One kind of picture's things: words(label, lang) gives {"x", "name", optional "does", "say": [...], "other": [...], "ask": [...]}."""

    def __init__(self, name: str, words) -> None:
        self.name = name
        self.words = words

    def say(self, rng: random.Random, label: str, lang: str, style: str, unseen: bool) -> tuple[str, bool]:
        """One option's text in the given style: name, x, does or say."""
        w = self.words(label, lang)
        if style == "name":
            return w["name"], False
        if style == "x":
            return w["x"], False
        if style == "does" and w.get("does"):
            return w["does"], False
        if "say" in w:
            return pick(rng, w["say"], unseen)
        template, new = pick(rng, SAY[self.name][lang], unseen)
        return template.format(**w), new

    def styles(self, label: str, lang: str) -> list[str]:
        w = self.words(label, lang)
        out = ["say", "say"]
        if "say" not in w:
            out += ["name", "name", "x"] if w["x"] != w["name"] else ["name", "name"]
        if w.get("does"):
            out += ["does", "does"]
        return out

    def choice(self, rng: random.Random, lang: str, answer: str, others: list[str], unseen: bool = False) -> dict:
        """A choice among the answer and the others, in one style, shuffled."""
        style = rng.choice(self.styles(answer, lang))
        labels = [answer] + list(others)
        rng.shuffle(labels)
        keys_as = rng.choices(["slug", "letter", "self"], [0.5, 0.25, 0.25])[0] if style in ("name", "x") else rng.choices(["slug", "letter"], [0.6, 0.4])[0]
        criteria, key_of, new = {}, {}, False
        template = None
        if style == "say" and "say" not in self.words(answer, lang):
            template, new = pick(rng, SAY[self.name][lang], unseen)  # one wording for all of a question's options
        for i, label in enumerate(labels):
            w = self.words(label, lang)
            if template is not None:
                text = template.format(**w)
            else:
                text, n = self.say(rng, label, lang, style, unseen)
                new = new or n
            key = LETTERS[i] if keys_as == "letter" else text if keys_as == "self" else slug(self.words(label, "en")["name"])
            if key in criteria:  # two things with one name in this language ("cactus")
                key = f"{key}_{i}"
            criteria[key] = None if keys_as == "self" else text
            key_of[label] = key
        q = {"type": "choice", "instructions": rng.choice(CHOOSE[self.name][lang]), "criteria": criteria}
        return {"q": q, "answer": key_of[answer], "kind": "choice", "form": style, "lang": lang, "unseen_wording": new, "n": len(labels)}

    def noul(self, rng: random.Random, lang: str, label: str, truth: bool, unseen: bool = False) -> dict:
        """A yes/no question about one thing: with what yes and no look like, with only the yes, or bare."""
        w = self.words(label, lang)
        form = rng.choices(["pair", "true", "bare"], [0.4, 0.25, 0.35])[0]
        ask, new = pick(rng, w["ask"], unseen) if "ask" in w else pick(rng, ASK[self.name][lang], unseen)
        ask = ask.format(**w)
        q: dict = {"type": "noul", "instructions": ask}
        if form != "bare":
            yes, n1 = self.say(rng, label, lang, rng.choice([s for s in self.styles(label, lang) if s not in ("name", "x")]), unseen)
            q["criteria"] = {"true": yes}
            new = n1  # the question itself isn't read when a yes is described
            if form == "pair":
                no, n2 = pick(rng, w["other"], unseen) if "other" in w else pick(rng, OTHER[self.name][lang], unseen)
                q["criteria"]["false"] = no.format(**w)
                new = new or n2
        return {"q": q, "answer": "true" if truth else "false", "kind": "noul", "form": form, "lang": lang, "unseen_wording": new, "n": 2}


def language(rng: random.Random) -> str:
    return rng.choices(LANGS, LANG_WEIGHTS)[0]
