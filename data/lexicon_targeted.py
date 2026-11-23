# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Targeted building blocks for the decisions pool v1 covered badly (found with Atelier's eval): matching a request
to a skill by meaning rather than shared words, personal facts mentioned in passing, and how much work a message
is. Written fresh: nothing here repeats Atelier's eval cases, so the eval stays a fair test. The teacher labels."""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------------------------------- skills
# (name, description, requests that need it but share few words with the description)

SKILL_CATALOG: list[tuple[str, str, dict[str, list[str]]]] = [
    ("weekly-update", "Every Friday: collects what I worked on from my agenda and outbox and drafts the team's progress report.", {
        "en": ["friday again, can you put together what I did this week for the team?", "time for the weekly write-up for my boss",
               "summarise my week for the Monday standup doc", "draft this week's progress note please"],
        "nl": ["kun je mijn weekverslag voor het team maken?", "het is vrijdag, zet even op een rij wat ik deze week gedaan heb"],
        "de": ["kannst du meinen Wochenbericht fürs Team schreiben?"], "fr": ["tu peux préparer mon point hebdo pour l'équipe ?"],
        "es": ["¿puedes preparar mi informe semanal para el equipo?"]}),
    ("expenses", "Gets work costs I paid myself reimbursed: reads receipts and completes the claim form at work.", {
        "en": ["I paid for the client lunch myself, get my money back", "here are the hotel and train tickets from the conference, submit them",
               "need to be reimbursed for the parking at the airport", "file these Uber receipts with finance"],
        "nl": ["ik heb de lunch met de klant zelf betaald, kun je dat declareren?", "dien deze bonnetjes van de reis in"],
        "de": ["reich bitte meine Reisekosten von letzter Woche ein"], "fr": ["fais ma note de frais avec ces tickets"],
        "es": ["presenta estos tickets de taxi como gastos"]}),
    ("dinner-planner", "Seven meat-free evening meals for the household, plus the supermarket list.", {
        "en": ["no idea what to cook next week, sort it out and tell me what to buy", "we need meal ideas for the next 7 days, no meat",
               "what's for dinner monday to sunday? and the groceries", "help me plan our evening meals for the week"],
        "nl": ["wat eten we volgende week? maak ook een boodschappenlijst", "bedenk avondeten voor de hele week, vegetarisch"],
        "de": ["was kochen wir diese Woche? mit Einkaufsliste"], "fr": ["prévois les dîners de la semaine et la liste de courses"],
        "es": ["planifica las cenas de la semana con la lista de la compra"]}),
    ("social-post", "Writes professional-network updates that sound like me.", {
        "en": ["write something for my network about our team winning the award", "I want to announce my new job on linkedin",
               "turn this conference talk into a post for my followers", "share our product launch with my professional contacts"],
        "nl": ["schrijf een berichtje voor linkedin over mijn nieuwe baan", "kun je een post maken over ons jubileum voor mijn netwerk?"],
        "de": ["schreib einen LinkedIn-Beitrag über unser neues Büro"], "fr": ["rédige un post LinkedIn sur notre levée de fonds"],
        "es": ["escribe una publicación para LinkedIn sobre mi ascenso"]}),
    ("code-review", "Reads code changes on GitHub, spots bugs and style issues, and comments on the lines.", {
        "en": ["can you go over this PR before I merge it? github.com/org/app/pull/88", "check my colleague's changes for mistakes",
               "look through the diff and flag anything risky", "is this pull request ok to ship?"],
        "nl": ["kun je deze pull request nakijken op fouten?", "bekijk de wijzigingen van mijn collega even"],
        "de": ["schau dir bitte diesen Pull Request an"], "fr": ["relis cette pull request et commente"], "es": ["revisa este pull request antes de fusionarlo"]}),
    ("trip-packing", "Works out what to take on a journey, given where, when and the forecast.", {
        "en": ["off to Norway in February, what do I need to take?", "weekend in Rome with the kids, help me not forget anything",
               "camping in the Ardennes next week, what should go in the bag?", "going skiing, list what to bring"],
        "nl": ["we gaan twee weken naar Portugal, wat moet ik inpakken?", "wat neem ik mee voor een weekend Londen?"],
        "de": ["was muss ich für eine Woche Mallorca einpacken?"], "fr": ["qu'est-ce que je dois emporter pour une semaine en Écosse ?"],
        "es": ["¿qué meto en la maleta para un viaje a Noruega?"]}),
    ("payment-reminders", "Spots overdue customer bills in the bookkeeping app and writes friendly payment chasers.", {
        "en": ["some clients still haven't paid, chase them", "who is late with their payment? write to them nicely",
               "send gentle nudges to customers with overdue bills", "follow up on the outstanding invoices from October"],
        "nl": ["welke klanten hebben nog niet betaald? stuur ze een herinnering", "maak betalingsherinneringen voor de openstaande facturen"],
        "de": ["schick Zahlungserinnerungen an die säumigen Kunden"], "fr": ["relance les clients qui n'ont pas payé"],
        "es": ["recuerda a los clientes las facturas pendientes"]}),
    ("meeting-minutes", "Condenses a call or meeting transcript into what was decided and who does what next.", {
        "en": ["here's the recording text from the board call, what did we agree?", "who has to do what after this meeting? transcript below",
               "clean up these call notes into decisions and to-dos", "make minutes from this conversation"],
        "nl": ["maak notulen van dit gesprek", "wat hebben we besloten in de vergadering? hier is de transcriptie"],
        "de": ["erstelle ein Protokoll aus diesem Transkript"], "fr": ["fais le compte rendu de cette réunion"], "es": ["haz el acta de esta reunión"]}),
    ("price-tracker", "Checks the price of a product on a few shops every morning and tells me when it drops.", {
        "en": ["let me know when the PS6 gets cheaper", "keep an eye on this TV and ping me if it goes on sale",
               "I'll buy the stroller once it's under 300 euros, watch it for me"],
        "nl": ["laat het weten als deze fiets in de aanbieding is", "houd de prijs van deze laptop in de gaten"],
        "de": ["sag mir, wenn der Preis von dem Fernseher fällt"], "fr": ["préviens-moi quand ce casque baisse de prix"],
        "es": ["avísame cuando baje el precio de esta tele"]}),
    ("inbox-triage", "Sorts my inbox into to-do, waiting and read later, and drafts short replies.", {
        "en": ["my mail is a mess, sort it out", "go through my emails and tell me what needs an answer",
               "clean up my inbox and draft quick responses"],
        "nl": ["ruim mijn inbox op", "welke mails moet ik nog beantwoorden?"], "de": ["räum meinen Posteingang auf"],
        "fr": ["trie ma boîte mail"], "es": ["organiza mi bandeja de entrada"]}),
    ("tab-compare", "Compares products or offers from my open tabs in a table.", {
        "en": ["which of these three vacuum cleaners I have open is the best deal?", "put the offers in my tabs side by side",
               "compare the insurance quotes I've opened"],
        "nl": ["vergelijk de telefoons in mijn tabbladen", "welke van deze aanbiedingen is het voordeligst?"],
        "de": ["vergleiche die Angebote in meinen Tabs"], "fr": ["compare les offres dans mes onglets"], "es": ["compara las ofertas de mis pestañas"]}),
    ("job-application", "Tailors my CV and a cover letter to a job ad I open.", {
        "en": ["I want to apply for the job on this page", "adapt my resume to this vacancy and write the letter",
               "help me go for this marketing position"],
        "nl": ["ik wil solliciteren op deze vacature", "pas mijn cv aan voor deze baan"], "de": ["ich möchte mich auf diese Stelle bewerben"],
        "fr": ["je veux postuler à cette offre"], "es": ["quiero solicitar este puesto"]}),
    ("flight-finder", "Searches flights on a few airline sites and lists the cheapest with baggage rules.", {
        "en": ["get me to Barcelona in early May as cheaply as possible", "what's the cheapest way to fly to Dublin with a suitcase?",
               "find flights Amsterdam to Rome for next weekend"],
        "nl": ["zoek goedkope vluchten naar Lissabon in juni", "hoe vlieg ik het voordeligst naar Kreta?"],
        "de": ["finde günstige Flüge nach Lissabon"], "fr": ["trouve un vol pas cher pour Lisbonne"], "es": ["busca vuelos baratos a Ámsterdam"]}),
    ("page-translate", "Translates the current page into Dutch, keeping the layout.", {
        "en": ["I can't read this German site, make it Dutch", "put this page in Dutch for my mum"],
        "nl": ["kun je deze pagina naar het Nederlands vertalen?", "ik snap deze Franse site niet, vertaal hem"],
        "de": ["übersetze diese Seite ins Niederländische"], "fr": ["traduis cette page en néerlandais"], "es": ["traduce esta página al neerlandés"]}),
    ("budget", "Reads my bank export and shows where my money went this month.", {
        "en": ["where did all my money go in October?", "how much did I spend on eating out this month?",
               "look at my bank statement and break down my spending"],
        "nl": ["waar is mijn geld deze maand aan opgegaan?", "hoeveel heb ik uitgegeven aan boodschappen?"],
        "de": ["wofür habe ich diesen Monat Geld ausgegeben?"], "fr": ["où est passé mon argent ce mois-ci ?"], "es": ["¿en qué gasté mi dinero este mes?"]}),
    ("flashcards", "Turns lecture notes into flashcards for spaced repetition.", {
        "en": ["I have an exam on Friday, make me something to study these notes with", "quiz cards from my biology slides please"],
        "nl": ["maak oefenkaartjes van mijn aantekeningen", "help me leren voor mijn tentamen met deze stof"],
        "de": ["mach Karteikarten aus meinen Notizen"], "fr": ["fais des fiches de révision"], "es": ["haz tarjetas de estudio con mis apuntes"]}),
    ("bakery-socials", "Plans a month of social media posts for my bakery.", {
        "en": ["what should the shop post on instagram in December?", "content calendar for the bakery's socials"],
        "nl": ["wat posten we volgende maand op instagram voor de bakkerij?"], "de": ["plane die Instagram-Posts für die Bäckerei"],
        "fr": ["planifie les posts de la boulangerie"], "es": ["planifica las publicaciones de la panadería"]}),
    ("house-hunt", "Checks new listings on Funda for my search and summarises the ones worth viewing.", {
        "en": ["any new houses for us this week?", "which of the new homes for sale should we go and see?"],
        "nl": ["staan er nieuwe huizen op funda die bij ons passen?", "welke woningen moeten we gaan bezichtigen?"],
        "de": ["gibt es neue Wohnungen für uns?"], "fr": ["y a-t-il de nouvelles maisons à visiter ?"], "es": ["¿hay pisos nuevos que valga la pena ver?"]}),
    ("returns", "Starts a return for an online order and books the parcel pickup.", {
        "en": ["the shoes don't fit, send them back", "I want to return the blender I bought last week"],
        "nl": ["ik wil deze jas terugsturen", "regel een retour voor mijn bestelling"], "de": ["ich möchte die Bestellung zurückschicken"],
        "fr": ["je veux renvoyer ma commande"], "es": ["quiero devolver mi pedido"]}),
    ("research-brief", "Researches a topic across several sources and writes a cited two-page brief.", {
        "en": ["I need to understand the new EU AI rules before Monday's meeting", "dig into heat pumps vs gas boilers and give me the facts with sources"],
        "nl": ["zoek alles uit over de nieuwe huurwet en geef me een overzicht met bronnen"], "de": ["recherchiere das Thema und fasse es mit Quellen zusammen"],
        "fr": ["fais-moi une synthèse sourcée sur ce sujet"], "es": ["investiga este tema y hazme un resumen con fuentes"]}),
]


def skill_request(rng: random.Random, code: str) -> tuple[list[dict], str]:
    """A chunk of 3-8 skills and a message: aimed at one of them (by meaning), or at none."""
    k = rng.choice([3, 5, 8])
    chunk = rng.sample(SKILL_CATALOG, k)
    skills = [{"name": n, "description": d} for n, d, _ in chunk]
    if rng.random() < 0.65:
        _, _, requests = rng.choice(chunk)
        message = rng.choice(requests.get(code) or requests["en"])
    else:
        # A request for a skill NOT offered (a hard negative), or something unrelated.
        others = [s for s in SKILL_CATALOG if s not in chunk]
        if others and rng.random() < 0.6:
            message = rng.choice(rng.choice(others)[2].get(code) or rng.choice(others)[2]["en"])
        else:
            message = rng.choice(UNRELATED.get(code, UNRELATED["en"]))
    return skills, message


UNRELATED = {"en": ["what's the weather like tomorrow?", "who won the match last night?", "thanks, that's all", "how tall is the eiffel tower",
                    "tell me a joke", "convert 5 miles to km", "what time does the supermarket close"],
             "nl": ["wat voor weer wordt het morgen?", "hoe laat is het in New York?", "bedankt!", "hoeveel is 15% van 80"],
             "de": ["wie wird das Wetter morgen?", "danke!"], "fr": ["quel temps fera-t-il demain ?", "merci !"],
             "es": ["¿qué tiempo hará mañana?", "¡gracias!"]}

# ---------------------------------------------------------------------------------------------------- memory

FACTS = {
    "en": ["we don't eat pork at home", "my husband is diabetic", "I'm coeliac", "our son Max is turning 6 in March",
           "I teach maths at a secondary school", "I'm saving for a house", "we have two cats", "I live in Haarlem",
           "I'm learning Spanish", "my manager's name is Ingrid", "I always fly KLM for the miles", "I hate early meetings",
           "I'm pregnant with our second", "I run a small web shop selling candles", "my mother lives with us",
           "I'm doing my thesis on wind energy", "I work four days a week, never on Wednesdays", "please write to me informally"],
    "nl": ["we eten thuis geen vlees", "mijn vriendin is lactose-intolerant", "onze dochter Lotte zit in groep 5", "ik werk als verpleegkundige",
           "ik spaar voor een camper", "we wonen in Zwolle", "ik ben met een cursus Italiaans bezig", "mijn baas heet Wouter",
           "ik werk nooit op vrijdag", "ik heb een eigen fietsenwinkel", "ik schrijf mijn scriptie over armoede", "spreek me alsjeblieft met je aan"],
    "de": ["ich bin Veganerin", "meine Tochter heißt Mia", "ich arbeite als Ingenieur in Köln", "wir haben einen Hund"],
    "fr": ["je suis végétarien", "mon fils s'appelle Hugo", "je travaille à Lyon comme infirmière", "j'ai un chat"],
    "es": ["soy alérgica al marisco", "mi hijo se llama Pablo", "trabajo como profesora en Valencia", "tenemos dos perros"],
}
TASKS = {
    "en": ["can you suggest a recipe for tonight?", "find me a hotel for the weekend", "what's a good present for a birthday?",
           "plan a day out", "draft an email to the school", "look for a course I could take"],
    "nl": ["kun je een recept voor vanavond voorstellen?", "zoek een hotel voor het weekend", "wat is een leuk cadeau?", "plan een dagje uit"],
    "de": ["schlag mir ein Rezept vor", "such mir ein Hotel"], "fr": ["propose-moi une recette", "trouve-moi un hôtel"],
    "es": ["sugiéreme una receta", "búscame un hotel"],
}
REPLIES = {"en": ["Sure! Here are three options that fit.", "Here's a suggestion that takes that into account.", "Done, see below."],
           "nl": ["Zeker! Hier zijn drie opties die passen.", "Hier is een voorstel dat daar rekening mee houdt."],
           "de": ["Klar! Hier sind drei Vorschläge."], "fr": ["Bien sûr ! Voici trois idées."], "es": ["¡Claro! Aquí tienes tres opciones."]}
ONEOFF = {"en": ["how many grams in a cup of flour?", "what's the capital of Canada?", "translate 'good luck' into French",
                 "when does the bakery on the corner close?", "is it going to rain at 5?", "spell 'necessary'"],
          "nl": ["hoeveel gram is een kopje bloem?", "wat is de hoofdstad van Canada?", "regent het om vijf uur?"],
          "de": ["wie spät ist es in Tokio?"], "fr": ["quelle est la capitale du Canada ?"], "es": ["¿cuál es la capital de Canadá?"]}
ONEOFF_REPLIES = ["About 120 grams.", "Ottawa.", "Bonne chance.", "It closes at 18:00.", "Light rain is expected around then."]


def memory_chat(rng: random.Random, code: str) -> str:
    r = rng.random()
    if r < 0.55:  # a fact in passing, then a task
        fact = rng.choice(FACTS.get(code) or FACTS["en"])
        task = rng.choice(TASKS.get(code) or TASKS["en"])
        you = rng.choice([f"{fact}, {task}", f"{task} ({fact})", f"{fact}. {task}"])
        return f"You: {you}\nCompanion: {rng.choice(REPLIES.get(code) or REPLIES['en'])}"
    if r < 0.7:  # a fact on its own
        fact = rng.choice(FACTS.get(code) or FACTS["en"])
        return f"You: just so you know, {fact}\nCompanion: {rng.choice(['Noted.', 'Thanks, good to know.', 'Got it.'])}"
    q = rng.choice(ONEOFF.get(code) or ONEOFF["en"])  # nothing lasting
    return f"You: {q}\nCompanion: {rng.choice(ONEOFF_REPLIES)}"


# ---------------------------------------------------------------------------------------------------- effort

BIG = {
    "en": ["Go through the last three years of our energy bills, find what drove the increase and suggest five ways to cut it.",
           "Research the best family cars under 35k, compare range, boot space and safety ratings, and recommend two.",
           "Find every conference on AI in Europe next spring, with dates, prices and deadlines for talks, in a spreadsheet.",
           "Plan our two-week road trip through Scotland: route, a hotel each night, and the things the kids would like.",
           "Write a full business plan for my candle shop, with costs, pricing and a marketing section.",
           "Check all 40 product pages on our site for broken links and wrong prices and list what needs fixing.",
           "Compare the mortgage offers from four banks over 30 years, including fees, and tell me which is cheapest overall."],
    "nl": ["Zoek uit welke energieleverancier het goedkoopst is voor ons verbruik, vergelijk vijf contracten en zet het in een tabel.",
           "Plan een rondreis van tien dagen door Italië met hotels, treinen en een budget per dag.",
           "Vergelijk de zorgverzekeringen voor volgend jaar op dekking en premie en geef een advies voor ons gezin.",
           "Schrijf een uitgebreid projectplan voor de verbouwing met planning, kosten en wie wat doet."],
    "de": ["Vergleiche zehn Laptops unter 1000 Euro nach Akku, Gewicht und Bewertungen und empfiehl mir drei.",
           "Plane eine Woche Wien mit Hotel, Museen und Restaurants für jeden Tag."],
    "fr": ["Compare cinq assurances habitation en détail et recommande la meilleure pour nous.",
           "Organise un voyage de dix jours au Japon avec les trains, les hôtels et un budget."],
    "es": ["Compara diez portátiles por batería, peso y opiniones y recomiéndame tres.",
           "Planifica una semana en Andalucía con hoteles, rutas y presupuesto diario."],
}
PLAN_OFFERS = {"en": ["I can go through all 25 listings, check each one against your list and shortlist the best five. Shall I start?",
                      "I could research the options in depth and write up a comparison. Want me to do that?"],
               "nl": ["Ik kan alle 25 woningen doorlopen en de beste vijf voor je selecteren. Zal ik beginnen?"],
               "de": ["Ich kann alle Angebote prüfen und eine Tabelle machen. Soll ich?"], "fr": ["Je peux tout comparer et faire un tableau. Je commence ?"],
               "es": ["Puedo revisar todas las opciones y hacer una tabla. ¿Empiezo?"]}
GO = {"en": ["yes please", "go for it", "sure, do it", "yep"], "nl": ["ja graag", "doe maar", "ja, ga je gang"], "de": ["ja bitte", "mach"],
      "fr": ["oui vas-y", "d'accord"], "es": ["sí, adelante", "vale, hazlo"]}
TINY = {"en": ["thank you so much!", "ok great", "how many days in a leap year?", "what's 12 times 8?", "good night!", "who wrote Hamlet?"],
        "nl": ["dankjewel!", "top", "hoeveel dagen heeft een schrikkeljaar?", "welterusten!", "wie schreef de Max Havelaar?"],
        "de": ["vielen Dank!", "gute Nacht!"], "fr": ["merci beaucoup !", "bonne nuit !"], "es": ["¡muchas gracias!", "¡buenas noches!"]}
ORDINARY = {"en": ["can you shorten this paragraph?", "what's a good name for a golden retriever?", "explain what an index fund is",
                   "write a two-line birthday message for my aunt"],
            "nl": ["kun je deze alinea korter maken?", "leg uit wat een indexfonds is", "schrijf een kort verjaardagsberichtje"],
            "de": ["erklär mir, was ein ETF ist"], "fr": ["explique-moi ce qu'est un ETF"], "es": ["explícame qué es un fondo indexado"]}


def effort_message(rng: random.Random, code: str) -> tuple[str, str]:
    r = rng.random()
    if r < 0.35:
        return rng.choice(BIG.get(code) or BIG["en"]), ""
    if r < 0.55:
        return rng.choice(GO.get(code) or GO["en"]), rng.choice(PLAN_OFFERS.get(code) or PLAN_OFFERS["en"])
    if r < 0.8:
        return rng.choice(TINY.get(code) or TINY["en"]), ""
    return rng.choice(ORDINARY.get(code) or ORDINARY["en"]), ""


# Compositional multi-step tasks, so "a lot of work" comes in thousands of distinct forms per language.
BIG_PARTS = {
    "en": {"verb": ["Compare", "Research", "Find and rank", "Go through", "Look into"],
           "things": ["{n} electric bikes under €{p}", "{n} family hotels in {c}", "{n} laptops for video editing", "the {n} cheapest flights to {c}",
                      "{n} daycare centres near us", "{n} project management tools", "{n} health insurance plans", "{n} used cars on Marktplaats"],
           "on": ["on price, reviews and delivery time", "on range, weight and warranty", "on fees, cancellation rules and ratings",
                  "on safety, space and running costs"],
           "extra": ["check what other people complain about", "read the small print", "work out the total cost over three years",
                     "check availability for the second week of {m}"],
           "out": ["put it all in a table with links", "and recommend the best two", "then write me a one-page summary", "and draft an email to the winner"]},
    "nl": {"verb": ["Vergelijk", "Zoek uit", "Onderzoek"],
           "things": ["{n} elektrische fietsen onder de €{p}", "{n} hotels in {c}", "{n} kinderopvanglocaties bij ons in de buurt", "{n} energiecontracten",
                      "{n} tweedehands auto's"],
           "on": ["op prijs, reviews en levertijd", "op kosten, voorwaarden en beoordelingen"],
           "extra": ["lees de kleine lettertjes", "reken de totale kosten over drie jaar uit", "kijk wat mensen klagen"],
           "out": ["en zet alles in een tabel", "en geef een advies", "en schrijf een samenvatting van één pagina"]},
    "de": {"verb": ["Vergleiche", "Recherchiere"], "things": ["{n} E-Bikes unter {p} Euro", "{n} Hotels in {c}", "{n} Stromtarife"],
           "on": ["nach Preis, Bewertungen und Lieferzeit"], "extra": ["prüfe das Kleingedruckte", "rechne die Kosten über drei Jahre"],
           "out": ["und mach eine Tabelle", "und empfiehl zwei"]},
    "fr": {"verb": ["Compare", "Recherche"], "things": ["{n} vélos électriques à moins de {p} €", "{n} hôtels à {c}", "{n} assurances auto"],
           "on": ["selon le prix, les avis et le délai"], "extra": ["lis les petites lignes", "calcule le coût sur trois ans"],
           "out": ["et fais un tableau", "et recommande les deux meilleurs"]},
    "es": {"verb": ["Compara", "Investiga"], "things": ["{n} bicis eléctricas por menos de {p} €", "{n} hoteles en {c}", "{n} seguros de coche"],
           "on": ["por precio, opiniones y plazo de entrega"], "extra": ["lee la letra pequeña", "calcula el coste en tres años"],
           "out": ["y haz una tabla", "y recomienda los dos mejores"]},
}
SMALL_PARTS = {
    "en": ["what's {a} plus {b}?", "how do you say '{w}' in Spanish?", "what year did {e} happen?", "thanks a lot {name}!", "ok, cheers",
           "is {c} in Spain?", "what does '{w}' mean?"],
    "nl": ["hoeveel is {a} plus {b}?", "wat betekent '{w}'?", "ligt {c} in Spanje?", "top, bedankt!", "dankje {name}!"],
    "de": ["was ist {a} plus {b}?", "danke {name}!"], "fr": ["combien font {a} plus {b} ?", "merci {name} !"],
    "es": ["¿cuánto es {a} más {b}?", "¡gracias {name}!"],
}
WORDS = ["serendipity", "gezellig", "Fernweh", "ubuntu", "hygge", "saudade"]
EVENTS = ["the moon landing", "the fall of the Berlin Wall", "the first iPhone launch"]
CITIES_T = ["Lisbon", "Porto", "Valencia", "Bologna", "Ghent", "Bilbao", "Kraków", "Lyon"]


def big_task(rng: random.Random, code: str) -> str:
    p = BIG_PARTS.get(code) or BIG_PARTS["en"]
    fill = lambda s: s.format(n=rng.randint(3, 12), p=rng.choice([500, 1500, 2000, 3000]), c=rng.choice(CITIES_T),  # noqa: E731
                              m=rng.choice(["May", "June", "August", "October"]))
    parts = [rng.choice(p["verb"]), fill(rng.choice(p["things"])), rng.choice(p["on"])]
    text = " ".join(parts)
    if rng.random() < 0.7:
        text += ", " + fill(rng.choice(p["extra"]))
    return text + " " + rng.choice(p["out"]) + "."


def small_message(rng: random.Random, code: str) -> str:
    t = rng.choice(SMALL_PARTS.get(code) or SMALL_PARTS["en"])
    return t.format(a=rng.randint(2, 99), b=rng.randint(2, 99), w=rng.choice(WORDS), e=rng.choice(EVENTS), c=rng.choice(CITIES_T),
                    name=rng.choice(["mate", "buddy", "Pip", ""])).replace("  ", " ")


def effort_message_v2(rng: random.Random, code: str) -> tuple[str, str]:
    r = rng.random()
    if r < 0.4:
        return big_task(rng, code), ""
    if r < 0.6:
        offer = rng.choice(PLAN_OFFERS.get(code) or PLAN_OFFERS["en"])
        return rng.choice(GO.get(code) or GO["en"]), offer
    if r < 0.85:
        return small_message(rng, code), ""
    return rng.choice(ORDINARY.get(code) or ORDINARY["en"]), ""
