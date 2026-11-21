"""Multilingual building blocks (en, nl, de, fr, es) for the request pool. They only need to be varied and
realistic: the teacher decides the labels. Kept deliberately plain so it is easy to extend when a decision goes
wrong in real use."""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------------------------------- shared nouns

CITIES = ["Amsterdam", "Rotterdam", "Utrecht", "Lisbon", "Berlin", "Paris", "Madrid", "London", "Rome", "Antwerp",
          "Brussels", "Munich", "Barcelona", "Vienna", "Copenhagen", "New York", "Tokyo", "Eindhoven", "Groningen",
          "Lyon", "Sevilla", "Hamburg", "Gent", "Den Haag", "Porto", "Prague", "Dublin", "Zurich"]
PRODUCTS = ["iphone 17", "airpods pro", "macbook air m5", "pixel 11", "nintendo switch 2", "dyson v15", "kindle",
            "standing desk", "e-bike", "air fryer", "running shoes", "winter jacket", "robot vacuum", "espresso machine",
            "4k monitor", "noise cancelling headphones", "lego technic", "garden chair", "baby stroller", "ps6",
            "electric toothbrush", "smartwatch", "tent 4 person", "sofa bed", "led strip", "drill"]
BRANDS = ["ikea", "albert heijn", "jumbo", "bol.com", "coolblue", "zalando", "decathlon", "hema", "action", "mediamarkt",
          "lidl", "aldi", "h&m", "primark", "apple store", "kruidvat", "gamma", "praxis", "blokker", "douglas"]
SITES = ["nos.nl", "nu.nl", "bbc.co.uk", "youtube", "github", "gmail", "linkedin", "reddit", "wikipedia", "spotify",
         "netflix", "marktplaats", "funda", "buienradar", "ns.nl", "9292", "thuisbezorgd", "booking.com", "airbnb",
         "google maps", "outlook", "notion", "figma", "stackoverflow", "spiegel.de", "lemonde.fr", "elpais.com"]
PEOPLE = ["Taylor Swift", "Max Verstappen", "Mark Rutte", "Lionel Messi", "Barack Obama", "Emma Watson", "Elon Musk",
          "Frenkie de Jong", "Beyoncé", "Angela Merkel", "Emmanuel Macron", "Rosalía", "Stromae", "Tim Cook"]
NAMES = ["Sam", "Lisa", "Daan", "Emma", "Noah", "Sophie", "Jonas", "Priya", "Lucas", "Fatima", "Anna", "Mehmet", "Julia",
         "Thomas", "Sara", "Pedro", "Chloé", "Lukas", "Marta", "Kees"]

# ---------------------------------------------------------------------------------------------------- search vs ask

SEARCH_T = {
    "en": ["{p} price", "buy {p}", "{p} review", "{b} opening hours", "{b} near me", "weather {c}", "{c} weather tomorrow",
           "{s}", "{s} login", "{person}", "{person} age", "flights to {c}", "hotels {c}", "news {c}", "{c} restaurants",
           "train {c} to {c2}", "cheap {p}", "{p} vs {p2}", "{b} customer service phone number", "best {p} 2026",
           "{c} population", "pizza near me", "pharmacy open now", "{c} museum tickets", "{p} manual pdf",
           "{b} sale", "ajax score", "euro dollar exchange rate", "{p} second hand", "{c} map"],
    "nl": ["{p} prijs", "{p} kopen", "{b} openingstijden", "{b} in de buurt", "weer {c}", "weerbericht {c} morgen", "{s}",
           "{person} leeftijd", "vluchten naar {c}", "hotel {c}", "nieuws {c}", "restaurants {c}", "trein {c} {c2}",
           "goedkope {p}", "{p} of {p2}", "beste {p} 2026", "apotheek open zondag", "{b} klantenservice telefoonnummer",
           "{c} museum tickets", "{p} tweedehands", "huizen te koop {c}", "dienstregeling ns", "pinautomaat in de buurt"],
    "de": ["{p} preis", "{p} kaufen", "{b} öffnungszeiten", "wetter {c}", "{s}", "flüge nach {c}", "hotel {c}",
           "nachrichten {c}", "{p} test", "bester {p} 2026", "apotheke notdienst", "zug {c} {c2}", "{person} alter"],
    "fr": ["prix {p}", "acheter {p}", "horaires {b}", "météo {c}", "{s}", "vol pour {c}", "hôtel {c}", "actualités {c}",
           "meilleur {p} 2026", "pharmacie de garde", "train {c} {c2}", "{person} âge", "restaurant {c}"],
    "es": ["precio {p}", "comprar {p}", "horario {b}", "tiempo {c}", "{s}", "vuelos a {c}", "hotel {c}", "noticias {c}",
           "mejor {p} 2026", "farmacia de guardia", "tren {c} {c2}", "{person} edad", "restaurantes {c}"],
}
ASK_T = {
    "en": ["write a polite email to {n} saying I can't make it on {d}", "summarise this page", "summarise the tabs I have open",
           "explain how {topic} works", "compare the {p} and the {p2} for me", "plan a {k}-day trip to {c}",
           "translate this page into {lang}", "what should I cook tonight with {food}", "help me write a cover letter for a {job} job",
           "make a packing list for {c} in {month}", "draft a reply to {n} about the invoice", "calculate 15% tip on €{k}0",
           "is it worth buying a {p}? pros and cons", "fill in this form with my details", "book a table for {k} at a restaurant in {c}",
           "find the cheapest {p} across these tabs and put them in a table", "rewrite this paragraph to sound more formal",
           "why is the sky blue", "what's the difference between {topic} and {topic2}", "remind me to call {n} at 5",
           "give me 5 ideas for a birthday present for {n}", "check this text for spelling mistakes"],
    "nl": ["schrijf een beleefde mail aan {n} dat ik {d} niet kan", "vat deze pagina samen", "leg uit hoe {topic} werkt",
           "vergelijk de {p} met de {p2}", "plan een reis van {k} dagen naar {c}", "vertaal deze pagina naar het {lang}",
           "wat kan ik vanavond koken met {food}", "help me een sollicitatiebrief te schrijven voor een baan als {job}",
           "maak een paklijst voor {c} in {month}", "schrijf een antwoord aan {n} over de factuur",
           "is een {p} het waard? voor- en nadelen", "zoek de goedkoopste {p} in mijn tabbladen", "herschrijf dit wat formeler",
           "waarom is de lucht blauw", "herinner me eraan om {n} te bellen", "geef me 5 cadeau-ideeën voor {n}"],
    "de": ["schreib eine höfliche E-Mail an {n}, dass ich am {d} nicht kann", "fasse diese Seite zusammen",
           "erkläre mir wie {topic} funktioniert", "vergleiche {p} und {p2}", "plane eine {k}-tägige Reise nach {c}",
           "übersetze diese Seite ins {lang}", "was kann ich heute mit {food} kochen", "schreib ein Anschreiben für eine Stelle als {job}"],
    "fr": ["écris un e-mail poli à {n} pour dire que je ne peux pas {d}", "résume cette page", "explique-moi comment fonctionne {topic}",
           "compare {p} et {p2}", "organise un voyage de {k} jours à {c}", "traduis cette page en {lang}",
           "que puis-je cuisiner ce soir avec {food}", "aide-moi à écrire une lettre de motivation pour un poste de {job}"],
    "es": ["escribe un correo educado a {n} diciendo que no puedo el {d}", "resume esta página", "explícame cómo funciona {topic}",
           "compara {p} y {p2}", "planifica un viaje de {k} días a {c}", "traduce esta página al {lang}",
           "qué puedo cocinar esta noche con {food}", "ayúdame a escribir una carta de presentación para un puesto de {job}"],
}
TOPICS = ["compound interest", "a mortgage", "solar panels", "a heat pump", "inflation", "machine learning", "photosynthesis",
          "the electoral college", "vaccines", "blockchain", "a VPN", "income tax", "the stock market", "jet lag", "sourdough"]
FOODS = ["rice and eggs", "pasta and spinach", "chicken and potatoes", "tofu", "leftover vegetables", "lentils", "salmon"]
JOBS = ["product manager", "nurse", "teacher", "data analyst", "chef", "developer", "accountant", "designer"]
LANG_NAMES = {"en": ["German", "Spanish", "Dutch", "French", "English"], "nl": ["Engels", "Duits", "Spaans", "Frans"],
              "de": ["Englische", "Niederländische", "Französische"], "fr": ["anglais", "allemand", "espagnol"],
              "es": ["inglés", "alemán", "francés"]}
DAYS = {"en": ["Monday", "Tuesday", "Friday", "the 12th"], "nl": ["maandag", "dinsdag", "vrijdag"],
        "de": ["Montag", "Freitag"], "fr": ["venir lundi", "venir vendredi"], "es": ["lunes", "viernes"]}
MONTHS = ["March", "May", "July", "October", "December"]


def _fill(rng: random.Random, template: str, code: str) -> str:
    p, p2 = rng.sample(PRODUCTS, 2)
    c, c2 = rng.sample(CITIES, 2)
    t, t2 = rng.sample(TOPICS, 2)
    return template.format(p=p, p2=p2, b=rng.choice(BRANDS), c=c, c2=c2, s=rng.choice(SITES), person=rng.choice(PEOPLE),
                           n=rng.choice(NAMES), d=rng.choice(DAYS[code]), topic=t, topic2=t2, k=rng.randint(2, 12),
                           lang=rng.choice(LANG_NAMES[code]), food=rng.choice(FOODS), job=rng.choice(JOBS),
                           month=rng.choice(MONTHS))


def search_query(rng: random.Random, code: str) -> str:
    return _fill(rng, rng.choice(SEARCH_T[code]), code)


def ask_query(rng: random.Random, code: str) -> str:
    return _fill(rng, rng.choice(ASK_T[code]), code)


# ---------------------------------------------------------------------------------------------------- agent replies

REPLY_DONE = {
    "en": ["Done! I {did}.", "I {did}. Let me know if you'd like any changes.", "All set: I {did}. Anything else?",
           "I {did}. Want me to do anything else?", "Here you go: I {did}.", "I {did}. Should I also {more}?",
           "Finished. I {did}, and everything is saved."],
    "nl": ["Klaar! Ik heb {did_nl}.", "Ik heb {did_nl}. Laat het weten als je iets wilt aanpassen.", "Gedaan: ik heb {did_nl}. Nog iets?",
           "Ik heb {did_nl}. Zal ik ook {more_nl}?"],
    "de": ["Fertig! Ich habe {did_de}.", "Ich habe {did_de}. Sag Bescheid, wenn du Änderungen möchtest."],
    "fr": ["C'est fait ! J'ai {did_fr}.", "J'ai {did_fr}. Dis-moi si tu veux des modifications."],
    "es": ["¡Listo! He {did_es}.", "He {did_es}. Dime si quieres algún cambio."],
}
REPLY_BLOCKED = {
    "en": ["I found {k} options. Which one should I pick: {opts}?", "Before I send it, should it go to {n} or to the whole team?",
           "I need your {thing} to continue. What is it?", "The form asks for a {thing}. Tell me and I'll fill it in.",
           "There are two accounts signed in. Which one should I use?", "Do you want me to go ahead and book it for €{k}9?",
           "I can't find a {thing} on the page. Where should I look?", "Pick a date between Monday and Friday and I'll continue."],
    "nl": ["Ik heb {k} opties gevonden. Welke wil je: {opts}?", "Zal ik het naar {n} sturen of naar het hele team?",
           "Ik heb je {thing_nl} nodig om verder te gaan. Wat is het?", "Moet ik het boeken voor €{k}9?"],
    "de": ["Ich habe {k} Optionen gefunden. Welche soll ich nehmen: {opts}?", "Ich brauche deine {thing_de}, um weiterzumachen."],
    "fr": ["J'ai trouvé {k} options. Laquelle dois-je choisir : {opts} ?", "J'ai besoin de ton {thing_fr} pour continuer."],
    "es": ["He encontrado {k} opciones. ¿Cuál elijo: {opts}?", "Necesito tu {thing_es} para continuar."],
}
DID = {"en": ["added the meeting to your calendar", "summarised the article", "grouped your tabs", "filled in the form",
              "translated the page", "sent the email to Sam", "saved the PDF to Downloads", "made a table of the prices"],
       "nl": ["de afspraak in je agenda gezet", "het artikel samengevat", "je tabbladen gegroepeerd", "het formulier ingevuld"],
       "de": ["den Termin eingetragen", "den Artikel zusammengefasst", "das Formular ausgefüllt"],
       "fr": ["ajouté le rendez-vous", "résumé l'article", "rempli le formulaire"],
       "es": ["añadido la cita", "resumido el artículo", "rellenado el formulario"]}
MORE = {"en": ["translate the PDF", "set a reminder", "share it with the team"], "nl": ["een herinnering zetten", "het delen"]}
THINGS = {"en": ["order number", "postcode", "delivery date", "phone number", "budget"],
          "nl": ["ordernummer", "postcode", "leverdatum"], "de": ["Bestellnummer", "Postleitzahl"],
          "fr": ["numéro de commande", "code postal"], "es": ["número de pedido", "código postal"]}


def agent_reply(rng: random.Random, code: str) -> str:
    blocked = rng.random() < 0.5
    template = rng.choice((REPLY_BLOCKED if blocked else REPLY_DONE)[code])
    opts = ", ".join(rng.sample(["the 8:05", "the 12:40", "the 18:15", "the blue one", "the cheaper one", "Hotel Amstel",
                                 "the Krasnapolsky", "option A", "option B"], 2))
    text = template.format(k=rng.randint(2, 6), opts=opts, n=rng.choice(NAMES),
                           thing=rng.choice(THINGS["en"]), thing_nl=rng.choice(THINGS["nl"]), thing_de=rng.choice(THINGS["de"]),
                           thing_fr=rng.choice(THINGS["fr"]), thing_es=rng.choice(THINGS["es"]),
                           did=rng.choice(DID["en"]), did_nl=rng.choice(DID["nl"]), did_de=rng.choice(DID["de"]),
                           did_fr=rng.choice(DID["fr"]), did_es=rng.choice(DID["es"]),
                           more=rng.choice(MORE["en"]), more_nl=rng.choice(MORE["nl"]))
    if rng.random() < 0.5:  # a body before the closing line, as real replies have
        body = rng.choice(["Here's what I found:\n- {a}\n- {b}\n\n", "I checked the page and compared the prices.\n\n",
                           "The article argues that remote work raises productivity, with two caveats.\n\n", ""])
        text = body.format(a=rng.choice(PRODUCTS), b=rng.choice(PRODUCTS)) + text
    return text


# ---------------------------------------------------------------------------------------------------- chat messages

SMALL = {"en": ["thanks!", "ok", "great, thank you", "yes", "no thanks", "hi", "that's all", "perfect", "cool", "yes go ahead",
                "ok do it", "never mind", "good morning!", "how are you?"],
         "nl": ["dank je!", "oké", "top, bedankt", "ja", "nee hoor", "hoi", "dat was het", "prima", "ja doe maar", "goedemorgen!",
                "hoe gaat het?"],
         "de": ["danke!", "ok", "ja", "nein danke", "hallo", "passt", "ja mach"],
         "fr": ["merci !", "ok", "oui", "non merci", "salut", "parfait", "vas-y"],
         "es": ["¡gracias!", "vale", "sí", "no gracias", "hola", "perfecto", "adelante"]}
PREVIOUS = {"en": ["I can research all {k} suppliers and draft an email to the best three. Should I go ahead?",
                   "I've saved the document to your Downloads folder.", "Here are three hotels near the station.",
                   "Do you want the short version or the detailed comparison?"],
            "nl": ["Ik kan alle {k} leveranciers vergelijken en een mail opstellen. Zal ik beginnen?",
                   "Ik heb het document opgeslagen.", "Hier zijn drie hotels bij het station."]}


def chat_message(rng: random.Random, code: str) -> tuple[str, str]:
    r = rng.random()
    previous = ""
    if r < 0.35:
        message = rng.choice(SMALL[code])
        if rng.random() < 0.6:
            previous = rng.choice(PREVIOUS.get(code, PREVIOUS["en"])).format(k=rng.randint(3, 15))
    elif r < 0.65:
        message = ask_query(rng, code)
    elif r < 0.8:
        message = search_query(rng, code) + rng.choice(["?", "", " please"])
    else:
        a, b = ask_query(rng, code), ask_query(rng, code)
        message = f"{a}. Then {b}, and put everything in a table with sources." if code == "en" else f"{a}. {b}."
    return message, previous


MEMORY_FACTS = {"en": ["I'm vegetarian", "my daughter {n} starts school in September", "I work at a bank in {c}",
                       "I prefer short answers without bullet points", "I'm allergic to nuts", "we're moving to {c} next month",
                       "my manager is {n}", "I'm training for a marathon in {month}", "always write emails in British English"],
                "nl": ["ik ben vegetariër", "mijn dochter {n} gaat in september naar school", "ik werk bij een bank in {c}",
                       "ik wil korte antwoorden zonder opsommingstekens", "ik heb een notenallergie", "we verhuizen volgende maand naar {c}"]}


def memory_chat(rng: random.Random, code: str) -> str:
    if rng.random() < 0.5:
        q = search_query(rng, code) if rng.random() < 0.5 else ask_query(rng, code)
        a = rng.choice(["Here you go: {p} costs about €{k}9.", "It will be 14°C with light rain.", "Done, I summarised it in five points.",
                        "Lima.", "30 USD is about 27.60 EUR."]).format(p=rng.choice(PRODUCTS), k=rng.randint(1, 9))
    else:
        fact = rng.choice(MEMORY_FACTS.get(code, MEMORY_FACTS["en"])).format(n=rng.choice(NAMES), c=rng.choice(CITIES),
                                                                         month=rng.choice(MONTHS))
        q = f"{fact}, {ask_query(rng, code)}"
        a = "Got it. " + rng.choice(["I'll keep that in mind.", "Here's a first draft.", "Noted for next time."])
    return f"You: {q}\nCompanion: {a}"


# ---------------------------------------------------------------------------------------------------- voice

FIELD_LABELS = {"en": ["Search", "Comment", "Message", "Subject", "To", "Reply", "Notes", "Description", "Name", "Address", "Title"],
                "nl": ["Zoeken", "Reactie", "Bericht", "Onderwerp", "Aan", "Notities", "Omschrijving", "Naam", "Adres"],
                "de": ["Suche", "Kommentar", "Nachricht", "Betreff", "An"], "fr": ["Rechercher", "Commentaire", "Message", "Objet"],
                "es": ["Buscar", "Comentario", "Mensaje", "Asunto"]}
EDITS = {"en": ["make this shorter", "make it more formal", "translate it into {lang}", "fix the spelling", "add a friendly sign-off",
                "rewrite this politely", "turn this into bullet points", "make it sound less angry", "delete the last sentence",
                "summarise what I wrote", "change the date to {d}"],
         "nl": ["maak dit korter", "maak het formeler", "vertaal het naar het {lang}", "verbeter de spelling", "voeg een groet toe",
                "herschrijf dit beleefder", "verwijder de laatste zin"],
         "de": ["mach das kürzer", "formeller bitte", "übersetze es ins {lang}", "korrigiere die Rechtschreibung"],
         "fr": ["raccourcis ça", "rends-le plus formel", "traduis-le en {lang}", "corrige l'orthographe"],
         "es": ["hazlo más corto", "hazlo más formal", "tradúcelo al {lang}", "corrige la ortografía"]}
DICTATED = {"en": ["Hi {n}, thanks for the quick reply, {d} works for me.", "The package arrived damaged and I would like a replacement.",
                   "and please bring the slides from last week", "Great article, thanks for sharing!", "See you tomorrow at nine.",
                   "cheap flights to {c} in {month}", "I'll be ten minutes late, sorry!", "Can we move our call to {d}?",
                   "best regards, {n}", "the meeting is about the budget for next year"],
         "nl": ["Hoi {n}, bedankt voor je snelle reactie, {d} komt goed uit.", "Het pakket is beschadigd aangekomen.",
                "groetjes, {n}", "tot morgen!", "ik ben tien minuten later, sorry", "goedkope vluchten naar {c}"],
         "de": ["Hallo {n}, danke für die schnelle Antwort.", "bis morgen!", "viele Grüße, {n}"],
         "fr": ["Bonjour {n}, merci pour ta réponse rapide.", "à demain !", "cordialement, {n}"],
         "es": ["Hola {n}, gracias por la respuesta rápida.", "¡hasta mañana!", "saludos, {n}"]}
FIELD_TEXTS = {"en": ["Thank you so much for getting back to me so quickly.", "Teh meeting is tomorow", "hey can u send the file",
                      "Here are the numbers you asked for.", "See you tomorrow at nine."],
               "nl": ["Bedankt voor je snelle reactie.", "Tot morgen!", "hoi kun je het bestand sturen"],
               "de": ["Danke für deine Nachricht.", "Bis morgen!"], "fr": ["Merci pour ton message.", "À demain !"],
               "es": ["Gracias por tu mensaje.", "¡Hasta mañana!"]}


def _f(rng, template, code):
    return template.format(n=rng.choice(NAMES), d=rng.choice(DAYS[code]), c=rng.choice(CITIES), month=rng.choice(MONTHS),
                           lang=rng.choice(LANG_NAMES[code]))


def edit_request(rng, code):
    return _f(rng, rng.choice(EDITS[code]), code)


def dictated_text(rng, code):
    return _f(rng, rng.choice(DICTATED[code]), code)


def field_text(rng, code):
    return rng.choice(FIELD_TEXTS[code])


# ---------------------------------------------------------------------------------------------------- controls

CONTROLS = {
    "none": {"en": ["Next", "Continue", "Show more", "Filters", "Search", "Open", "Details", "Back", "Read more", "Sort by price",
                    "Load more results", "Menu", "Expand", "Settings", "View cart", "Learn more", "Close", "Previous page"],
             "nl": ["Volgende", "Verder", "Meer tonen", "Zoeken", "Openen", "Terug", "Lees meer", "Sorteren", "Winkelwagen bekijken"],
             "de": ["Weiter", "Mehr anzeigen", "Suchen", "Zurück", "Details", "Filter"],
             "fr": ["Suivant", "Continuer", "Afficher plus", "Rechercher", "Retour", "Filtres"],
             "es": ["Siguiente", "Continuar", "Ver más", "Buscar", "Volver", "Filtros"]},
    "purchase": {"en": ["Place order", "Buy now", "Complete purchase", "Subscribe", "Book now", "Upgrade to Pro", "Donate €10",
                        "Confirm booking", "Start free trial", "Add to cart and checkout"],
                 "nl": ["Bestelling plaatsen", "Nu kopen", "Afrekenen en betalen", "Abonneren", "Nu boeken", "Doneren"],
                 "de": ["Jetzt kaufen", "Zahlungspflichtig bestellen", "Abonnieren", "Jetzt buchen"],
                 "fr": ["Commander", "Acheter maintenant", "S'abonner", "Réserver"],
                 "es": ["Comprar ahora", "Realizar pedido", "Suscribirse", "Reservar"]},
    "payment": {"en": ["Pay now", "Send money", "Transfer", "Pay €49.95", "Confirm payment", "Pay invoice"],
                "nl": ["Overmaken", "Nu betalen", "Betaling bevestigen", "Geld versturen", "Tikkie betalen"],
                "de": ["Überweisen", "Jetzt bezahlen", "Zahlung bestätigen"], "fr": ["Payer", "Virement", "Confirmer le paiement"],
                "es": ["Pagar", "Transferir", "Confirmar pago"]},
    "delete": {"en": ["Delete", "Remove", "Delete file", "Empty trash", "Remove from list", "Delete all", "Discard draft"],
               "nl": ["Verwijderen", "Wissen", "Prullenbak legen", "Alles verwijderen"], "de": ["Löschen", "Entfernen"],
               "fr": ["Supprimer", "Retirer"], "es": ["Eliminar", "Borrar"]},
    "account": {"en": ["Delete account", "Cancel subscription", "Close account", "Create account", "End membership"],
                "nl": ["Account verwijderen", "Abonnement opzeggen", "Account aanmaken", "Lidmaatschap beëindigen"],
                "de": ["Konto löschen", "Abo kündigen"], "fr": ["Supprimer le compte", "Résilier l'abonnement"],
                "es": ["Eliminar cuenta", "Cancelar suscripción"]},
    "message": {"en": ["Send", "Send message", "Reply", "Send email", "Reply all", "Send invite"],
                "nl": ["Versturen", "Verzenden", "Beantwoorden", "Bericht sturen"], "de": ["Senden", "Antworten"],
                "fr": ["Envoyer", "Répondre"], "es": ["Enviar", "Responder"]},
    "post": {"en": ["Post", "Publish", "Tweet", "Share publicly", "Submit review", "Go live"],
             "nl": ["Plaatsen", "Publiceren", "Delen", "Review plaatsen"], "de": ["Veröffentlichen", "Posten"],
             "fr": ["Publier", "Partager"], "es": ["Publicar", "Compartir"]},
    "submit": {"en": ["Submit", "I agree", "Accept terms", "Confirm", "Sign contract", "Submit application"],
               "nl": ["Indienen", "Akkoord", "Voorwaarden accepteren", "Bevestigen", "Aanvraag versturen"],
               "de": ["Absenden", "Zustimmen", "Bestätigen"], "fr": ["Soumettre", "J'accepte", "Valider"],
               "es": ["Enviar solicitud", "Acepto", "Confirmar"]},
    "security": {"en": ["Change password", "Turn off two-factor", "Sign out all other sessions", "Remove passkey", "Update recovery email"],
                 "nl": ["Wachtwoord wijzigen", "Tweestapsverificatie uitzetten", "Overal uitloggen"],
                 "de": ["Passwort ändern", "Zwei-Faktor deaktivieren"], "fr": ["Changer le mot de passe"],
                 "es": ["Cambiar contraseña"]},
}
PAGES = {"none": ["Search results", "Product page", "Hotels in {c}", "News", "Shipping details", "Home"],
         "purchase": ["Checkout", "Cart", "Plans and pricing", "Booking summary"], "payment": ["Payments", "Invoice", "Transfer money"],
         "delete": ["Files", "Inbox", "Photos", "Drafts"], "account": ["Account settings", "Subscription", "Membership"],
         "message": ["New message", "Compose", "Inbox"], "post": ["New post", "Write a review", "Create story"],
         "submit": ["Application form", "Terms of service", "Contract"], "security": ["Security", "Password and sign-in"]}
DOMAINS = {"en": ["shop.example", "mail.example.com", "bank.example", "app.example.io", "social.example"],
           "nl": ["winkel.example.nl", "mijn.bank.example.nl", "mail.example.nl"], "de": ["shop.example.de", "bank.example.de"],
           "fr": ["boutique.example.fr", "app.example.fr"], "es": ["tienda.example.es", "mail.example.es"]}


def control(rng: random.Random, code: str) -> tuple[str, str, str, str]:
    kind = rng.choices(list(CONTROLS), weights=[4, 2, 1, 1, 1, 1, 1, 1, 1])[0]
    name = rng.choice(CONTROLS[kind].get(code) or CONTROLS[kind]["en"])
    # Ambiguity on purpose: sometimes the page or site says something else than the label.
    page_kind = kind if rng.random() < 0.8 else rng.choice(list(PAGES))
    page = rng.choice(PAGES[page_kind]).format(c=rng.choice(CITIES))
    role = rng.choice(["button", "button", "link", "menuitem"])
    url = f"https://{rng.choice(DOMAINS[code])}/{page_kind}"
    return name, role, page, url


# ---------------------------------------------------------------------------------------------------- tabs

TAB_SITES = [
    ("www.amazon.nl", "{p} - Amazon.nl"), ("www.bol.com", "{p} | bol.com"), ("www.coolblue.nl", "{p} - Coolblue"),
    ("mail.google.com", "Inbox (3) - {n}@company.com - Gmail"), ("docs.google.com", "Q3 roadmap - Google Docs"),
    ("www.bbc.com", "{headline} - BBC News"), ("nos.nl", "{headline_nl} | NOS"), ("www.youtube.com", "{video} - YouTube"),
    ("www.booking.com", "Hotels in {c} - Booking.com"), ("www.klm.nl", "Flights Amsterdam to {c} | KLM"),
    ("github.com", "{repo} at main · {org}"), ("stackoverflow.com", "How to {code_q} - Stack Overflow"),
    ("en.wikipedia.org", "{topic} - Wikipedia"), ("nl.wikipedia.org", "{topic} - Wikipedia"),
    ("www.bbcgoodfood.com", "Easy {dish} recipe | BBC Good Food"), ("www.ah.nl", "{dish} recept | Allerhande"),
    ("mijn.ing.nl", "Mijn ING - Overzicht betaalrekening"), ("www.linkedin.com", "{n} | LinkedIn"),
    ("www.reddit.com", "r/{sub}: {headline}"), ("open.spotify.com", "{video} - Spotify"), ("www.netflix.com", "Netflix"),
    ("www.funda.nl", "Huis te koop: {street} in {c} | Funda"), ("www.thuisbezorgd.nl", "Pizza bestellen in {c} | Thuisbezorgd"),
    ("www.ikea.com", "{furniture} - IKEA"), ("www.coursera.org", "{topic} course | Coursera"), ("www.notion.so", "Project plan"),
    ("app.slack.com", "general (Channel) - {org} - Slack"), ("www.thuisarts.nl", "{health} | Thuisarts.nl"),
    ("www.strava.com", "Morning run | Strava"), ("www.marktplaats.nl", "{p} | Marktplaats"), ("www.belastingdienst.nl", "Mijn Belastingdienst"),
]
HEADLINES = ["Storm hits the coast as thousands lose power", "Election results: what we know so far", "Central bank holds rates",
             "New study links sleep to memory", "Train strike announced for Friday"]
HEADLINES_NL = ["Storm raast over het land", "Kabinet presenteert begroting", "Treinstaking vrijdag"]
VIDEOS = ["lofi hip hop radio - beats to relax/study to", "How to fix a leaking tap", "Champions League highlights", "Daily mix 1"]
DISHES = ["vegetarian lasagne", "banana bread", "pad thai", "stamppot", "shakshuka"]
FURNITURE = ["METOD kitchen cabinets", "BILLY bookcase", "MALM bed frame", "KALLAX shelf"]
HEALTH = ["Rugpijn", "Griep", "Hoofdpijn", "Slapeloosheid"]
GROUPS = {"en": ["Lisbon trip", "Kitchen renovation", "Thesis", "Wedding", "Job search", "Q3 planning", "House hunting",
                 "Birthday party", "Tax return", "New laptop"],
          "nl": ["Reis Lissabon", "Verbouwing keuken", "Scriptie", "Bruiloft", "Sollicitaties", "Huis zoeken", "Belastingaangifte"],
          "de": ["Urlaub Lissabon", "Küche", "Masterarbeit", "Hochzeit"], "fr": ["Voyage Lisbonne", "Cuisine", "Mémoire"],
          "es": ["Viaje Lisboa", "Cocina", "Tesis"]}


def tab(rng: random.Random, code: str) -> tuple[str, str]:
    host, template = rng.choice(TAB_SITES)
    title = template.format(p=rng.choice(PRODUCTS), n=rng.choice(NAMES).lower(), headline=rng.choice(HEADLINES),
                            headline_nl=rng.choice(HEADLINES_NL), video=rng.choice(VIDEOS), c=rng.choice(CITIES),
                            repo=rng.choice(["src/index.ts", "README.md", "packages/core/src/query.ts"]),
                            org=rng.choice(["TanStack/query", "vercel/next.js", "acme/api"]),
                            code_q=rng.choice(["merge two dicts in python", "center a div", "undo a git commit"]),
                            topic=rng.choice(TOPICS).title(), dish=rng.choice(DISHES), sub=rng.choice(["Netherlands", "cooking", "personalfinance"]),
                            street=rng.choice(["Keizersgracht 12", "Dorpsstraat 4"]), furniture=rng.choice(FURNITURE),
                            health=rng.choice(HEALTH))
    return title, f"https://{host}/"


# ---------------------------------------------------------------------------------------------------- skills

SKILLS = [
    {"name": "weekly-report", "description": "Writes my weekly status report from my calendar and sent mail, in our team template."},
    {"name": "expense-claim", "description": "Fills in an expense claim on the company portal from receipts I attach."},
    {"name": "recipe-plan", "description": "Plans a week of vegetarian dinners with a shopping list."},
    {"name": "linkedin-post", "description": "Drafts a LinkedIn post in my voice about a topic I give."},
    {"name": "pr-review", "description": "Reviews a GitHub pull request for bugs and style, and leaves comments."},
    {"name": "travel-pack", "description": "Makes a packing list for a trip from the destination, dates and weather."},
    {"name": "invoice-chase", "description": "Finds unpaid invoices in my accounting tool and drafts polite reminder emails."},
    {"name": "meeting-notes", "description": "Turns a meeting transcript into notes with decisions and action items."},
    {"name": "price-watch", "description": "Checks the price of a product on a few shops every morning and tells me when it drops."},
    {"name": "inbox-zero", "description": "Sorts my inbox into to-do, waiting and read later, and drafts short replies."},
    {"name": "page-summary", "description": "Summarises the page I'm on in five bullet points."},
    {"name": "compare-options", "description": "Compares products or offers from my open tabs in a table."},
    {"name": "job-apply", "description": "Tailors my CV and a cover letter to a job ad I open."},
    {"name": "flight-search", "description": "Searches flights on a few airline sites and lists the cheapest with baggage rules."},
    {"name": "translate-page", "description": "Translates the current page into Dutch, keeping the layout."},
    {"name": "budget-check", "description": "Reads my bank export and shows where my money went this month."},
    {"name": "study-cards", "description": "Turns lecture notes into flashcards for spaced repetition."},
    {"name": "social-calendar", "description": "Plans a month of social media posts for my bakery."},
]
SKILL_ASKS = {"en": ["can you {d}?", "please {d}", "I need you to {d}", "{d} for me"],
              "nl": ["kun je {d}?", "wil je {d}", "{d} graag"]}


def skill_message(rng: random.Random, skill: dict, code: str) -> str:
    gist = skill["description"].rstrip(".").split(",")[0]
    gist = gist[0].lower() + gist[1:]
    gist = gist.replace("Writes ", "write ").replace("my ", "my ")
    for a, b in (("writes", "write"), ("fills", "fill"), ("plans", "plan"), ("drafts", "draft"), ("reviews", "review"),
                 ("makes", "make"), ("finds", "find"), ("turns", "turn"), ("checks", "check"), ("sorts", "sort"),
                 ("summarises", "summarise"), ("compares", "compare"), ("tailors", "tailor"), ("searches", "search"),
                 ("translates", "translate"), ("reads", "read")):
        if gist.startswith(a):
            gist = b + gist[len(a):]
    return rng.choice(SKILL_ASKS.get(code, SKILL_ASKS["en"])).format(d=gist)


# ---------------------------------------------------------------------------------------------------- label_items

EMAIL_SENDERS = ["Priya (my manager)", "Mum", "bol.com", "Vattenfall", "The Verge", "Jonas", "HR", "Zalando", "Ziggo",
                 "Acme (a client)", "Belastingdienst", "KLM", "Lenny's Newsletter", "the school", "Coolblue", "{n}"]
EMAIL_BODIES = ["Can you send me the slides before the meeting today?", "Your order #{k}512 has shipped.",
                "Your bill of €{k}4.20 is due today.", "This week in tech: ten stories you missed.", "Are we still on for dinner tonight?",
                "Reminder: submit your timesheet by the end of the day.", "Your refund of €{k}9.95 will arrive within 5 days.",
                "The production site is down. Please call us as soon as possible.", "Lovely to see you on Sunday!",
                "Your boarding pass for tomorrow is ready.", "Je voorlopige aanslag staat klaar. Betaal vóór 1 februari.",
                "Kun je vandaag nog even reageren op de offerte?", "Bitte bestätige den Termin bis morgen."]
PRODUCT_ITEMS = ["{p} - €{k}99 - 4.5 stars - delivered tomorrow", "{p} (refurbished) - €{k}49 - 3.9 stars - ships in 2 weeks",
                 "{p} bundle - €{k}79 - 4.8 stars - out of stock"]
LABEL_FAMILIES = [
    {"name": "email", "context": "Emails in someone's inbox, each with its sender and first lines.",
     "questions": {"topic": {"type": "choice", "instructions": "What is this email mainly about?",
                             "criteria": {"work": "Their job: colleagues, clients, tasks, meetings.", "personal": "Friends and family.",
                                          "shopping": "Orders, deliveries, returns and refunds.",
                                          "bills": "Invoices, payments due, taxes, subscriptions to pay.",
                                          "newsletter": "Newsletters, digests and marketing they subscribed to."}},
                   "today": {"type": "noul", "instructions": "Does this email ask the person to reply or act today?"}},
     "item": lambda rng, code: f"From: {rng.choice(EMAIL_SENDERS).format(n=rng.choice(NAMES))}. "
                               f"{rng.choice(EMAIL_BODIES).format(k=rng.randint(1, 9))}"},
    {"name": "product", "context": "Search results on a shop, each with name, price, rating and delivery.",
     "questions": {"fits": {"type": "noul", "instructions": "Can this be delivered within three days for under €200?"},
                   "quality": {"type": "score", "instructions": "How good a buy does this look?",
                               "criteria": ["Poor", "Okay", "Good", "Excellent"]}},
     "item": lambda rng, code: rng.choice(PRODUCT_ITEMS).format(p=rng.choice(PRODUCTS), k=rng.randint(0, 3))},
    {"name": "headline", "context": "News headlines from someone's feed.",
     "questions": {"section": {"type": "choice", "instructions": "Which section does this belong in?",
                               "criteria": {"politics": None, "economy": None, "science": None, "sport": None,
                                            "culture": None, "local": None}},
                   "urgent": {"type": "noul", "instructions": "Is this breaking news that affects people today?"}},
     "item": lambda rng, code: rng.choice(HEADLINES + HEADLINES_NL)},
]
