# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Browser flows generated with their answers: every page of a shop order, a search, a form, an account area, a
cookie banner, in English, Dutch, German, French and Spanish, and at every page the right next step, known because
the flow was built that way. No model labels anything.

Mind2Web (step/mind2web.py) teaches which element real people used on real sites. It can't teach the rest of the
runtime's step: when the goal is visibly done, when the flow is blocked (a login wall, a required value nobody gave,
an error), and which of the person's values goes into which field of a form in five languages. These flows do.

Each example is one page in one flow:
    {goal, values: [{name, value}], before: [what was done], page, elements: [{role, label, value?, options?, checked?,
     context?}], op: click|type|done|blocked, gold: element index or None, fields: {element index: value index or -1},
     lang, family, split}
`fields` holds the right value for every empty field on the page (index into values, -1: none of them), so one page
teaches the whole form at once. Elements are what the runtime's describe() shows (runtime/node/src/process/steps.ts).

Families are held out whole for the test (HELD_OUT), and so is one language per family, so the test asks about
kinds of flows and wordings the model never saw.

    python -I step/synth.py --out data/cache/synth-steps.jsonl --flows 6000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LANGS = ["en", "nl", "de", "fr", "es"]
LANG_WEIGHTS = [0.4, 0.2, 0.15, 0.13, 0.12]

# ── words per language ──────────────────────────────────────────────────────────────────────────────────────────────
# Each entry: several ways sites write the same thing.
W: dict[str, dict[str, list[str]]] = {
    "en": {
        "first name": ["First name", "Given name", "First name *"], "last name": ["Last name", "Surname", "Family name", "Last name *"],
        "full name": ["Full name", "Name", "Your name", "Name on the order"], "email address": ["Email", "Email address", "E-mail", "Your email *"],
        "phone number": ["Phone", "Phone number", "Mobile number", "Telephone (optional)"], "street": ["Street", "Street name"],
        "house number": ["House number", "No.", "Number"], "street and house number": ["Address", "Street and house number", "Address line 1", "Street address"],
        "postcode": ["Postcode", "ZIP code", "Postal code", "ZIP / Postcode"], "city": ["City", "Town", "Town or city"], "country": ["Country"],
        "date of birth": ["Date of birth", "Birthday (dd-mm-yyyy)"], "date": ["Date", "When did it happen?", "Preferred date"],
        "message": ["Message", "Your message", "Description", "Tell us more", "Comments"], "company": ["Company (optional)", "Company name"],
        "order number": ["Order number", "Order no."], "coupon": ["Coupon code", "Discount code"], "password": ["Password"],
        "search": ["Search", "Search products", "What are you looking for?", "Search the site"],
        "size": ["Size", "Size (EU)", "Choose your size"], "quantity": ["Quantity", "Qty"], "subject": ["Subject", "What is it about?", "Topic"],
        "b.search": ["Search", "Go", "Find"], "b.add": ["Add to cart", "Add to basket", "Add to bag"], "b.wish": ["Add to wishlist", "Save for later"],
        "b.cart": ["Cart", "Basket", "View cart", "Go to cart"], "b.checkout": ["Checkout", "Proceed to checkout", "Go to checkout"],
        "b.continue": ["Continue", "Next", "Continue to payment", "Next step"], "b.back": ["Back", "Previous"], "b.pay": ["Pay now", "Place order", "Pay €{amount}", "Confirm and pay"],
        "b.submit": ["Submit", "Send", "Send message", "Submit request"], "b.signin": ["Sign in", "Log in"], "b.signup": ["Create account", "Register"],
        "b.reject": ["Reject all", "Only necessary cookies", "Decline all"], "b.accept": ["Accept all", "Accept cookies", "Allow all"], "b.settings": ["Cookie settings", "Manage options", "Customise"],
        "b.save": ["Save", "Save changes", "Save preferences"], "b.view": ["View order", "Details", "Show order"], "b.invoice": ["Download invoice", "Invoice (PDF)"],
        "b.remove": ["Remove", "Delete"], "b.filter": ["Apply filters", "Show results"],
        "n.home": ["Home"], "n.shop": ["Shop", "Products", "All products"], "n.sale": ["Sale", "Offers"], "n.about": ["About us"], "n.contact": ["Contact", "Customer service"],
        "n.account": ["My account", "Account"], "n.orders": ["My orders", "Orders", "Order history"], "n.settings": ["Account settings", "Settings"],
        "n.notifications": ["Email preferences", "Notifications", "Email & notifications"], "n.addresses": ["Address book", "Addresses"], "n.help": ["Help", "FAQ"],
        "n.returns": ["Returns", "Returns & refunds"], "n.privacy": ["Privacy", "Privacy policy"], "n.terms": ["Terms"], "n.careers": ["Careers", "Jobs"], "n.blog": ["Blog", "Stories"],
        "t.added": ["Added to your cart", "Item added to your basket", "Added to your bag"], "t.thanks": ["Thank you! Your message has been sent.", "Thanks, we received your request."],
        "t.payment": ["Payment", "Choose how to pay", "Step 3 of 3: payment"], "t.required": ["Please fill in all required fields.", "This field is required."],
        "t.login": ["Sign in to continue", "Please log in to see your orders"], "t.stock": ["Sold out", "Out of stock"], "t.declined": ["Your card was declined."],
        "t.cookies": ["We use cookies to improve your experience.", "This site uses cookies."], "t.results": ["{n} results for “{q}”", "Results for “{q}”"],
        "t.saved": ["Your changes have been saved.", "Preferences saved."], "t.robot": ["Please confirm you are not a robot"],
        "c.robot": ["I'm not a robot"], "c.terms": ["I agree to the terms and conditions"], "c.news": ["Send me the newsletter", "Keep me updated by email"],
        "c.marketing": ["Marketing emails: on"], "c.updates": ["Email me updates about my request"],
        "r.standard": ["Standard delivery (free)", "Standard shipping (2–3 days)"], "r.express": ["Express delivery (€ 6.95)", "Next-day delivery"], "r.pickup": ["Pick up in store"],
        "sel.choose": ["Choose", "Select…", "Please choose"],
    },
    "nl": {
        "first name": ["Voornaam", "Voornaam *"], "last name": ["Achternaam", "Achternaam *"], "full name": ["Naam", "Volledige naam"],
        "email address": ["E-mailadres", "E-mail", "Uw e-mailadres *"], "phone number": ["Telefoonnummer", "Mobiel nummer", "Telefoon (niet verplicht)"],
        "street": ["Straat", "Straatnaam"], "house number": ["Huisnummer", "Nr."], "street and house number": ["Adres", "Straat en huisnummer"],
        "postcode": ["Postcode"], "city": ["Plaats", "Woonplaats"], "country": ["Land"], "date of birth": ["Geboortedatum", "Geboortedatum (dd-mm-jjjj)"],
        "date": ["Datum", "Wanneer was dit?", "Gewenste datum"], "message": ["Bericht", "Uw bericht", "Omschrijving", "Toelichting"], "company": ["Bedrijfsnaam (optioneel)"],
        "order number": ["Bestelnummer"], "coupon": ["Kortingscode"], "password": ["Wachtwoord"], "search": ["Zoeken", "Zoek producten", "Waar ben je naar op zoek?"],
        "size": ["Maat", "Kies je maat"], "quantity": ["Aantal"], "subject": ["Onderwerp", "Waar gaat het over?"],
        "b.search": ["Zoeken", "Zoek"], "b.add": ["In winkelwagen", "Toevoegen aan winkelmand", "In mijn winkelmandje"], "b.wish": ["Op verlanglijst", "Bewaren"],
        "b.cart": ["Winkelwagen", "Naar winkelwagen", "Bekijk winkelmand"], "b.checkout": ["Afrekenen", "Verder naar bestellen", "Bestelling afronden"],
        "b.continue": ["Volgende", "Verder", "Ga verder", "Verder naar betalen"], "b.back": ["Vorige", "Terug"], "b.pay": ["Nu betalen", "Bestelling plaatsen", "Betaal € {amount}"],
        "b.submit": ["Versturen", "Verzenden", "Melding versturen"], "b.signin": ["Inloggen"], "b.signup": ["Account aanmaken", "Registreren"],
        "b.reject": ["Alles weigeren", "Alleen noodzakelijke cookies", "Weigeren"], "b.accept": ["Alles accepteren", "Akkoord"], "b.settings": ["Cookie-instellingen", "Voorkeuren instellen"],
        "b.save": ["Opslaan", "Wijzigingen opslaan"], "b.view": ["Bestelling bekijken", "Details"], "b.invoice": ["Factuur downloaden", "Factuur (pdf)"],
        "b.remove": ["Verwijderen"], "b.filter": ["Filters toepassen", "Toon resultaten"],
        "n.home": ["Home", "Startpagina"], "n.shop": ["Winkel", "Producten"], "n.sale": ["Uitverkoop", "Aanbiedingen"], "n.about": ["Over ons"], "n.contact": ["Contact", "Klantenservice"],
        "n.account": ["Mijn account", "Account"], "n.orders": ["Mijn bestellingen", "Bestellingen"], "n.settings": ["Accountinstellingen", "Instellingen"],
        "n.notifications": ["E-mailvoorkeuren", "Meldingen"], "n.addresses": ["Adresboek", "Mijn adressen"], "n.help": ["Help", "Veelgestelde vragen"], "n.returns": ["Retourneren"],
        "n.privacy": ["Privacy", "Privacyverklaring"], "n.terms": ["Voorwaarden"], "n.careers": ["Vacatures"], "n.blog": ["Blog", "Inspiratie"],
        "t.added": ["Toegevoegd aan je winkelwagen", "Het artikel zit in je winkelmand"], "t.thanks": ["Bedankt! Uw melding is verstuurd.", "Dank je, we hebben je bericht ontvangen."],
        "t.payment": ["Betalen", "Kies een betaalmethode", "Stap 3 van 3: betalen"], "t.required": ["Vul alle verplichte velden in.", "Dit veld is verplicht."],
        "t.login": ["Log in om verder te gaan", "Log in om je bestellingen te zien"], "t.stock": ["Uitverkocht", "Niet op voorraad"], "t.declined": ["Je betaling is geweigerd."],
        "t.cookies": ["Wij gebruiken cookies om de site te verbeteren.", "Deze website gebruikt cookies."], "t.results": ["{n} resultaten voor “{q}”", "Resultaten voor “{q}”"],
        "t.saved": ["Je wijzigingen zijn opgeslagen.", "Voorkeuren opgeslagen."], "t.robot": ["Bevestig dat je geen robot bent"],
        "c.robot": ["Ik ben geen robot"], "c.terms": ["Ik ga akkoord met de voorwaarden"], "c.news": ["Stuur mij de nieuwsbrief", "Houd mij op de hoogte per e-mail"],
        "c.marketing": ["Marketing-e-mails: aan"], "c.updates": ["Houd mij per e-mail op de hoogte van mijn melding"],
        "r.standard": ["Standaard bezorging (gratis)", "Standaard verzending (2–3 dagen)"], "r.express": ["Expresbezorging (€ 6,95)", "Morgen in huis"], "r.pickup": ["Ophalen in de winkel"],
        "sel.choose": ["Kies", "Maak een keuze", "Kies een onderwerp"],
    },
    "de": {
        "first name": ["Vorname", "Vorname *"], "last name": ["Nachname", "Nachname *"], "full name": ["Name", "Vollständiger Name"],
        "email address": ["E-Mail", "E-Mail-Adresse", "Ihre E-Mail *"], "phone number": ["Telefon", "Telefonnummer", "Mobilnummer (optional)"],
        "street": ["Straße"], "house number": ["Hausnummer", "Nr."], "street and house number": ["Adresse", "Straße und Hausnummer"],
        "postcode": ["PLZ", "Postleitzahl"], "city": ["Ort", "Stadt", "Wohnort"], "country": ["Land"], "date of birth": ["Geburtsdatum"],
        "date": ["Datum", "Wann war das?", "Wunschtermin"], "message": ["Nachricht", "Ihre Nachricht", "Beschreibung"], "company": ["Firma (optional)"],
        "order number": ["Bestellnummer"], "coupon": ["Gutscheincode"], "password": ["Passwort"], "search": ["Suche", "Produkte suchen", "Wonach suchen Sie?"],
        "size": ["Größe", "Größe wählen"], "quantity": ["Menge", "Anzahl"], "subject": ["Betreff", "Worum geht es?"],
        "b.search": ["Suchen", "Los"], "b.add": ["In den Warenkorb", "Zum Warenkorb hinzufügen"], "b.wish": ["Auf die Wunschliste", "Merken"],
        "b.cart": ["Warenkorb", "Zum Warenkorb"], "b.checkout": ["Zur Kasse", "Weiter zur Kasse"], "b.continue": ["Weiter", "Nächster Schritt", "Weiter zur Zahlung"],
        "b.back": ["Zurück"], "b.pay": ["Jetzt bezahlen", "Zahlungspflichtig bestellen", "€ {amount} bezahlen"], "b.submit": ["Absenden", "Senden", "Nachricht senden"],
        "b.signin": ["Anmelden", "Einloggen"], "b.signup": ["Konto erstellen", "Registrieren"], "b.reject": ["Alle ablehnen", "Nur notwendige Cookies"],
        "b.accept": ["Alle akzeptieren", "Zustimmen"], "b.settings": ["Einstellungen", "Cookie-Einstellungen"], "b.save": ["Speichern", "Änderungen speichern"],
        "b.view": ["Bestellung ansehen", "Details"], "b.invoice": ["Rechnung herunterladen", "Rechnung (PDF)"], "b.remove": ["Entfernen", "Löschen"], "b.filter": ["Filter anwenden", "Ergebnisse anzeigen"],
        "n.home": ["Startseite"], "n.shop": ["Shop", "Produkte"], "n.sale": ["Sale", "Angebote"], "n.about": ["Über uns"], "n.contact": ["Kontakt", "Kundenservice"],
        "n.account": ["Mein Konto", "Konto"], "n.orders": ["Meine Bestellungen", "Bestellungen"], "n.settings": ["Kontoeinstellungen", "Einstellungen"],
        "n.notifications": ["E-Mail-Einstellungen", "Benachrichtigungen"], "n.addresses": ["Adressbuch"], "n.help": ["Hilfe", "FAQ"], "n.returns": ["Rücksendungen"],
        "n.privacy": ["Datenschutz"], "n.terms": ["AGB"], "n.careers": ["Karriere"], "n.blog": ["Magazin"],
        "t.added": ["Zum Warenkorb hinzugefügt", "Der Artikel liegt im Warenkorb"], "t.thanks": ["Vielen Dank! Ihre Nachricht wurde gesendet."],
        "t.payment": ["Zahlung", "Zahlungsart wählen", "Schritt 3 von 3: Zahlung"], "t.required": ["Bitte füllen Sie alle Pflichtfelder aus.", "Dieses Feld ist erforderlich."],
        "t.login": ["Bitte melden Sie sich an", "Melden Sie sich an, um Ihre Bestellungen zu sehen"], "t.stock": ["Ausverkauft", "Nicht vorrätig"], "t.declined": ["Ihre Zahlung wurde abgelehnt."],
        "t.cookies": ["Wir verwenden Cookies.", "Diese Website verwendet Cookies."], "t.results": ["{n} Ergebnisse für „{q}“", "Ergebnisse für „{q}“"],
        "t.saved": ["Ihre Änderungen wurden gespeichert."], "t.robot": ["Bitte bestätigen Sie, dass Sie kein Roboter sind"],
        "c.robot": ["Ich bin kein Roboter"], "c.terms": ["Ich akzeptiere die AGB"], "c.news": ["Newsletter abonnieren", "Per E-Mail informieren"],
        "c.marketing": ["Marketing-E-Mails: an"], "c.updates": ["Per E-Mail über meine Meldung informieren"],
        "r.standard": ["Standardversand (kostenlos)"], "r.express": ["Expressversand (€ 6,95)"], "r.pickup": ["Abholung im Geschäft"], "sel.choose": ["Bitte wählen", "Auswählen"],
    },
    "fr": {
        "first name": ["Prénom", "Prénom *"], "last name": ["Nom", "Nom de famille"], "full name": ["Nom complet", "Nom et prénom"],
        "email address": ["E-mail", "Adresse e-mail", "Votre e-mail *"], "phone number": ["Téléphone", "Numéro de portable"], "street": ["Rue"],
        "house number": ["Numéro"], "street and house number": ["Adresse", "Numéro et rue"], "postcode": ["Code postal"], "city": ["Ville"], "country": ["Pays"],
        "date of birth": ["Date de naissance"], "date": ["Date", "Quand cela s'est-il produit ?"], "message": ["Message", "Votre message", "Description"],
        "company": ["Société (facultatif)"], "order number": ["Numéro de commande"], "coupon": ["Code promo"], "password": ["Mot de passe"],
        "search": ["Rechercher", "Que cherchez-vous ?"], "size": ["Taille", "Pointure"], "quantity": ["Quantité"], "subject": ["Objet", "Sujet"],
        "b.search": ["Rechercher", "OK"], "b.add": ["Ajouter au panier"], "b.wish": ["Ajouter aux favoris"], "b.cart": ["Panier", "Voir le panier"],
        "b.checkout": ["Commander", "Passer la commande", "Valider mon panier"], "b.continue": ["Continuer", "Suivant", "Continuer vers le paiement"], "b.back": ["Retour", "Précédent"],
        "b.pay": ["Payer maintenant", "Valider et payer", "Payer {amount} €"], "b.submit": ["Envoyer", "Valider", "Envoyer le message"], "b.signin": ["Se connecter", "Connexion"],
        "b.signup": ["Créer un compte"], "b.reject": ["Tout refuser", "Continuer sans accepter"], "b.accept": ["Tout accepter", "Accepter"], "b.settings": ["Paramétrer", "Personnaliser"],
        "b.save": ["Enregistrer"], "b.view": ["Voir la commande", "Détails"], "b.invoice": ["Télécharger la facture"], "b.remove": ["Supprimer"], "b.filter": ["Appliquer les filtres"],
        "n.home": ["Accueil"], "n.shop": ["Boutique", "Produits"], "n.sale": ["Soldes", "Promotions"], "n.about": ["Qui sommes-nous"], "n.contact": ["Contact", "Service client"],
        "n.account": ["Mon compte"], "n.orders": ["Mes commandes", "Commandes"], "n.settings": ["Paramètres du compte"], "n.notifications": ["Préférences e-mail", "Notifications"],
        "n.addresses": ["Carnet d'adresses"], "n.help": ["Aide", "FAQ"], "n.returns": ["Retours"], "n.privacy": ["Confidentialité"], "n.terms": ["CGV"], "n.careers": ["Recrutement"], "n.blog": ["Blog"],
        "t.added": ["Ajouté à votre panier", "L'article a été ajouté au panier"], "t.thanks": ["Merci ! Votre message a été envoyé."], "t.payment": ["Paiement", "Choisissez votre moyen de paiement"],
        "t.required": ["Veuillez remplir tous les champs obligatoires.", "Ce champ est obligatoire."], "t.login": ["Connectez-vous pour continuer"], "t.stock": ["Épuisé", "Rupture de stock"],
        "t.declined": ["Votre paiement a été refusé."], "t.cookies": ["Nous utilisons des cookies.", "Ce site utilise des cookies."], "t.results": ["{n} résultats pour « {q} »"],
        "t.saved": ["Vos modifications ont été enregistrées."], "t.robot": ["Confirmez que vous n'êtes pas un robot"],
        "c.robot": ["Je ne suis pas un robot"], "c.terms": ["J'accepte les conditions générales"], "c.news": ["Recevoir la newsletter"], "c.marketing": ["E-mails marketing : activés"],
        "c.updates": ["Me tenir informé par e-mail"], "r.standard": ["Livraison standard (gratuite)"], "r.express": ["Livraison express (6,95 €)"], "r.pickup": ["Retrait en magasin"],
        "sel.choose": ["Choisir", "Sélectionnez"],
    },
    "es": {
        "first name": ["Nombre", "Nombre *"], "last name": ["Apellidos", "Apellido"], "full name": ["Nombre completo"], "email address": ["Correo electrónico", "Email", "Tu email *"],
        "phone number": ["Teléfono", "Móvil"], "street": ["Calle"], "house number": ["Número"], "street and house number": ["Dirección", "Calle y número"],
        "postcode": ["Código postal"], "city": ["Ciudad", "Localidad"], "country": ["País"], "date of birth": ["Fecha de nacimiento"], "date": ["Fecha", "¿Cuándo ocurrió?"],
        "message": ["Mensaje", "Tu mensaje", "Descripción"], "company": ["Empresa (opcional)"], "order number": ["Número de pedido"], "coupon": ["Código de descuento"],
        "password": ["Contraseña"], "search": ["Buscar", "¿Qué estás buscando?"], "size": ["Talla", "Elige tu talla"], "quantity": ["Cantidad"], "subject": ["Asunto", "Motivo"],
        "b.search": ["Buscar"], "b.add": ["Añadir al carrito", "Añadir a la cesta"], "b.wish": ["Añadir a favoritos"], "b.cart": ["Carrito", "Ver carrito", "Cesta"],
        "b.checkout": ["Tramitar pedido", "Finalizar compra"], "b.continue": ["Continuar", "Siguiente", "Continuar al pago"], "b.back": ["Volver", "Anterior"],
        "b.pay": ["Pagar ahora", "Realizar pedido", "Pagar {amount} €"], "b.submit": ["Enviar", "Enviar mensaje"], "b.signin": ["Iniciar sesión", "Entrar"], "b.signup": ["Crear cuenta", "Registrarse"],
        "b.reject": ["Rechazar todo", "Solo cookies necesarias"], "b.accept": ["Aceptar todo", "Aceptar"], "b.settings": ["Configurar", "Preferencias de cookies"], "b.save": ["Guardar", "Guardar cambios"],
        "b.view": ["Ver pedido", "Detalles"], "b.invoice": ["Descargar factura"], "b.remove": ["Eliminar"], "b.filter": ["Aplicar filtros"],
        "n.home": ["Inicio"], "n.shop": ["Tienda", "Productos"], "n.sale": ["Rebajas", "Ofertas"], "n.about": ["Quiénes somos"], "n.contact": ["Contacto", "Atención al cliente"],
        "n.account": ["Mi cuenta"], "n.orders": ["Mis pedidos", "Pedidos"], "n.settings": ["Ajustes de la cuenta"], "n.notifications": ["Preferencias de correo", "Notificaciones"],
        "n.addresses": ["Mis direcciones"], "n.help": ["Ayuda"], "n.returns": ["Devoluciones"], "n.privacy": ["Privacidad"], "n.terms": ["Condiciones"], "n.careers": ["Empleo"], "n.blog": ["Blog"],
        "t.added": ["Añadido a tu carrito", "El artículo se ha añadido a la cesta"], "t.thanks": ["¡Gracias! Tu mensaje se ha enviado."], "t.payment": ["Pago", "Elige cómo pagar"],
        "t.required": ["Rellena todos los campos obligatorios.", "Este campo es obligatorio."], "t.login": ["Inicia sesión para continuar"], "t.stock": ["Agotado", "Sin existencias"],
        "t.declined": ["Tu pago ha sido rechazado."], "t.cookies": ["Usamos cookies.", "Este sitio utiliza cookies."], "t.results": ["{n} resultados para «{q}»"],
        "t.saved": ["Tus cambios se han guardado."], "t.robot": ["Confirma que no eres un robot"], "c.robot": ["No soy un robot"], "c.terms": ["Acepto las condiciones"],
        "c.news": ["Quiero recibir la newsletter"], "c.marketing": ["Correos de marketing: activados"], "c.updates": ["Avisarme por correo"],
        "r.standard": ["Envío estándar (gratis)"], "r.express": ["Envío urgente (6,95 €)"], "r.pickup": ["Recogida en tienda"], "sel.choose": ["Elige", "Selecciona"],
    },
}

# People are generated for every flow (names, streets, postcodes and phones as each country writes them), so the model
# matches a field to a value by what the value is, never by a person it has seen before. None is the lab's person.
FIRST = {
    "en": ["Oliver", "Amelia", "Jack", "Isla", "Harry", "Grace", "Leo", "Freya", "Noah", "Ruby", "Ethan", "Chloe", "Daniel", "Hannah", "Aaron", "Priya"],
    "nl": ["Daan", "Emma", "Sem", "Julia", "Lucas", "Tess", "Milan", "Sara", "Bram", "Lotte", "Thijs", "Fleur", "Jesse", "Iris", "Ruben", "Yara"],
    "de": ["Jonas", "Mia", "Felix", "Hannah", "Paul", "Lea", "Lukas", "Emilia", "Finn", "Clara", "Moritz", "Anna", "Tobias", "Marie", "Jan", "Sophie"],
    "fr": ["Louis", "Chloé", "Hugo", "Léa", "Jules", "Manon", "Arthur", "Inès", "Théo", "Zoé", "Nathan", "Jade", "Gabriel", "Lina", "Raphaël", "Alice"],
    "es": ["Hugo", "Lucía", "Martín", "Sofía", "Pablo", "Martina", "Mateo", "Valeria", "Diego", "Paula", "Álvaro", "Carmen", "Javier", "Elena", "Adrián", "Irene"],
}
LAST = {
    "en": ["Turner", "Patel", "Hughes", "Walsh", "Bennett", "Clarke", "Murphy", "Lewis", "Khan", "Fletcher", "Morgan", "Shaw"],
    "nl": ["de Vries", "Jansen", "Bakker", "Visser", "Smit", "Meijer", "de Boer", "Mulder", "Bos", "Vos", "Peters", "Hendriks"],
    "de": ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Becker", "Wagner", "Schulz", "Koch", "Richter", "Klein", "Wolf"],
    "fr": ["Bernard", "Dubois", "Thomas", "Robert", "Petit", "Durand", "Leroy", "Moreau", "Simon", "Laurent", "Lefebvre", "Michel"],
    "es": ["García", "Fernández", "López", "Martínez", "Sánchez", "Pérez", "Gómez", "Ruiz", "Díaz", "Moreno", "Muñoz", "Romero"],
}
STREETS = {
    "en": ["Station Road", "Church Lane", "Park Avenue", "Victoria Street", "Mill Road", "Queens Road", "Green Lane", "High Street"],
    "nl": ["Dorpsstraat", "Kerkstraat", "Molenweg", "Stationsweg", "Beukenlaan", "Prinsengracht", "Schoolstraat", "Wilhelminastraat"],
    "de": ["Hauptstraße", "Bahnhofstraße", "Schulstraße", "Lindenweg", "Bergstraße", "Kirchgasse", "Am Markt", "Rosenweg"],
    "fr": ["rue de la Paix", "avenue Victor Hugo", "rue du Moulin", "boulevard Voltaire", "rue des Écoles", "place de la Mairie", "rue Pasteur", "chemin des Vignes"],
    "es": ["Calle Real", "Avenida de la Constitución", "Calle del Sol", "Plaza Mayor", "Calle Nueva", "Paseo del Prado", "Calle de la Iglesia", "Camino Viejo"],
}
CITIES = {
    "en": ["Leeds", "Bristol", "York", "Norwich", "Cardiff", "Brighton"], "nl": ["Zwolle", "Haarlem", "Breda", "Leiden", "Groningen", "Arnhem"],
    "de": ["Münster", "Freiburg", "Leipzig", "Bremen", "Kassel", "Mainz"], "fr": ["Nantes", "Lille", "Rennes", "Dijon", "Tours", "Grenoble"],
    "es": ["Valencia", "Sevilla", "Bilbao", "Zaragoza", "Málaga", "Granada"],
}
COUNTRY = {"en": "United Kingdom", "nl": "Netherlands", "de": "Germany", "fr": "France", "es": "Spain"}
DOMAINS = ["mailpost.test", "inbox.test", "postvak.test", "courriel.test", "correo.test", "webmail.test"]


def person(rng: random.Random, lang: str) -> dict:
    first, last = rng.choice(FIRST[lang]), rng.choice(LAST[lang])
    letters = "ABCDEFGHJKLMNPRSTVWXZ"
    postcode = {
        "nl": f"{rng.randint(1000, 9999)} {rng.choice(letters)}{rng.choice(letters)}",
        "en": f"{rng.choice(letters)}{rng.choice(letters)}{rng.randint(1, 20)} {rng.randint(1, 9)}{rng.choice(letters)}{rng.choice(letters)}",
    }.get(lang, f"{rng.randint(1000, 99999):05d}")
    phone = {
        "nl": f"06 {rng.randint(10000000, 99999999)}", "en": f"07{rng.randint(100, 999)} {rng.randint(100000, 999999)}",
        "de": f"01{rng.randint(50, 79)} {rng.randint(1000000, 9999999)}", "fr": f"0{rng.choice([6, 7])} {' '.join(f'{rng.randint(0, 99):02d}' for _ in range(4))}",
        "es": f"6{rng.randint(10, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)}",
    }[lang]
    plain = lambda t: "".join(c for c in t.lower().replace(" ", "") if c.isalnum() and c.isascii())
    return {"first": first, "last": last, "email": f"{plain(first)}.{plain(last)}@{rng.choice(DOMAINS)}", "phone": phone, "street": rng.choice(STREETS[lang]),
            "no": str(rng.randint(1, 240)) + (rng.choice(["", "", "", "a", "B"]) if lang == "nl" else ""), "postcode": postcode, "city": rng.choice(CITIES[lang]),
            "country": COUNTRY[lang], "dob": f"{rng.randint(1, 28):02d}-{rng.randint(1, 12):02d}-{rng.randint(1950, 2004)}"}


PRODUCTS = {
    "en": [("Trail running shoes", "shoes"), ("Rain jacket", "clothes"), ("Hiking backpack 30 L", "plain"), ("2-person tent", "plain"), ("Merino T-shirt", "clothes"), ("Wireless headphones", "plain"), ("Coffee grinder", "plain"), ("Desk lamp", "plain"), ("Running socks (3 pairs)", "clothes"), ("Winter boots", "shoes")],
    "nl": [("Trailschoenen", "shoes"), ("Regenjas", "clothes"), ("Wandelrugzak 30 L", "plain"), ("Tweepersoonstent", "plain"), ("Merino T-shirt", "clothes"), ("Draadloze koptelefoon", "plain"), ("Koffiemolen", "plain"), ("Bureaulamp", "plain"), ("Hardloopsokken (3 paar)", "clothes"), ("Winterlaarzen", "shoes")],
    "de": [("Trailschuhe", "shoes"), ("Regenjacke", "clothes"), ("Wanderrucksack 30 L", "plain"), ("Zwei-Personen-Zelt", "plain"), ("Merino-T-Shirt", "clothes"), ("Kabellose Kopfhörer", "plain"), ("Kaffeemühle", "plain"), ("Schreibtischlampe", "plain"), ("Laufsocken (3 Paar)", "clothes"), ("Winterstiefel", "shoes")],
    "fr": [("Chaussures de trail", "shoes"), ("Veste de pluie", "clothes"), ("Sac à dos 30 L", "plain"), ("Tente 2 places", "plain"), ("T-shirt en mérinos", "clothes"), ("Casque sans fil", "plain"), ("Moulin à café", "plain"), ("Lampe de bureau", "plain"), ("Chaussettes de course (3 paires)", "clothes"), ("Bottes d'hiver", "shoes")],
    "es": [("Zapatillas de trail", "shoes"), ("Chaqueta impermeable", "clothes"), ("Mochila de 30 L", "plain"), ("Tienda para 2 personas", "plain"), ("Camiseta de merino", "clothes"), ("Auriculares inalámbricos", "plain"), ("Molinillo de café", "plain"), ("Lámpara de escritorio", "plain"), ("Calcetines de running (3 pares)", "clothes"), ("Botas de invierno", "shoes")],
}
BRANDS = ["Northway", "Kestrel", "Alder", "Brightline", "Corvana", "Tessel", "Fjordmark", "Lumen", "Orbis", "Pinecrest"]
SIZES = {"shoes": [str(s) for s in range(36, 47)], "clothes": ["XS", "S", "M", "L", "XL", "XXL"]}
SUBJECTS = {
    "en": ["Missed bin collection", "Broken street light", "Loose paving stones", "Fly-tipping", "Noise nuisance", "Something else"],
    "nl": ["Afval niet opgehaald", "Kapotte straatverlichting", "Losliggende stoeptegels", "Zwerfafval of dumping", "Overlast", "Iets anders"],
    "de": ["Müll nicht abgeholt", "Straßenlaterne defekt", "Lose Gehwegplatten", "Illegale Müllablagerung", "Lärmbelästigung", "Sonstiges"],
    "fr": ["Poubelle non ramassée", "Éclairage public en panne", "Pavés descellés", "Dépôt sauvage", "Nuisances sonores", "Autre"],
    "es": ["Basura no recogida", "Farola averiada", "Baldosas sueltas", "Vertido ilegal", "Ruido", "Otro"],
}
MESSAGES = {
    "en": "Our green bin on {street} {no} was not emptied on Tuesday.", "nl": "Onze groene container aan de {street} {no} is dinsdag niet geleegd.",
    "de": "Unsere Biotonne in der {street} {no} wurde am Dienstag nicht geleert.", "fr": "Notre bac vert au {no} {street} n'a pas été vidé mardi.",
    "es": "Nuestro contenedor verde en {street} {no} no se vació el martes.",
}
SITES = ["Northfield", "Brookhaven", "Lindeoord", "Waldheim", "Valmont", "Sierra Verde", "Kestrel Outdoor", "Hollowcraft", "Maple & Co", "Riverside Market"]

# Families: the kinds of flow. HELD_OUT families and the held-out language of every family form the test.
FAMILIES = ["shop-order", "shop-add", "search-open", "form-report", "form-contact", "form-signup", "account-invoice", "account-marketing", "cookies", "blocked-login", "blocked-missing", "blocked-error"]
HELD_OUT = {"form-signup", "account-marketing"}
HELD_OUT_LANG = {f: LANGS[1 + i % 4] for i, f in enumerate(FAMILIES)}  # one non-English language per family


def w(rng: random.Random, lang: str, key: str, **fmt) -> str:
    text = rng.choice(W[lang][key])
    return text.format(**fmt) if fmt else text


def el(role: str, label: str, **extra) -> dict:
    return {"role": role, "label": label, **{k: v for k, v in extra.items() if v is not None}}


def chrome(rng: random.Random, lang: str, site: str, cart: int) -> tuple[list[dict], list[dict]]:
    """Header and footer around any page: navigation, search, account, cart, footer links."""
    head = [el("link", site), el("field", w(rng, lang, "search"), value="")]
    head.append(el("button", w(rng, lang, "b.search")))
    nav = ["n.shop", "n.sale", "n.about", "n.contact", "n.blog"]
    for k in rng.sample(nav, rng.randint(2, 4)):
        head.append(el("link", w(rng, lang, k)))
    head += [el("link", w(rng, lang, "n.account")), el("link", f"{w(rng, lang, 'b.cart')} ({cart})")]
    foot = [el("link", w(rng, lang, k)) for k in rng.sample(["n.help", "n.returns", "n.privacy", "n.terms", "n.careers", "n.contact"], rng.randint(2, 5))]
    if rng.random() < 0.4:
        foot.insert(0, el("field", "Newsletter", value=""))
    return head, foot


# What a planner may call a value: the model must match a field to a value by what both mean, not by one fixed name.
NAME_VARIANTS = {
    "first name": ["first name", "given name", "first"], "last name": ["last name", "surname", "family name"],
    "full name": ["full name", "name", "your name", "customer name"], "email address": ["email address", "email", "e-mail", "contact email"],
    "phone number": ["phone number", "phone", "mobile number", "telephone"], "street": ["street", "street name"],
    "house number": ["house number", "number", "house no."], "street and house number": ["street and house number", "address", "street address", "home address", "address line 1"],
    "postcode": ["postcode", "zip code", "postal code", "post code"], "city": ["city", "town", "place of residence"], "country": ["country"],
    "date of birth": ["date of birth", "birthday", "born on"], "product to search for": ["product to search for", "product", "search term", "what to search for", "item"],
    "size": ["size", "shoe size", "EU size", "clothing size", "size to choose"], "delivery option": ["delivery option", "delivery", "shipping method", "delivery choice"],
    "subject": ["subject", "topic", "report type", "category", "what it is about"], "message": ["message", "description", "text for the form", "comments", "message text"],
    "date": ["date", "date of the problem", "when it happened", "collection date"],
}
# Values a planner may write down that belong in no field on these pages.
DISTRACTORS = [("quantity", "1"), ("budget", "under €200"), ("loyalty number", "LX-4471"), ("order number", "#1077"), ("preferred time", "after 17:00"),
               ("customer number", "100245"), ("colour", "dark green"), ("gift wrap", "no")]


def values_for(p: dict, lang: str, need: list[str], rng: random.Random, extra: list[tuple[str, str]] = ()) -> list[dict]:
    """The person's values the planner would write down, shuffled with the task's extras and a distractor or two.
    Each has a key (what it is, for the answers) and a name as a planner might write it: the usual name, another
    wording, or a bare "value N" so only its content tells what it is."""
    table = {"first name": p["first"], "last name": p["last"], "full name": f"{p['first']} {p['last']}", "email address": p["email"], "phone number": p["phone"],
             "street": p["street"], "house number": p["no"], "street and house number": f"{p['street']} {p['no']}" if lang not in ("fr",) else f"{p['no']} {p['street']}",
             "postcode": p["postcode"], "city": p["city"], "country": p["country"], "date of birth": p["dob"]}
    # Any amount of the person's details in any kind of flow (all of them a third of the time), so how many there are
    # never tells the model what kind of flow it is in.
    names = list(dict.fromkeys(need + (list(table) if rng.random() < 0.33 else rng.sample(list(table), rng.randint(2, 9)))))
    out = [{"key": n, "value": table[n]} for n in names if n in table] + [{"key": a, "value": b} for a, b in extra]
    out += [{"key": f"x-{a}", "name": a, "value": b} for a, b in rng.sample(DISTRACTORS, rng.randint(0, 2))]
    rng.shuffle(out)
    style = rng.random()
    used = set()
    for i, v in enumerate(out):
        if "name" in v:
            pass
        elif style < 0.35:
            v["name"] = v["key"]
        elif style < 0.8:
            v["name"] = rng.choice(NAME_VARIANTS.get(v["key"], [v["key"]]))
        else:
            v["name"] = f"value {i + 1}"
        while v["name"] in used:  # two keys may share a wording ("number"): keep names unique
            v["name"] = v["key"] if v["key"] not in used else f"{v['name']} ({i + 1})"
        used.add(v["name"])
    return out


PERSON_KEYS = {"first name", "last name", "full name", "email address", "phone number", "street", "house number", "street and house number", "postcode", "city", "country", "date of birth"}


def vindex(values: list[dict], key: str) -> int:
    return next((i for i, v in enumerate(values) if v.get("key", v["name"]) == key), -1)


class Flow:
    """Pages of one flow, each with its right next step."""

    def __init__(self, rng: random.Random, lang: str, family: str, goal: str, values: list[dict], site: str) -> None:
        self.rng, self.lang, self.family, self.goal, self.values, self.site = rng, lang, family, goal, values, site
        self.before: list[str] = []
        self.pages: list[dict] = []
        self.milestones: list[tuple[str, int]] = []
        self.cart = rng.randint(0, 2)

    def milestone(self, name: str) -> None:
        """The next page shows this milestone: a stretch goal that ends here is done on that page."""
        self.milestones.append((name, len(self.pages)))

    def page(self, title: str, content: list[dict], op: str, gold: dict | None, fields: dict[int, int] | None = None, did: str | None = None, bare: bool = False) -> None:
        head, foot = ([], []) if bare else chrome(self.rng, self.lang, self.site, self.cart)
        # The page's own controls win over the header's: no two controls read the same.
        sig = lambda e: (e["role"], e["label"].lower(), e.get("context"))
        mine = {sig(e) for e in content}
        head = [e for e in head if sig(e) not in mine and not (e["role"] == "field" and any(c["role"] == "field" and c["label"] == e["label"] for c in content))]
        foot = [e for e in foot if sig(e) not in mine]
        elements = head + content + foot
        g = None if gold is None else next(i for i, e in enumerate(elements) if e is gold)
        f = {}
        for i, e in enumerate(elements):
            for c_i, v in (fields or {}).items():
                if e is content[c_i]:
                    f[i] = v
        self.pages.append({"goal": self.goal, "values": self.values, "before": list(self.before[-4:]), "page": f"{title} – {self.site}", "elements": elements,
                           "op": op, "gold": g, "fields": f, "lang": self.lang, "family": self.family})
        if did:
            self.before.append(did)


def text_line(label: str) -> dict:
    """Plain text the runtime shows among elements (a notice, a heading): a non-clickable line."""
    return {"role": "text", "label": label}


def fill_form(flow: Flow, title: str, fields: list[tuple[str, str]], after: list[dict], next_btn: dict, extra_top: list[dict] = (), blocked_missing: str | None = None) -> None:
    """A form page: one example per field to type (the field and its value), then the button. fields: (kind, label)."""
    rng, lang = flow.rng, flow.lang
    elements = list(extra_top) + [el("field", label, value="") if kind != "subject" else el("select", label, value="", options=SUBJECTS[lang]) for kind, label in fields]
    elements += after + [next_btn]
    order = list(range(len(extra_top), len(extra_top) + len(fields)))
    missing = False
    for step, idx in enumerate(order):
        kind = fields[idx - len(extra_top)][0]
        f = {}
        for j in order[step:]:
            k = fields[j - len(extra_top)][0]
            f[j] = vindex(flow.values, k) if k != "subject" else vindex(flow.values, "subject")
        target = elements[idx]
        v = vindex(flow.values, kind) if kind != "subject" else vindex(flow.values, "subject")
        if v < 0:
            # Nothing to put in it: a required one blocks the form once everything else is filled; others stay empty.
            missing = missing or kind == blocked_missing
            continue
        flow.page(title, elements, "type", target, f, did=f"typed into {target['label']}")
        val = flow.values[v]["value"]
        if target["role"] == "select":
            elements[idx] = el("select", target["label"], value=next((o for o in target["options"] if o.lower() == val.lower()), val), options=target["options"])
        else:
            elements[idx] = el("field", target["label"], value=val)
    if missing:
        left = {j: -1 for j in order if elements[j].get("value", "") == ""}
        flow.page(title, elements, "blocked", None, left)
        return
    for k, a in enumerate(after):
        if a.get("tick"):
            flow.page(title, elements, "click", a, did=f"clicked {a['label']}")
            after_new = dict(a)
            after_new.pop("tick")
            after_new["checked"] = True
            elements[elements.index(a)] = after_new
    flow.page(title, elements, "click", next_btn, did=f"clicked {next_btn['label']}")


def product_name(rng: random.Random, lang: str) -> tuple[str, str]:
    name, kind = rng.choice(PRODUCTS[lang])
    return f"{rng.choice(BRANDS)} {name}", kind


def results_page(flow: Flow, query: str, wanted: str, kind: str) -> dict:
    rng, lang = flow.rng, flow.lang
    others = [product_name(rng, lang)[0] for _ in range(rng.randint(3, 8))]
    names = list(dict.fromkeys([wanted] + others))
    rng.shuffle(names)
    content = [text_line(w(rng, lang, "t.results", n=len(names), q=query))]
    target = None
    for n in names:
        price = f"€ {rng.randint(15, 320)},{rng.choice(['00', '95', '50'])}"
        link = el("link", n, context=price)
        content.append(link)
        if n == wanted:
            target = link
    content.append(el("button", w(rng, lang, "b.filter")))
    flow.page(f"{query}", content, "click", target, did=f"clicked {wanted}")
    return target


def product_page(flow: Flow, name: str, kind: str, size: str | None, stop_after_add: bool) -> None:
    rng, lang = flow.rng, flow.lang
    flow.milestone("product")
    price = f"€ {rng.randint(20, 300)},{rng.choice(['00', '95'])}"
    content = [text_line(name), text_line(price)]
    sel = el("select", w(rng, lang, "size"), value="", options=SIZES[kind]) if kind in SIZES else None
    if sel:
        content.append(sel)
    qty = el("field", w(rng, lang, "quantity"), value="1")
    add = el("button", w(rng, lang, "b.add"))
    wish = el("button", w(rng, lang, "b.wish"))
    content += [qty, add, wish]
    if sel and size:
        flow.page(name, content, "type", sel, {content.index(sel): vindex(flow.values, "size")}, did=f"typed into {sel['label']}")
        content[content.index(sel)] = el("select", sel["label"], value=size, options=sel["options"])
    flow.page(name, content, "click", add, did=f"clicked {add['label']}")
    flow.cart += 1
    notice = [text_line(w(rng, lang, "t.added") + f": {name}"), el("link", w(rng, lang, "b.cart")), el("link", w(rng, lang, "n.shop"))]
    flow.milestone("cart-added")
    if stop_after_add:
        flow.page(name, content + notice, "done", None)
    else:
        flow.page(name, content + notice, "click", notice[1], did=f"clicked {notice[1]['label']}")


def checkout(flow: Flow, name: str, delivery: str, pay: bool) -> None:
    rng, lang = flow.rng, flow.lang
    total = f"{rng.randint(30, 400)},{rng.choice(['00', '95'])}"
    flow.milestone("cart")
    cart_content = [text_line(f"{name} · 1 ×"), el("button", w(rng, lang, "b.remove")), el("field", w(rng, lang, "coupon"), value=""), el("link", w(rng, lang, "b.checkout"))]
    flow.page(w(rng, lang, "b.cart"), cart_content, "click", cart_content[3], did=f"clicked {cart_content[3]['label']}")
    one_line = rng.random() < 0.5
    names = ["full name"] if rng.random() < 0.5 else ["first name", "last name"]
    addr = ["street and house number"] if one_line else ["street", "house number"]
    kinds = names + ["email address"] + addr + ["postcode", "city"] + (["phone number"] if rng.random() < 0.6 else [])
    fields = [(k, w(rng, lang, k)) for k in kinds]
    radios = [el("radio", w(rng, lang, "r.standard"), checked=False), el("radio", w(rng, lang, "r.express"), checked=False)]
    if rng.random() < 0.4:
        radios.append(el("radio", w(rng, lang, "r.pickup"), checked=False))
    want = {"standard": 0, "express": 1}[delivery]
    radios[want]["tick"] = True
    news = el("checkbox", w(rng, lang, "c.news"), checked=False)
    btn = el("button", w(rng, lang, "b.continue"))
    flow.milestone("checkout-form")
    fill_form(flow, w(rng, lang, "b.checkout"), fields, radios + [news], btn)
    pay_btn = el("button", w(rng, lang, "b.pay", amount=total))
    content = [text_line(w(rng, lang, "t.payment")), el("radio", "iDEAL", checked=True), el("radio", "Card", checked=False), pay_btn, el("link", w(rng, lang, "b.back"))]
    flow.page(w(rng, lang, "t.payment"), content, "click" if pay else "done", pay_btn if pay else None, did="clicked pay" if pay else None)


# ── goals ──
GOALS = {
    "shop-order": ["Order the {product}{size_part} with {delivery} delivery to my home address. Don't pay: stop on the payment page.",
                   "Buy the {product}{size_part}, {delivery} delivery to my address, and stop on the payment page."],
    "shop-add": ["Find the {product}{size_part} and add it to the cart.", "Put the {product}{size_part} in my cart. Stop once it is in the cart."],
    "search-open": ["Search for {product} and open its product page.", "Find the {product} and open its page."],
    "form-report": ["Report to the municipality that our bin wasn't emptied. Fill in the form with my details; don't send it: stop on the last page.",
                    "Fill in the report form about the missed bin collection with my details and stop before sending."],
    "form-contact": ["Send the shop a message asking about my order: fill in the contact form and stop before sending.", "Fill in the contact form with my details and the message; don't send it."],
    "form-signup": ["Create an account with my details; stop before submitting.", "Make me an account here, but don't submit it yet.",
                    "Create an account with my details and sign me up for the newsletter too; stop before submitting."],
    "account-invoice": ["Get the invoice for my last order.", "Download the invoice of my most recent order."],
    "account-marketing": ["Stop the marketing emails.", "Turn off the marketing emails in my account."],
    "cookies": ["Refuse the cookies that aren't necessary, then search for {product}.", "Reject the cookie banner and find the {product}."],
    "blocked-login": ["Show me my orders.", "Open my order history."],
    "blocked-missing": ["Fill in the form to request a quote.", "Fill in the contact form and stop before sending."],
    "blocked-error": ["Order the {product} and stop on the payment page.", "Add the {product} to the cart."],
}


def flow_for(family: str, rng: random.Random, lang: str) -> Flow:
    p = person(rng, lang if rng.random() < 0.7 else rng.choice(LANGS))
    site = rng.choice(SITES)
    product, kind = product_name(rng, lang)
    size = rng.choice(SIZES[kind]) if kind in SIZES else None
    delivery = rng.choice(["standard", "express"])
    fmt = {"product": product, "size_part": f" in size {size}" if size else "", "delivery": delivery}
    goal = rng.choice(GOALS[family]).format(**fmt)
    if family in ("shop-order", "shop-add", "blocked-error"):
        extra = [("product to search for", product)] + ([("size", size)] if size else []) + ([("delivery option", f"{delivery} delivery")] if family == "shop-order" else [])
        values = values_for(p, lang, ["full name", "email address", "street and house number", "postcode", "city"] if family == "shop-order" else [], rng, extra)
        flow = Flow(rng, lang, family, goal, values, site)
        # home: type the search, press search
        search = el("field", w(rng, lang, "search"), value="")
        btn = el("button", w(rng, lang, "b.search"))
        teasers = [el("link", product_name(rng, lang)[0]) for _ in range(rng.randint(2, 5))]
        home = [search, btn] + teasers
        flow.page(site, home, "type", search, {0: vindex(values, "product to search for")}, did=f"typed into {search['label']}")
        home[0] = el("field", search["label"], value=product)
        flow.page(site, home, "click", btn, did=f"clicked {btn['label']}")
        results_page(flow, product, product, kind)
        if family == "blocked-error":
            content = [text_line(product), text_line(w(rng, lang, "t.stock")), el("button", w(rng, lang, "b.wish")), el("link", w(rng, lang, "n.shop"))]
            flow.page(product, content, "blocked", None)
            return flow
        product_page(flow, product, kind, size, stop_after_add=family == "shop-add")
        if family == "shop-order":
            checkout(flow, product, delivery, pay=False)
        return flow
    if family == "search-open":
        values = values_for(p, lang, [], rng, [("product to search for", product)])
        flow = Flow(rng, lang, family, goal, values, site)
        search = el("field", w(rng, lang, "search"), value="")
        btn = el("button", w(rng, lang, "b.search"))
        home = [search, btn] + [el("link", product_name(rng, lang)[0]) for _ in range(rng.randint(2, 5))]
        flow.page(site, home, "type", search, {0: vindex(values, "product to search for")}, did=f"typed into {search['label']}")
        home[0] = el("field", search["label"], value=product)
        flow.page(site, home, "click", btn, did=f"clicked {btn['label']}")
        results_page(flow, product, product, kind)
        content = [text_line(product), text_line(f"€ {rng.randint(20, 300)},00"), el("button", w(rng, lang, "b.add")), el("button", w(rng, lang, "b.wish"))]
        flow.page(product, content, "done", None)
        return flow
    if family in ("form-report", "form-contact", "blocked-missing"):
        subject = SUBJECTS[lang][0] if family == "form-report" else rng.choice(SUBJECTS[lang][:-1])
        message = MESSAGES[lang].format(street=p["street"], no=p["no"])
        extra = [("subject", subject), ("message", message)]
        if family == "form-report":
            extra.append(("date", rng.choice(["24-11-2026", "18-11-2026", "02-12-2026"])))
        values = values_for(p, lang, ["first name", "last name", "email address", "street", "house number", "postcode"], rng, extra)
        if family == "blocked-missing":
            values = [v for v in values if v["name"] != "phone number"]
        flow = Flow(rng, lang, family, goal, values, site)
        start = el("link", w(rng, lang, "n.contact") if family != "form-report" else {"en": "Report a problem", "nl": "Melding doorgeven", "de": "Schaden melden", "fr": "Signaler un problème", "es": "Comunicar una incidencia"}[lang])
        flow.page(site, [text_line(site), start, el("link", w(rng, lang, "n.help"))], "click", start, did=f"clicked {start['label']}")
        page1 = [("subject", w(rng, lang, "subject"))] + ([("date", w(rng, lang, "date"))] if family == "form-report" else []) + [("message", w(rng, lang, "message"))]
        if family == "form-report":
            page1 += [("street", w(rng, lang, "street")), ("house number", w(rng, lang, "house number")), ("postcode", w(rng, lang, "postcode"))]
            flow.milestone("form")
            fill_form(flow, start["label"], page1, [], el("button", w(rng, lang, "b.continue")))
            flow.milestone("next-form")
            page2 = [("first name", w(rng, lang, "first name")), ("last name", w(rng, lang, "last name")), ("email address", w(rng, lang, "email address")), ("phone number", w(rng, lang, "phone number"))]
            updates = el("checkbox", w(rng, lang, "c.updates"), checked=False)
            if rng.random() < 0.5:
                updates["tick"] = True
                flow.goal += " I'd like updates by email."
            fill_form(flow, start["label"], page2, [updates], el("button", w(rng, lang, "b.continue")))
            send = el("button", w(rng, lang, "b.submit"))
            flow.milestone("review")
            flow.page(start["label"], [text_line(subject), el("checkbox", w(rng, lang, "c.terms"), checked=False), send, el("link", w(rng, lang, "b.back"))], "done", None)
            return flow
        fields = [("first name", w(rng, lang, "first name")), ("last name", w(rng, lang, "last name")), ("email address", w(rng, lang, "email address"))]
        if family == "blocked-missing":
            label = w(rng, lang, "phone number").replace(" (optional)", "").replace(" (niet verplicht)", "").replace(" (optional)", "") + " *"
            fields.append(("phone number", label))
        fields += page1
        send = el("button", w(rng, lang, "b.submit"))
        flow.milestone("form")
        fill_form(flow, start["label"], fields, [], send, blocked_missing="phone number" if family == "blocked-missing" else None)
        if family == "form-contact":
            flow.pages[-1]["op"], flow.pages[-1]["gold"] = "done", None  # filled, not sent: the goal said stop before sending
            flow.before = flow.before[:-1]
        return flow
    if family == "form-signup":
        values = values_for(p, lang, ["first name", "last name", "email address"], rng)
        flow = Flow(rng, lang, family, goal, values, site)
        link = el("link", w(rng, lang, "b.signup"))
        flow.page(site, [text_line(site), link, el("link", w(rng, lang, "b.signin"))], "click", link, did=f"clicked {link['label']}")
        fields = [("first name", w(rng, lang, "first name")), ("last name", w(rng, lang, "last name")), ("email address", w(rng, lang, "email address"))]
        news = el("checkbox", w(rng, lang, "c.news"), checked=False)
        if "newsletter" in goal:
            news["tick"] = True  # only when the goal asks for it
        flow.milestone("form")
        fill_form(flow, link["label"], fields, [news], el("button", w(rng, lang, "b.signup")))
        flow.pages[-1]["op"], flow.pages[-1]["gold"] = "done", None
        flow.before = flow.before[:-1]
        return flow
    if family in ("account-invoice", "account-marketing", "blocked-login"):
        values = values_for(p, lang, [], rng)
        flow = Flow(rng, lang, family, goal, values, site)
        acct = el("link", w(rng, lang, "n.account"))
        flow.page(site, [text_line(site), acct, el("link", w(rng, lang, "n.help"))], "click", acct, did=f"clicked {acct['label']}")
        if family == "blocked-login":
            content = [text_line(w(rng, lang, "t.login")), el("field", w(rng, lang, "email address"), value=""), el("field", w(rng, lang, "password"), value=""), el("button", w(rng, lang, "b.signin")), el("link", w(rng, lang, "b.signup"))]
            flow.page(w(rng, lang, "b.signin"), content, "blocked", None)
            return flow
        links = {k: el("link", w(rng, lang, k)) for k in ["n.orders", "n.settings", "n.addresses", "n.returns"]}
        page = [text_line(f"{p['first']}")] + list(links.values()) + [el("button", w(rng, lang, "b.signin").replace("Sign in", "Sign out"))]
        if family == "account-invoice":
            flow.page(w(rng, lang, "n.account"), page, "click", links["n.orders"], did=f"clicked {links['n.orders']['label']}")
            orders = []
            for n, (no, date) in enumerate([("#1077", "19-11-2026"), ("#1043", "14-10-2026"), ("#0988", "02-08-2026")]):
                orders.append(el("link", w(rng, lang, "b.view"), context=f"{no} · {date}"))
            flow.milestone("orders")
            flow.page(links["n.orders"]["label"], orders + [el("link", w(rng, lang, "b.back"))], "click", orders[0], did=f"clicked {orders[0]['label']}")
            flow.milestone("order")
            inv = el("button", w(rng, lang, "b.invoice"))
            flow.page("#1077", [text_line("#1077"), inv, el("link", w(rng, lang, "n.returns"))], "click", inv, did=f"clicked {inv['label']}")
            flow.page("#1077", [text_line("#1077"), inv, el("link", w(rng, lang, "n.returns"))], "done", None)
            return flow
        flow.page(w(rng, lang, "n.account"), page, "click", links["n.settings"], did=f"clicked {links['n.settings']['label']}")
        notif = el("link", w(rng, lang, "n.notifications"))
        flow.page(links["n.settings"]["label"], [el("field", w(rng, lang, "full name"), value=f"{p['first']} {p['last']}"), notif, el("link", w(rng, lang, "n.addresses"))], "click", notif, did=f"clicked {notif['label']}")
        toggle = el("checkbox", w(rng, lang, "c.marketing"), checked=True)
        save = el("button", w(rng, lang, "b.save"))
        content = [toggle, el("checkbox", w(rng, lang, "c.news"), checked=True), save]
        flow.milestone("notifications")
        flow.page(notif["label"], content, "click", toggle, did=f"clicked {toggle['label']}")
        off = el("checkbox", toggle["label"].replace(": on", ": off").replace(": aan", ": uit").replace(": an", ": aus").replace(": activés", ": désactivés").replace(": activados", ": desactivados"), checked=False)
        content = [off, content[1], save]
        flow.page(notif["label"], content, "click", save, did=f"clicked {save['label']}")
        flow.page(notif["label"], [text_line(w(rng, lang, "t.saved"))] + content, "done", None)
        return flow
    if family == "cookies":
        values = values_for(p, lang, [], rng, [("product to search for", product)])
        flow = Flow(rng, lang, family, goal, values, site)
        reject = el("button", w(rng, lang, "b.reject"))
        banner = [text_line(w(rng, lang, "t.cookies")), el("button", w(rng, lang, "b.accept")), reject, el("link", w(rng, lang, "b.settings"))]
        rng.shuffle(banner[1:])
        flow.page(site, banner + [el("link", product_name(rng, lang)[0])], "click", reject, did=f"clicked {reject['label']}")
        search = el("field", w(rng, lang, "search"), value="")
        btn = el("button", w(rng, lang, "b.search"))
        home = [search, btn, el("link", product_name(rng, lang)[0])]
        flow.milestone("banner-gone")
        flow.page(site, home, "type", search, {0: vindex(values, "product to search for")}, did=f"typed into {search['label']}")
        home[0] = el("field", search["label"], value=product)
        flow.page(site, home, "click", btn, did=f"clicked {btn['label']}")
        return flow
    raise ValueError(family)


# Stretch goals as a planner writes them, one part of a longer task. Worded apart from the lab's processes
# (runtime/node/processes), which test the model on stretches it never saw word for word.
STRETCH = {
    "product": ["Look the product up and open its page. Stop there", "Get to the product's own page and stop", "Open the product page and go no further"],
    "cart-added": ["Put the product in the basket and stop when the site confirms it", "Add it to the cart, then stop", "Get the item into the cart; nothing more"],
    "cart": ["Open the cart and stop there", "Go to the shopping cart. Stop when it is showing"],
    "checkout-form": ["Head to the checkout and stop when the delivery form appears", "Get to the checkout page with the address form; go no further", "Proceed to checkout and stop at the delivery details"],
    "form": ["Get the right form on screen and stop", "Find the form and open it; nothing else yet", "Navigate to the form. Stop once it is open"],
    "next-form": ["Move to the following page of the form and stop there", "Press on to the form's next page; stop when it appears"],
    "review": ["Reach the summary page before sending and stop; never send", "Go through to the overview page. Do not submit"],
    "orders": ["Get to the list of orders and stop", "Open the order history; nothing more"],
    "order": ["Open the right order and stop on its page", "Show that order's details, then stop"],
    "notifications": ["Get to the email settings and stop", "Open the notification preferences; go no further"],
    "banner-gone": ["Deal with the cookie banner and stop once it is gone", "Refuse the cookies, then stop"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--flows", type=int, default=6000)
    parser.add_argument("--seed", default="2028-steps-synth")
    parser.add_argument("--stretch-share", type=float, default=0.6, help="share of milestones that also become a stretch goal")
    parser.add_argument("--again-share", type=float, default=0.35, help="share of click/done/blocked pages also shown with another person's details")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows, counts = [], {}
    for n in range(args.flows):
        family = FAMILIES[n % len(FAMILIES)]
        lang = rng.choices(LANGS, LANG_WEIGHTS)[0]
        flow = flow_for(family, random.Random(f"{args.seed}-{n}"), lang)
        split = "family-unseen" if family in HELD_OUT else "lang-unseen" if lang == HELD_OUT_LANG[family] else "train"
        values = [{"name": v["name"], "value": v["value"]} for v in flow.values]
        arng = random.Random(f"{args.seed}-again-{n}")
        for pg in flow.pages:
            for e in pg["elements"]:
                e.pop("tick", None)
            rows.append({**pg, "values": values, "split": split, "flow": n})
            # The same page with someone else's details: the answer must not move (clicks, done and blocked only,
            # where no field's value depends on the list).
            if not pg["fields"] and arng.random() < args.again_share:
                other = values_for(person(arng, arng.choice(LANGS)), lang, [], arng, [(v["key"], v["value"]) for v in flow.values if v["key"] not in PERSON_KEYS])
                rows.append({**pg, "values": [{"name": v["name"], "value": v["value"]} for v in other], "split": split, "flow": n, "again": True})
        # Stretch goals: one part of the task ("stop as soon as the form shows"), done on the page that shows it.
        prev = 0
        srng = random.Random(f"{args.seed}-stretch-{n}")
        for name, idx in flow.milestones:
            if idx >= len(flow.pages) or flow.pages[idx]["op"] in ("done", "blocked") or srng.random() > args.stretch_share:
                prev = min(idx, len(flow.pages) - 1)
                continue
            stretch = srng.choice(STRETCH[name])
            whole = flow.pages[idx]["goal"]
            # From where the stretch begins: the milestone before it, or one further back, or the start.
            earlier = [0] + [i for _, i in flow.milestones if i < idx]
            prev = srng.choice(earlier[-3:]) if srng.random() < 0.5 else min(prev, idx)
            goal = f"{stretch}: {whole}" if srng.random() < 0.7 else f"{stretch}."
            for k in range(prev, idx + 1):
                pg = flow.pages[k]
                row = {**pg, "goal": goal, "values": values, "split": split, "flow": n, "stretch": name}
                if k == idx:
                    row.update({"op": "done", "gold": None})
                rows.append(row)
            prev = idx
        counts[(split, family)] = counts.get((split, family), 0) + len(flow.pages)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    ops = {}
    for r in rows:
        ops[(r["split"], r["op"])] = ops.get((r["split"], r["op"]), 0) + 1
    print(f"{len(rows)} pages from {args.flows} flows")
    for k in sorted(ops):
        print(f"  {k[0]:<14} {k[1]:<8} {ops[k]}")


if __name__ == "__main__":
    main()
