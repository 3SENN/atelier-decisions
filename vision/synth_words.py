# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""The words vision/synth.py draws its documents and pages with, and what each kind is called when asked about, in
English, Dutch, German, French and Spanish. Written for this project by Claude (Anthropic), CC BY 4.0.
"""

from __future__ import annotations

LANGS = ["en", "nl", "de", "fr", "es"]

# kind | then for en, nl, de, fr, es: the kind with its article | one sentence describing it
_DOCUMENTS = """
receipt|a receipt|A till receipt from a shop or a restaurant|een kassabon|Een kassabon van een winkel of restaurant|ein Kassenbon|Ein Kassenbon aus einem Geschäft oder Restaurant|un ticket de caisse|Un ticket de caisse d'un magasin ou d'un restaurant|un tique de compra|Un tique de compra de una tienda o un restaurante
invoice|an invoice|An invoice from a company, with amounts to pay|een factuur|Een factuur van een bedrijf, met te betalen bedragen|eine Rechnung|Eine Rechnung von einer Firma mit zu zahlenden Beträgen|une facture|Une facture d'une entreprise, avec des montants à payer|una factura|Una factura de una empresa, con importes a pagar
boarding|a boarding pass|A boarding pass for a flight|een instapkaart|Een instapkaart voor een vlucht|eine Bordkarte|Eine Bordkarte für einen Flug|une carte d'embarquement|Une carte d'embarquement pour un vol|una tarjeta de embarque|Una tarjeta de embarque para un vuelo
ticket|a ticket|A ticket for an event or a journey|een toegangskaartje|Een kaartje voor een evenement of een reis|eine Eintrittskarte|Eine Eintrittskarte oder ein Fahrschein|un billet|Un billet pour un événement ou un voyage|una entrada|Una entrada o un billete para un evento o un viaje
letter|a letter|A printed letter|een brief|Een gedrukte brief|ein Brief|Ein gedruckter Brief|une lettre|Une lettre imprimée|una carta|Una carta impresa
idcard|an identity card|An identity card or another ID document|een identiteitskaart|Een identiteitskaart of een ander identiteitsbewijs|ein Personalausweis|Ein Personalausweis oder ein anderes Ausweisdokument|une carte d'identité|Une carte d'identité ou une autre pièce d'identité|un documento de identidad|Un documento de identidad o un carné
bizcard|a business card|A business card with a name and contact details|een visitekaartje|Een visitekaartje met naam en contactgegevens|eine Visitenkarte|Eine Visitenkarte mit Namen und Kontaktdaten|une carte de visite|Une carte de visite avec un nom et des coordonnées|una tarjeta de visita|Una tarjeta de visita con nombre y datos de contacto
menu|a restaurant menu|A restaurant's menu with dishes and prices|een menukaart|De menukaart van een restaurant met gerechten en prijzen|eine Speisekarte|Die Speisekarte eines Restaurants mit Gerichten und Preisen|un menu de restaurant|Le menu d'un restaurant avec des plats et des prix|una carta de restaurante|La carta de un restaurante con platos y precios
calendar|a calendar|A calendar with the days of a month|een kalender|Een kalender met de dagen van een maand|ein Kalender|Ein Kalender mit den Tagen eines Monats|un calendrier|Un calendrier avec les jours d'un mois|un calendario|Un calendario con los días de un mes
spreadsheet|a spreadsheet|A spreadsheet with rows and columns of cells|een spreadsheet|Een spreadsheet met rijen en kolommen|eine Tabellenkalkulation|Eine Tabellenkalkulation mit Zeilen und Spalten|une feuille de calcul|Une feuille de calcul avec des lignes et des colonnes|una hoja de cálculo|Una hoja de cálculo con filas y columnas
chart|a chart|A chart or graph of numbers|een grafiek|Een grafiek of diagram van cijfers|ein Diagramm|Ein Diagramm mit Zahlen|un graphique|Un graphique de chiffres|un gráfico|Un gráfico de cifras
slide|a presentation slide|A slide from a presentation|een presentatiedia|Een dia uit een presentatie|eine Präsentationsfolie|Eine Folie aus einer Präsentation|une diapositive|Une diapositive d'une présentation|una diapositiva|Una diapositiva de una presentación
code|computer code|Computer code in an editor|computercode|Computercode in een editor|Programmcode|Programmcode in einem Editor|du code informatique|Du code informatique dans un éditeur|código informático|Código informático en un editor
chat|a chat conversation|A screenshot of a chat conversation|een chatgesprek|Een schermafbeelding van een chatgesprek|ein Chatverlauf|Ein Bildschirmfoto eines Chatverlaufs|une conversation par messages|Une capture d'écran d'une conversation par messages|una conversación de chat|Una captura de una conversación de chat
note|a handwritten note|A handwritten note or list|een handgeschreven briefje|Een handgeschreven briefje of lijstje|eine handgeschriebene Notiz|Eine handgeschriebene Notiz oder Liste|une note manuscrite|Une note ou une liste écrite à la main|una nota escrita a mano|Una nota o una lista escrita a mano
statement|a bank statement|A bank statement with a list of transactions|een bankafschrift|Een bankafschrift met een lijst transacties|ein Kontoauszug|Ein Kontoauszug mit einer Liste von Buchungen|un relevé bancaire|Un relevé bancaire avec une liste d'opérations|un extracto bancario|Un extracto bancario con una lista de movimientos
label|a shipping label|A shipping label on a parcel, with an address and a barcode|een verzendlabel|Een verzendlabel van een pakket, met adres en streepjescode|ein Versandetikett|Ein Versandetikett eines Pakets mit Adresse und Strichcode|une étiquette d'expédition|Une étiquette d'expédition de colis, avec une adresse et un code-barres|una etiqueta de envío|Una etiqueta de envío de un paquete, con dirección y código de barras
certificate|a certificate|A certificate or diploma|een certificaat|Een certificaat of diploma|eine Urkunde|Eine Urkunde oder ein Zeugnis|un certificat|Un certificat ou un diplôme|un certificado|Un certificado o un diploma
form|a form to fill in|A paper form with boxes to fill in|een formulier|Een papieren formulier met vakjes om in te vullen|ein Formular|Ein Papierformular mit Feldern zum Ausfüllen|un formulaire|Un formulaire papier avec des cases à remplir|un formulario|Un formulario en papel con casillas para rellenar
map|a street map|A map of streets|een plattegrond|Een plattegrond met straten|ein Stadtplan|Ein Stadtplan mit Straßen|un plan de ville|Un plan avec des rues|un plano de calles|Un plano con calles
email|an email inbox|An email inbox with a list of messages|een e-mailinbox|Een e-mailinbox met een lijst berichten|ein E-Mail-Postfach|Ein E-Mail-Postfach mit einer Liste von Nachrichten|une boîte de réception|Une boîte de réception avec une liste de messages|una bandeja de entrada|Una bandeja de entrada de correo con una lista de mensajes
article|a newspaper page|A newspaper page with columns of text|een krantenpagina|Een krantenpagina met kolommen tekst|eine Zeitungsseite|Eine Zeitungsseite mit Textspalten|une page de journal|Une page de journal avec des colonnes de texte|una página de periódico|Una página de periódico con columnas de texto
"""
HELD_OUT_DOCUMENTS = {"boarding", "bizcard", "chart", "label", "certificate"}


def documents() -> dict[str, dict[str, tuple[str, str]]]:
    out = {}
    for line in _DOCUMENTS.strip().splitlines():
        kind, *cells = line.split("|")
        if len(cells) != 2 * len(LANGS):
            raise ValueError(f"bad line in the documents table: {line}")
        out[kind] = {lang: (cells[2 * i], cells[2 * i + 1]) for i, lang in enumerate(LANGS)}
    return out


# A page's state: three ways to say it and three to ask about it per language; the last of each is never trained on.
PAGES: dict[str, dict[str, dict[str, list[str]]]] = {
    "ready": {
        "en": {"say": ["An ordinary page with nothing in the way.", "The page is ready to use: nothing covers it.", "A normal page you can read and use."],
               "ask": ["Is this page ready to use, with nothing covering it?", "Is this an ordinary page with nothing in the way?", "Can this page be used as it is?"]},
        "nl": {"say": ["Een gewone pagina waar niets in de weg staat.", "De pagina is klaar voor gebruik: niets bedekt haar.", "Een normale pagina die je kunt lezen en gebruiken."],
               "ask": ["Is deze pagina klaar voor gebruik, zonder dat iets haar bedekt?", "Is dit een gewone pagina waar niets in de weg staat?", "Kan deze pagina zo gebruikt worden?"]},
        "de": {"say": ["Eine gewöhnliche Seite, bei der nichts im Weg ist.", "Die Seite ist bereit: nichts verdeckt sie.", "Eine normale Seite, die man lesen und benutzen kann."],
               "ask": ["Ist diese Seite bereit, ohne dass etwas sie verdeckt?", "Ist das eine gewöhnliche Seite, bei der nichts im Weg ist?", "Kann diese Seite so benutzt werden?"]},
        "fr": {"say": ["Une page ordinaire, sans rien qui gêne.", "La page est prête à l'emploi : rien ne la recouvre.", "Une page normale que l'on peut lire et utiliser."],
               "ask": ["Cette page est-elle prête à l'emploi, sans rien qui la recouvre ?", "Est-ce une page ordinaire, sans rien qui gêne ?", "Peut-on utiliser cette page telle quelle ?"]},
        "es": {"say": ["Una página corriente sin nada que estorbe.", "La página está lista para usarse: nada la tapa.", "Una página normal que se puede leer y usar."],
               "ask": ["¿Está esta página lista para usarse, sin nada que la tape?", "¿Es una página corriente sin nada que estorbe?", "¿Se puede usar esta página tal como está?"]},
    },
    "cookies": {
        "en": {"say": ["A cookie notice covers part of the page.", "The page asks you to accept or refuse cookies.", "A cookie banner is waiting for an answer."],
               "ask": ["Is a cookie notice covering part of this page?", "Does this page ask about cookies?", "Is there a cookie banner here?"]},
        "nl": {"say": ["Een cookiemelding bedekt een deel van de pagina.", "De pagina vraagt je cookies te accepteren of te weigeren.", "Een cookiebanner wacht op een antwoord."],
               "ask": ["Bedekt een cookiemelding een deel van deze pagina?", "Vraagt deze pagina iets over cookies?", "Staat hier een cookiebanner?"]},
        "de": {"say": ["Ein Cookie-Hinweis verdeckt einen Teil der Seite.", "Die Seite bittet darum, Cookies zu akzeptieren oder abzulehnen.", "Ein Cookie-Banner wartet auf eine Antwort."],
               "ask": ["Verdeckt ein Cookie-Hinweis einen Teil dieser Seite?", "Fragt diese Seite nach Cookies?", "Gibt es hier ein Cookie-Banner?"]},
        "fr": {"say": ["Un avis sur les cookies recouvre une partie de la page.", "La page demande d'accepter ou de refuser les cookies.", "Un bandeau de cookies attend une réponse."],
               "ask": ["Un avis sur les cookies recouvre-t-il une partie de cette page ?", "Cette page pose-t-elle une question sur les cookies ?", "Y a-t-il un bandeau de cookies ici ?"]},
        "es": {"say": ["Un aviso de cookies tapa parte de la página.", "La página pide aceptar o rechazar las cookies.", "Un banner de cookies espera una respuesta."],
               "ask": ["¿Tapa un aviso de cookies parte de esta página?", "¿Pregunta esta página por las cookies?", "¿Hay aquí un banner de cookies?"]},
    },
    "popup": {
        "en": {"say": ["A pop-up covers the page.", "A dialog in the middle of the page wants an answer first.", "An overlay window is in front of the page."],
               "ask": ["Is a pop-up covering this page?", "Is a dialog in front of the page?", "Is something popping up over this page?"]},
        "nl": {"say": ["Een pop-up bedekt de pagina.", "Een dialoogvenster midden op de pagina wil eerst een antwoord.", "Er staat een venster vóór de pagina."],
               "ask": ["Bedekt een pop-up deze pagina?", "Staat er een dialoogvenster vóór de pagina?", "Verschijnt er iets over deze pagina heen?"]},
        "de": {"say": ["Ein Pop-up verdeckt die Seite.", "Ein Dialog mitten auf der Seite will zuerst eine Antwort.", "Ein Fenster liegt vor der Seite."],
               "ask": ["Verdeckt ein Pop-up diese Seite?", "Liegt ein Dialog vor der Seite?", "Erscheint etwas über dieser Seite?"]},
        "fr": {"say": ["Une fenêtre pop-up recouvre la page.", "Une boîte de dialogue au milieu de la page attend d'abord une réponse.", "Une fenêtre se trouve devant la page."],
               "ask": ["Une fenêtre pop-up recouvre-t-elle cette page ?", "Y a-t-il une boîte de dialogue devant la page ?", "Quelque chose s'affiche-t-il par-dessus cette page ?"]},
        "es": {"say": ["Una ventana emergente tapa la página.", "Un cuadro de diálogo en medio de la página pide antes una respuesta.", "Hay una ventana delante de la página."],
               "ask": ["¿Tapa una ventana emergente esta página?", "¿Hay un cuadro de diálogo delante de la página?", "¿Aparece algo por encima de esta página?"]},
    },
    "signin": {
        "en": {"say": ["The page asks you to sign in.", "A sign-in form with a password field.", "You have to log in before you can go on."],
               "ask": ["Does this page ask you to sign in?", "Is this a sign-in form?", "Do you have to log in here?"]},
        "nl": {"say": ["De pagina vraagt je in te loggen.", "Een inlogformulier met een wachtwoordveld.", "Je moet inloggen voordat je verder kunt."],
               "ask": ["Vraagt deze pagina je in te loggen?", "Is dit een inlogformulier?", "Moet je hier inloggen?"]},
        "de": {"say": ["Die Seite fordert zur Anmeldung auf.", "Ein Anmeldeformular mit einem Passwortfeld.", "Man muss sich anmelden, bevor es weitergeht."],
               "ask": ["Fordert diese Seite zur Anmeldung auf?", "Ist das ein Anmeldeformular?", "Muss man sich hier anmelden?"]},
        "fr": {"say": ["La page demande de se connecter.", "Un formulaire de connexion avec un champ de mot de passe.", "Il faut se connecter avant de continuer."],
               "ask": ["Cette page demande-t-elle de se connecter ?", "Est-ce un formulaire de connexion ?", "Faut-il se connecter ici ?"]},
        "es": {"say": ["La página pide iniciar sesión.", "Un formulario de inicio de sesión con un campo de contraseña.", "Hay que iniciar sesión antes de seguir."],
               "ask": ["¿Pide esta página iniciar sesión?", "¿Es un formulario de inicio de sesión?", "¿Hay que iniciar sesión aquí?"]},
    },
    "error": {
        "en": {"say": ["The page shows an error.", "An error page: the page could not be found or loaded.", "Something went wrong and the page says so."],
               "ask": ["Does this page show an error?", "Is this an error page?", "Did something go wrong on this page?"]},
        "nl": {"say": ["De pagina toont een fout.", "Een foutpagina: de pagina kon niet worden gevonden of geladen.", "Er ging iets mis en de pagina zegt dat."],
               "ask": ["Toont deze pagina een fout?", "Is dit een foutpagina?", "Ging er iets mis op deze pagina?"]},
        "de": {"say": ["Die Seite zeigt einen Fehler.", "Eine Fehlerseite: die Seite wurde nicht gefunden oder nicht geladen.", "Etwas ist schiefgegangen, und die Seite sagt es."],
               "ask": ["Zeigt diese Seite einen Fehler?", "Ist das eine Fehlerseite?", "Ist auf dieser Seite etwas schiefgegangen?"]},
        "fr": {"say": ["La page affiche une erreur.", "Une page d'erreur : la page est introuvable ou n'a pas pu être chargée.", "Quelque chose s'est mal passé et la page le dit."],
               "ask": ["Cette page affiche-t-elle une erreur ?", "Est-ce une page d'erreur ?", "Quelque chose s'est-il mal passé sur cette page ?"]},
        "es": {"say": ["La página muestra un error.", "Una página de error: no se pudo encontrar o cargar la página.", "Algo salió mal y la página lo dice."],
               "ask": ["¿Muestra esta página un error?", "¿Es una página de error?", "¿Salió algo mal en esta página?"]},
    },
    "loading": {
        "en": {"say": ["The page is still loading.", "Only placeholders are shown while the content loads.", "The content has not arrived yet."],
               "ask": ["Is this page still loading?", "Is the content still on its way?", "Has the page not finished loading?"]},
        "nl": {"say": ["De pagina is nog aan het laden.", "Er staan alleen plaatshouders terwijl de inhoud laadt.", "De inhoud is er nog niet."],
               "ask": ["Is deze pagina nog aan het laden?", "Is de inhoud nog onderweg?", "Is de pagina nog niet klaar met laden?"]},
        "de": {"say": ["Die Seite lädt noch.", "Es werden nur Platzhalter gezeigt, während der Inhalt lädt.", "Der Inhalt ist noch nicht da."],
               "ask": ["Lädt diese Seite noch?", "Ist der Inhalt noch unterwegs?", "Ist die Seite noch nicht fertig geladen?"]},
        "fr": {"say": ["La page est encore en cours de chargement.", "Seuls des espaces réservés s'affichent pendant le chargement du contenu.", "Le contenu n'est pas encore arrivé."],
               "ask": ["Cette page est-elle encore en cours de chargement ?", "Le contenu est-il encore en route ?", "La page n'a-t-elle pas fini de charger ?"]},
        "es": {"say": ["La página todavía se está cargando.", "Solo se ven marcadores de posición mientras carga el contenido.", "El contenido aún no ha llegado."],
               "ask": ["¿Sigue cargándose esta página?", "¿Está el contenido todavía en camino?", "¿No ha terminado de cargar la página?"]},
    },
    "robot": {
        "en": {"say": ["The page asks you to prove you are not a robot.", "A check that you are human, such as a CAPTCHA.", "A robot check blocks the page."],
               "ask": ["Does this page ask you to prove you are not a robot?", "Is there a CAPTCHA on this page?", "Is a robot check in the way?"]},
        "nl": {"say": ["De pagina vraagt je te bewijzen dat je geen robot bent.", "Een controle of je een mens bent, zoals een CAPTCHA.", "Een robotcontrole blokkeert de pagina."],
               "ask": ["Vraagt deze pagina je te bewijzen dat je geen robot bent?", "Staat er een CAPTCHA op deze pagina?", "Staat een robotcontrole in de weg?"]},
        "de": {"say": ["Die Seite verlangt den Nachweis, dass man kein Roboter ist.", "Eine Prüfung, ob man ein Mensch ist, etwa ein CAPTCHA.", "Eine Roboterprüfung blockiert die Seite."],
               "ask": ["Verlangt diese Seite den Nachweis, dass man kein Roboter ist?", "Gibt es auf dieser Seite ein CAPTCHA?", "Ist eine Roboterprüfung im Weg?"]},
        "fr": {"say": ["La page demande de prouver que vous n'êtes pas un robot.", "Une vérification que vous êtes humain, comme un CAPTCHA.", "Un contrôle anti-robot bloque la page."],
               "ask": ["Cette page demande-t-elle de prouver que vous n'êtes pas un robot ?", "Y a-t-il un CAPTCHA sur cette page ?", "Un contrôle anti-robot fait-il obstacle ?"]},
        "es": {"say": ["La página pide demostrar que no eres un robot.", "Una comprobación de que eres humano, como un CAPTCHA.", "Una verificación antirrobots bloquea la página."],
               "ask": ["¿Pide esta página demostrar que no eres un robot?", "¿Hay un CAPTCHA en esta página?", "¿Estorba una verificación antirrobots?"]},
    },
    "paywall": {
        "en": {"say": ["The article is cut off and asks you to subscribe.", "A paywall: you have to pay or subscribe to read on.", "Reading on needs a subscription."],
               "ask": ["Is a paywall blocking this article?", "Does the page ask you to subscribe to read on?", "Is the rest of the article behind a subscription?"]},
        "nl": {"say": ["Het artikel is afgebroken en vraagt je een abonnement te nemen.", "Een betaalmuur: je moet betalen of je abonneren om verder te lezen.", "Verder lezen kan alleen met een abonnement."],
               "ask": ["Blokkeert een betaalmuur dit artikel?", "Vraagt de pagina je een abonnement te nemen om verder te lezen?", "Zit de rest van het artikel achter een abonnement?"]},
        "de": {"say": ["Der Artikel bricht ab und fordert zum Abonnieren auf.", "Eine Bezahlschranke: zum Weiterlesen muss man zahlen oder abonnieren.", "Weiterlesen geht nur mit einem Abo."],
               "ask": ["Blockiert eine Bezahlschranke diesen Artikel?", "Fordert die Seite zum Abonnieren auf, um weiterzulesen?", "Liegt der Rest des Artikels hinter einem Abo?"]},
        "fr": {"say": ["L'article est coupé et demande de s'abonner.", "Un paywall : il faut payer ou s'abonner pour lire la suite.", "Lire la suite nécessite un abonnement."],
               "ask": ["Un paywall bloque-t-il cet article ?", "La page demande-t-elle de s'abonner pour lire la suite ?", "Le reste de l'article est-il réservé aux abonnés ?"]},
        "es": {"say": ["El artículo está cortado y pide suscribirse.", "Un muro de pago: hay que pagar o suscribirse para seguir leyendo.", "Seguir leyendo requiere una suscripción."],
               "ask": ["¿Bloquea un muro de pago este artículo?", "¿Pide la página suscribirse para seguir leyendo?", "¿Está el resto del artículo tras una suscripción?"]},
    },
}
PAGE_OTHER = {
    "en": ["The page shows something else.", "Anything else.", "That is not what the page shows."],
    "nl": ["De pagina toont iets anders.", "Iets anders.", "Dat is niet wat de pagina toont."],
    "de": ["Die Seite zeigt etwas anderes.", "Etwas anderes.", "Das ist nicht, was die Seite zeigt."],
    "fr": ["La page montre autre chose.", "Autre chose.", "Ce n'est pas ce que la page montre."],
    "es": ["La página muestra otra cosa.", "Otra cosa.", "Eso no es lo que muestra la página."],
}
HELD_OUT_PAGES = {"robot", "paywall"}

# What the drawn documents and pages say. Each entry: en, nl, de, fr, es.
_T = {
    "total": ("TOTAL", "TOTAAL", "SUMME", "TOTAL", "TOTAL"),
    "subtotal": ("Subtotal", "Subtotaal", "Zwischensumme", "Sous-total", "Subtotal"),
    "vat": ("VAT", "BTW", "MwSt.", "TVA", "IVA"),
    "date": ("Date", "Datum", "Datum", "Date", "Fecha"),
    "invoice": ("INVOICE", "FACTUUR", "RECHNUNG", "FACTURE", "FACTURA"),
    "invoice_no": ("Invoice no.", "Factuurnr.", "Rechnungsnr.", "Facture n°", "N.º de factura"),
    "bill_to": ("Bill to", "Factuur aan", "Rechnung an", "Facturé à", "Facturar a"),
    "description": ("Description", "Omschrijving", "Beschreibung", "Désignation", "Descripción"),
    "qty": ("Qty", "Aantal", "Menge", "Qté", "Cant."),
    "price": ("Price", "Prijs", "Preis", "Prix", "Precio"),
    "amount": ("Amount", "Bedrag", "Betrag", "Montant", "Importe"),
    "due": ("Due date", "Vervaldatum", "Fällig am", "Échéance", "Vencimiento"),
    "thanks": ("Thank you for your visit!", "Bedankt voor uw bezoek!", "Vielen Dank für Ihren Besuch!", "Merci de votre visite !", "¡Gracias por su visita!"),
    "card": ("Card", "Pin", "Karte", "Carte", "Tarjeta"),
    "cash": ("Cash", "Contant", "Bar", "Espèces", "Efectivo"),
    "boarding": ("BOARDING PASS", "INSTAPKAART", "BORDKARTE", "CARTE D'EMBARQUEMENT", "TARJETA DE EMBARQUE"),
    "passenger": ("Passenger", "Passagier", "Passagier", "Passager", "Pasajero"),
    "flight": ("Flight", "Vlucht", "Flug", "Vol", "Vuelo"),
    "gate": ("Gate", "Gate", "Gate", "Porte", "Puerta"),
    "seat": ("Seat", "Stoel", "Sitz", "Siège", "Asiento"),
    "boards": ("Boarding", "Boarding", "Boarding", "Embarquement", "Embarque"),
    "from": ("From", "Van", "Von", "De", "De"),
    "to": ("To", "Naar", "Nach", "À", "A"),
    "admit": ("ADMIT ONE", "TOEGANG 1 PERSOON", "EINTRITT 1 PERSON", "ENTRÉE 1 PERSONNE", "ENTRADA 1 PERSONA"),
    "row": ("Row", "Rij", "Reihe", "Rang", "Fila"),
    "doors": ("Doors open", "Deuren open", "Einlass", "Ouverture des portes", "Apertura de puertas"),
    "dear": ("Dear", "Beste", "Guten Tag", "Bonjour", "Estimado/a"),
    "regards": ("Kind regards,", "Met vriendelijke groet,", "Mit freundlichen Grüßen", "Cordialement,", "Un cordial saludo,"),
    "subject": ("Subject", "Betreft", "Betreff", "Objet", "Asunto"),
    "idcard": ("IDENTITY CARD", "IDENTITEITSKAART", "PERSONALAUSWEIS", "CARTE D'IDENTITÉ", "DOCUMENTO DE IDENTIDAD"),
    "specimen": ("SPECIMEN", "SPECIMEN", "MUSTER", "SPÉCIMEN", "ESPÉCIMEN"),
    "surname": ("Surname", "Achternaam", "Name", "Nom", "Apellidos"),
    "given": ("Given names", "Voornamen", "Vornamen", "Prénoms", "Nombre"),
    "born": ("Date of birth", "Geboortedatum", "Geburtsdatum", "Date de naissance", "Fecha de nacimiento"),
    "valid": ("Valid until", "Geldig tot", "Gültig bis", "Valable jusqu'au", "Válido hasta"),
    "starters": ("Starters", "Voorgerechten", "Vorspeisen", "Entrées", "Entrantes"),
    "mains": ("Main courses", "Hoofdgerechten", "Hauptgerichte", "Plats", "Platos principales"),
    "desserts": ("Desserts", "Nagerechten", "Desserts", "Desserts", "Postres"),
    "drinks": ("Drinks", "Dranken", "Getränke", "Boissons", "Bebidas"),
    "balance": ("Balance", "Saldo", "Kontostand", "Solde", "Saldo"),
    "statement": ("Account statement", "Rekeningafschrift", "Kontoauszug", "Relevé de compte", "Extracto de cuenta"),
    "account": ("Account", "Rekening", "Konto", "Compte", "Cuenta"),
    "ship_to": ("SHIP TO", "VERZENDEN AAN", "EMPFÄNGER", "DESTINATAIRE", "DESTINATARIO"),
    "sender": ("FROM", "AFZENDER", "ABSENDER", "EXPÉDITEUR", "REMITENTE"),
    "tracking": ("Tracking no.", "Track & trace", "Sendungsnr.", "N° de suivi", "N.º de seguimiento"),
    "certificate": ("CERTIFICATE", "CERTIFICAAT", "URKUNDE", "CERTIFICAT", "CERTIFICADO"),
    "awarded": ("is hereby awarded to", "wordt hierbij toegekend aan", "wird hiermit verliehen an", "est décerné à", "se otorga a"),
    "for": ("for completing the course", "voor het afronden van de cursus", "für den Abschluss des Kurses", "pour avoir terminé la formation", "por completar el curso"),
    "signature": ("Signature", "Handtekening", "Unterschrift", "Signature", "Firma"),
    "form": ("Application form", "Aanvraagformulier", "Antragsformular", "Formulaire de demande", "Formulario de solicitud"),
    "name": ("Name", "Naam", "Name", "Nom", "Nombre"),
    "address": ("Address", "Adres", "Adresse", "Adresse", "Dirección"),
    "phone": ("Phone", "Telefoon", "Telefon", "Téléphone", "Teléfono"),
    "email": ("Email", "E-mail", "E-Mail", "E-mail", "Correo"),
    "inbox": ("Inbox", "Inbox", "Posteingang", "Boîte de réception", "Bandeja de entrada"),
    "sent": ("Sent", "Verzonden", "Gesendet", "Envoyés", "Enviados"),
    "drafts": ("Drafts", "Concepten", "Entwürfe", "Brouillons", "Borradores"),
    "trash": ("Trash", "Prullenbak", "Papierkorb", "Corbeille", "Papelera"),
    "search": ("Search", "Zoeken", "Suchen", "Rechercher", "Buscar"),
    "type_message": ("Type a message", "Typ een bericht", "Nachricht schreiben", "Écrire un message", "Escribe un mensaje"),
    "todo": ("To do", "Te doen", "Zu erledigen", "À faire", "Por hacer"),
    "shopping": ("Shopping", "Boodschappen", "Einkaufen", "Courses", "Compra"),
    # pages
    "home": ("Home", "Home", "Startseite", "Accueil", "Inicio"),
    "products": ("Products", "Producten", "Produkte", "Produits", "Productos"),
    "about": ("About", "Over ons", "Über uns", "À propos", "Nosotros"),
    "contact": ("Contact", "Contact", "Kontakt", "Contact", "Contacto"),
    "news": ("News", "Nieuws", "Nachrichten", "Actualités", "Noticias"),
    "pricing": ("Pricing", "Prijzen", "Preise", "Tarifs", "Precios"),
    "sign_in": ("Sign in", "Inloggen", "Anmelden", "Se connecter", "Iniciar sesión"),
    "password": ("Password", "Wachtwoord", "Passwort", "Mot de passe", "Contraseña"),
    "forgot": ("Forgot your password?", "Wachtwoord vergeten?", "Passwort vergessen?", "Mot de passe oublié ?", "¿Has olvidado tu contraseña?"),
    "welcome_back": ("Welcome back", "Welkom terug", "Willkommen zurück", "Bon retour", "Bienvenido de nuevo"),
    "sign_in_to": ("Sign in to continue", "Log in om verder te gaan", "Melde dich an, um fortzufahren", "Connectez-vous pour continuer", "Inicia sesión para continuar"),
    "add_to_cart": ("Add to cart", "In winkelwagen", "In den Warenkorb", "Ajouter au panier", "Añadir al carrito"),
    "read_more": ("Read more", "Lees meer", "Weiterlesen", "Lire la suite", "Leer más"),
    "cookies_title": ("We value your privacy", "Wij geven om je privacy", "Deine Privatsphäre ist uns wichtig", "Nous respectons votre vie privée", "Valoramos tu privacidad"),
    "cookies_text": ("We use cookies to improve your experience, analyse traffic and show relevant content.",
                     "Wij gebruiken cookies om je ervaring te verbeteren, verkeer te analyseren en relevante inhoud te tonen.",
                     "Wir verwenden Cookies, um dein Erlebnis zu verbessern, Zugriffe zu analysieren und passende Inhalte zu zeigen.",
                     "Nous utilisons des cookies pour améliorer votre expérience, analyser le trafic et afficher du contenu pertinent.",
                     "Usamos cookies para mejorar tu experiencia, analizar el tráfico y mostrar contenido relevante."),
    "accept": ("Accept all", "Alles accepteren", "Alle akzeptieren", "Tout accepter", "Aceptar todo"),
    "reject": ("Reject all", "Alles weigeren", "Alle ablehnen", "Tout refuser", "Rechazar todo"),
    "manage": ("Manage settings", "Instellingen beheren", "Einstellungen verwalten", "Gérer les paramètres", "Gestionar ajustes"),
    "newsletter": ("Get 10% off your first order", "10% korting op je eerste bestelling", "10 % Rabatt auf deine erste Bestellung", "10 % de réduction sur votre première commande", "10 % de descuento en tu primer pedido"),
    "newsletter_text": ("Sign up for our newsletter and never miss an offer.", "Meld je aan voor onze nieuwsbrief en mis geen aanbieding.",
                        "Melde dich für unseren Newsletter an und verpasse kein Angebot.", "Inscrivez-vous à notre newsletter et ne manquez aucune offre.",
                        "Suscríbete a nuestro boletín y no te pierdas ninguna oferta."),
    "subscribe": ("Subscribe", "Aanmelden", "Abonnieren", "S'abonner", "Suscribirse"),
    "no_thanks": ("No thanks", "Nee, bedankt", "Nein danke", "Non merci", "No, gracias"),
    "not_found": ("Page not found", "Pagina niet gevonden", "Seite nicht gefunden", "Page introuvable", "Página no encontrada"),
    "not_found_text": ("The page you are looking for does not exist or has been moved.", "De pagina die je zoekt bestaat niet of is verplaatst.",
                       "Die gesuchte Seite existiert nicht oder wurde verschoben.", "La page que vous recherchez n'existe pas ou a été déplacée.",
                       "La página que buscas no existe o se ha movido."),
    "went_wrong": ("Something went wrong", "Er is iets misgegaan", "Etwas ist schiefgegangen", "Une erreur s'est produite", "Algo ha salido mal"),
    "try_again": ("Try again", "Probeer opnieuw", "Erneut versuchen", "Réessayer", "Reintentar"),
    "go_home": ("Back to home", "Terug naar home", "Zur Startseite", "Retour à l'accueil", "Volver al inicio"),
    "cant_reach": ("This site can't be reached", "Deze site is niet bereikbaar", "Die Website ist nicht erreichbar", "Ce site est inaccessible", "No se puede acceder a este sitio"),
    "loading": ("Loading…", "Laden…", "Wird geladen…", "Chargement…", "Cargando…"),
    "robot": ("I'm not a robot", "Ik ben geen robot", "Ich bin kein Roboter", "Je ne suis pas un robot", "No soy un robot"),
    "verify": ("Verify you are human", "Bevestig dat je een mens bent", "Bestätige, dass du ein Mensch bist", "Vérifiez que vous êtes humain", "Verifica que eres humano"),
    "verify_text": ("This helps us keep automated traffic away. It only takes a moment.", "Zo houden we geautomatiseerd verkeer buiten. Het duurt maar even.",
                    "So halten wir automatisierte Zugriffe fern. Es dauert nur einen Moment.", "Cela nous aide à écarter le trafic automatisé. Cela ne prend qu'un instant.",
                    "Así evitamos el tráfico automatizado. Solo tarda un momento."),
    "select_all": ("Select all squares with", "Selecteer alle vakjes met", "Wähle alle Felder mit", "Sélectionnez toutes les cases avec", "Selecciona todas las casillas con"),
    "things": ("traffic lights", "verkeerslichten", "Ampeln", "des feux de circulation", "semáforos"),
    "paywall": ("Subscribe to keep reading", "Abonneer je om verder te lezen", "Abonniere, um weiterzulesen", "Abonnez-vous pour lire la suite", "Suscríbete para seguir leyendo"),
    "paywall_text": ("You have reached your limit of free articles this month.", "Je hebt je limiet aan gratis artikelen deze maand bereikt.",
                     "Du hast dein Limit an kostenlosen Artikeln für diesen Monat erreicht.", "Vous avez atteint votre limite d'articles gratuits ce mois-ci.",
                     "Has alcanzado tu límite de artículos gratuitos este mes."),
    "per_month": ("per month", "per maand", "pro Monat", "par mois", "al mes"),
    "already": ("Already a subscriber? Sign in", "Al abonnee? Log in", "Schon Abonnent? Anmelden", "Déjà abonné ? Se connecter", "¿Ya eres suscriptor? Inicia sesión"),
    "results": ("results", "resultaten", "Ergebnisse", "résultats", "resultados"),
    "overview": ("Overview", "Overzicht", "Übersicht", "Vue d'ensemble", "Resumen"),
    "revenue": ("Revenue", "Omzet", "Umsatz", "Chiffre d'affaires", "Ingresos"),
    "users": ("Users", "Gebruikers", "Nutzer", "Utilisateurs", "Usuarios"),
    "orders": ("Orders", "Bestellingen", "Bestellungen", "Commandes", "Pedidos"),
    "docs": ("Documentation", "Documentatie", "Dokumentation", "Documentation", "Documentación"),
    "get_started": ("Get started", "Aan de slag", "Loslegen", "Commencer", "Empezar"),
}
WEEKDAYS = {"en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "nl": ["ma", "di", "wo", "do", "vr", "za", "zo"], "de": ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"],
            "fr": ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."], "es": ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]}
MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "nl": ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"],
    "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"],
    "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
}
# Ordinary words to make sentences of: nobody reads them at 256 pixels, but text in a language has that language's look.
FILLER = {
    "en": "the of and to in is that for it with as was on are this by be at from or have an they which one had not but what all were when we there can more if will out other up some time new first after also over only many these two may then about into than them her like him see now could made people water day years most my where know just place work part take get little world very still hand high".split(),
    "nl": "de het een van en in is dat op te voor met als zijn er maar om ook aan niet bij of naar uit nog wel wat door over ze dit kan meer tot heeft worden deze veel hebben onze waar nieuwe jaar tijd goed eerste mensen tussen alle zonder onder wordt moet elke week kunnen samen altijd werk huis water dag hand hoog klein groot weer".split(),
    "de": "der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er hat dass sie nach wird bei einer um am sind noch wie einem über so zum kann nur oder aber vor zur bis mehr durch man sein wurde sehr neue Jahre Zeit zwei gut erste Menschen Wasser Tag Hand hoch klein groß wieder".split(),
    "fr": "le de un et à il être en avoir que pour dans ce qui ne sur se pas plus pouvoir par je avec tout faire son mettre autre on mais nous comme ou si leur y dire elle devoir avant deux même prendre aussi celui donner bien où fois vous encore nouveau aller cela entre premier temps jour monde année eau main grand petit travail maison".split(),
    "es": "de la que el en y a los se del las un por con no una su para es al lo como más o pero sus le ha me si sin sobre este ya entre cuando todo esta ser son dos también fue había muy años hasta desde está mi porque qué solo han yo hay vez puede todos así nos ni parte tiene tiempo nuevo primer día agua mano grande pequeño trabajo casa".split(),
}
DISHES = {
    "en": ["Tomato soup", "Caesar salad", "Grilled salmon", "Mushroom risotto", "Steak and chips", "Roast chicken", "Vegetable curry", "Apple pie", "Chocolate cake", "Cheese board", "Fish and chips", "Garlic bread"],
    "nl": ["Tomatensoep", "Geitenkaassalade", "Gegrilde zalm", "Paddenstoelenrisotto", "Biefstuk met friet", "Kip van het spit", "Groentecurry", "Appeltaart", "Chocoladetaart", "Kaasplankje", "Kibbeling", "Stokbrood met kruidenboter"],
    "de": ["Tomatensuppe", "Gemischter Salat", "Gegrillter Lachs", "Pilzrisotto", "Wiener Schnitzel", "Brathähnchen", "Gemüsecurry", "Apfelstrudel", "Schokoladenkuchen", "Käseplatte", "Bratwurst mit Kartoffelsalat", "Knoblauchbrot"],
    "fr": ["Soupe à l'oignon", "Salade de chèvre chaud", "Saumon grillé", "Risotto aux champignons", "Steak frites", "Poulet rôti", "Curry de légumes", "Tarte aux pommes", "Fondant au chocolat", "Assiette de fromages", "Moules marinières", "Pain à l'ail"],
    "es": ["Gazpacho", "Ensalada mixta", "Salmón a la plancha", "Risotto de setas", "Entrecot con patatas", "Pollo asado", "Curry de verduras", "Tarta de manzana", "Tarta de chocolate", "Tabla de quesos", "Paella de marisco", "Pan con ajo"],
}
GROCERIES = {
    "en": ["Milk 1L", "Wholemeal bread", "Eggs x12", "Butter", "Cheddar", "Apples 1kg", "Bananas", "Coffee beans", "Pasta", "Tomato sauce", "Orange juice", "Yoghurt", "Rice 1kg", "Chicken breast", "Olive oil", "Toilet paper", "Washing-up liquid", "Chocolate"],
    "nl": ["Halfvolle melk", "Volkorenbrood", "Eieren 12 st", "Roomboter", "Jonge kaas", "Appels 1kg", "Bananen", "Koffiebonen", "Pasta", "Tomatensaus", "Sinaasappelsap", "Yoghurt", "Rijst 1kg", "Kipfilet", "Olijfolie", "Toiletpapier", "Afwasmiddel", "Chocolade"],
    "de": ["Vollmilch 1L", "Vollkornbrot", "Eier 12 St", "Butter", "Gouda", "Äpfel 1kg", "Bananen", "Kaffeebohnen", "Nudeln", "Tomatensoße", "Orangensaft", "Joghurt", "Reis 1kg", "Hähnchenbrust", "Olivenöl", "Toilettenpapier", "Spülmittel", "Schokolade"],
    "fr": ["Lait 1L", "Pain complet", "Œufs x12", "Beurre", "Comté", "Pommes 1kg", "Bananes", "Café en grains", "Pâtes", "Sauce tomate", "Jus d'orange", "Yaourt", "Riz 1kg", "Blanc de poulet", "Huile d'olive", "Papier toilette", "Liquide vaisselle", "Chocolat"],
    "es": ["Leche 1L", "Pan integral", "Huevos x12", "Mantequilla", "Queso manchego", "Manzanas 1kg", "Plátanos", "Café en grano", "Pasta", "Salsa de tomate", "Zumo de naranja", "Yogur", "Arroz 1kg", "Pechuga de pollo", "Aceite de oliva", "Papel higiénico", "Lavavajillas", "Chocolate"],
}
SERVICES = {
    "en": ["Consulting hours", "Website design", "Monthly subscription", "Installation", "Maintenance contract", "Travel costs", "Software licence", "Training day", "Hosting", "Support package"],
    "nl": ["Advies-uren", "Websiteontwerp", "Maandabonnement", "Installatie", "Onderhoudscontract", "Reiskosten", "Softwarelicentie", "Trainingsdag", "Hosting", "Supportpakket"],
    "de": ["Beratungsstunden", "Webdesign", "Monatsabo", "Installation", "Wartungsvertrag", "Reisekosten", "Softwarelizenz", "Schulungstag", "Hosting", "Supportpaket"],
    "fr": ["Heures de conseil", "Conception de site web", "Abonnement mensuel", "Installation", "Contrat de maintenance", "Frais de déplacement", "Licence logicielle", "Journée de formation", "Hébergement", "Forfait d'assistance"],
    "es": ["Horas de consultoría", "Diseño web", "Suscripción mensual", "Instalación", "Contrato de mantenimiento", "Gastos de viaje", "Licencia de software", "Jornada de formación", "Alojamiento web", "Paquete de soporte"],
}
AIRPORTS = ["AMS", "BCN", "LHR", "CDG", "FRA", "MAD", "LIS", "FCO", "VIE", "ZRH", "CPH", "OSL", "DUB", "BRU", "MUC", "ATH", "PRG", "HEL"]
SHOPS = ["Markt 12", "Fresh & Co", "De Hoek", "Kaufhalle", "Le Marché", "La Plaza", "Greenway", "Bakkerij Jansen", "Café Luna", "Brasserie Nord", "Mercado Sol", "Stadtcafé"]
COMPANIES = ["Northway Ltd", "Kestrel BV", "Alder GmbH", "Brightline SAS", "Corvana SL", "Tessel Studio", "Fjordmark AS", "Lumen Works", "Orbis Group", "Pinecrest Partners"]
BANKS = ["Northbank", "Kestrel Bank", "Alder Sparkasse", "Banque Lumière", "Banco Sierra", "Tessel Credit Union"]


def t(key: str, lang: str) -> str:
    return _T[key][LANGS.index(lang)]
