# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""What a button with only an icon on it does: 63 jobs, each drawn by nine open icon sets, named in five languages.

A page's code often says nothing about such a button (no label, no title), so atelier-step can't choose it. Its
picture says it. vision/synth.py draws these icons as buttons; this module fetches the sets and holds the table.

    python3 -I vision/icons.py fetch       # the nine sets from npm into data/cache/vision/icons (pinned versions)
    python3 -I vision/icons.py others      # and, from each set, icons that are none of the 63 jobs
    python3 -I vision/icons.py coverage    # which job each set has an icon for

The sets (each under its own licence, listed in NOTICE): Lucide (ISC), Tabler Icons (MIT), Bootstrap Icons (MIT),
Heroicons (MIT), Phosphor (MIT), Feather (MIT), Ionicons (MIT), Material Design Icons (Apache-2.0), Octicons (MIT).
Three sets are never trained on (HELD_OUT_SETS): the test of a drawing style nobody showed it. Eight jobs are never
trained on in any set (HELD_OUT_JOBS): the test of a kind of button nobody showed it.

A real page also has buttons that are none of these jobs: a brand's logo, an anchor, a coffee cup. `others` keeps
some of each set's other icons (ones whose names share no word with any job's icons, so a second bin or a ringing
bell is not among them). The model should not name those: the ones from the held-out sets measure how often it does.

The job names and sentences were written for this project by Claude (Anthropic), CC BY 4.0.
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICONS = ROOT / "data" / "cache" / "vision" / "icons"
LANGS = ["en", "nl", "de", "fr", "es"]

# set: (npm tarball, {style: (folder in the package, suffix before .svg)})
SETS: dict[str, tuple[str, dict[str, tuple[str, str]]]] = {
    "lucide": ("lucide-static/-/lucide-static-1.81.0.tgz", {"line": ("icons", "")}),
    "tabler": ("@tabler/icons/-/icons-3.53.0.tgz", {"line": ("icons/outline", ""), "filled": ("icons/filled", "")}),
    "bootstrap": ("bootstrap-icons/-/bootstrap-icons-1.13.2.tgz", {"line": ("icons", ""), "filled": ("icons", "-fill")}),
    "heroicons": ("heroicons/-/heroicons-2.2.0.tgz", {"line": ("24/outline", ""), "filled": ("24/solid", "")}),
    "phosphor": ("@phosphor-icons/core/-/core-2.1.1.tgz", {"line": ("assets/regular", ""), "bold": ("assets/bold", "-bold"), "filled": ("assets/fill", "-fill")}),
    "feather": ("feather-icons/-/feather-icons-4.29.2.tgz", {"line": ("dist/icons", "")}),
    "ionicons": ("ionicons/-/ionicons-8.1.1.tgz", {"line": ("dist/svg", "-outline"), "filled": ("dist/svg", ""), "sharp": ("dist/svg", "-sharp")}),
    "material": ("@material-design-icons/svg/-/svg-0.14.15.tgz", {"line": ("outlined", ""), "filled": ("filled", ""), "round": ("round", ""), "sharp": ("sharp", "")}),
    "octicons": ("@primer/octicons/-/octicons-19.43.0.tgz", {"line": ("build/svg", "-24"), "small": ("build/svg", "-16")}),
}
ORDER = list(SETS)
HELD_OUT_SETS = {"ionicons", "material", "octicons"}
HELD_OUT_JOBS = {"print", "gift", "dark", "attach", "zoomin", "logout", "tag", "language"}

# job | lucide | tabler | bootstrap | heroicons | phosphor | feather | ionicons | material | octicons
# Several names in one column are tried in order; "-" means the set has no such icon.
_ICONS = """
cart|shopping-cart|shopping-cart|cart,cart3|shopping-cart|shopping-cart|shopping-cart|cart|shopping_cart|-
search|search|search|search|magnifying-glass|magnifying-glass|search|search|search|search
close|x|x|x-lg,x|x-mark|x|x|close|close|x
menu|menu|menu-2|list|bars-3|list|menu|menu|menu|three-bars
settings|settings|settings|gear|cog-6-tooth|gear|settings|settings|settings|gear
account|user|user|person|user|user|user|person|person|person
home|house,home|home|house|home|house|home|home|home|home
favorite|heart|heart|heart|heart|heart|heart|heart|favorite_border,favorite|heart
star|star|star|star|star|star|star|star|star_border,star|star
bookmark|bookmark|bookmark|bookmark|bookmark|bookmark-simple|bookmark|bookmark|bookmark_border,bookmark|bookmark
share|share-2|share|share|share|share-network|share-2|share-social|share|share
download|download|download|download|arrow-down-tray|download-simple|download|download|file_download,download|download
upload|upload|upload|upload|arrow-up-tray|upload-simple|upload|cloud-upload|file_upload,upload|upload
delete|trash-2|trash|trash|trash|trash|trash-2|trash|delete|trash
edit|pencil|pencil|pencil|pencil|pencil-simple|edit-2|pencil,create|edit|pencil
add|plus|plus|plus-lg,plus|plus|plus|plus|add|add|plus
remove|minus|minus|dash-lg,dash|minus|minus|minus|remove|remove|dash
back|arrow-left|arrow-left|arrow-left|arrow-left|arrow-left|arrow-left|arrow-back|arrow_back|arrow-left
forward|arrow-right|arrow-right|arrow-right|arrow-right|arrow-right|arrow-right|arrow-forward|arrow_forward|arrow-right
play|play|player-play|play|play|play|play|play|play_arrow|play
pause|pause|player-pause|pause|pause|pause|pause|pause|pause|-
sound|volume-2|volume|volume-up|speaker-wave|speaker-high|volume-2|volume-high|volume_up|unmute
mute|volume-x|volume-off|volume-mute|speaker-x-mark|speaker-x|volume-x|volume-mute|volume_off|mute
microphone|mic|microphone|mic|microphone|microphone|mic|mic|mic|-
camera|camera|camera|camera|camera|camera|camera|camera|photo_camera|device-camera
image|image|photo|image|photo|image|image|image|image|image
calendar|calendar|calendar|calendar|calendar|calendar-blank|calendar|calendar|calendar_today,event|calendar
clock|clock|clock|clock|clock|clock|clock|time|schedule|clock
location|map-pin|map-pin|geo-alt|map-pin|map-pin|map-pin|location|place,location_on|location
mail|mail|mail|envelope|envelope|envelope-simple|mail|mail|mail,email|mail
phone|phone|phone|telephone|phone|phone|phone|call|phone,call|-
chat|message-circle|message-circle|chat|chat-bubble-left|chat-circle|message-circle|chatbubble|chat|comment
notifications|bell|bell|bell|bell|bell|bell|notifications|notifications|bell
lock|lock|lock|lock|lock-closed|lock|lock|lock-closed|lock|lock
filter|funnel,filter|filter|funnel|funnel|funnel|filter|funnel|filter_alt,filter_list|filter
refresh|refresh-cw|refresh|arrow-clockwise|arrow-path|arrow-clockwise|refresh-cw|refresh|refresh|sync
print|printer|printer|printer|printer|printer|printer|print|print|-
copy|copy|copy|copy,clipboard|document-duplicate|copy|copy|copy|content_copy|copy
link|link|link|link-45deg|link|link|link|link|link|link
info|info|info-circle|info-circle|information-circle|info|info|information-circle|info|info
help|circle-help,circle-question-mark,help-circle|help-circle,help|question-circle|question-mark-circle|question|help-circle|help-circle|help_outline,help|question
warning|triangle-alert,alert-triangle|alert-triangle|exclamation-triangle|exclamation-triangle|warning|alert-triangle|warning|warning,warning_amber|alert
confirm|check|check|check-lg,check|check|check|check|checkmark|check,done|check
show|eye|eye|eye|eye|eye|eye|eye|visibility|eye
hide|eye-off|eye-off|eye-slash|eye-slash|eye-slash|eye-off|eye-off|visibility_off|eye-closed
light|sun|sun|sun|sun|sun|sun|sunny|light_mode|sun
dark|moon|moon|moon|moon|moon|moon|moon|dark_mode|moon
language|globe|world|globe|globe-alt|globe|globe|globe|language|globe
more|ellipsis,more-horizontal|dots|three-dots|ellipsis-horizontal|dots-three|more-horizontal|ellipsis-horizontal|more_horiz|kebab-horizontal
grid|layout-grid,grid-2x2|layout-grid|grid|squares-2x2|squares-four|grid|grid|grid_view,apps|apps
list|list|list|list-ul|list-bullet|list-bullets|list|list|format_list_bulleted,list|list-unordered
tag|tag|tag|tag|tag|tag|tag|pricetag|sell,local_offer,label|tag
gift|gift|gift|gift|gift|gift|gift|gift|card_giftcard,redeem|gift
payment|credit-card|credit-card|credit-card|credit-card|credit-card|credit-card|card|credit_card|credit-card
logout|log-out|logout|box-arrow-right|arrow-right-start-on-rectangle,arrow-right-on-rectangle|sign-out|log-out|log-out|logout|sign-out
login|log-in|login|box-arrow-in-right|arrow-left-end-on-rectangle,arrow-left-on-rectangle|sign-in|log-in|log-in|login|sign-in
save|save|device-floppy|floppy,save|-|floppy-disk|save|save|save|-
send|send|send|send|paper-airplane|paper-plane-tilt|send|send|send|paper-airplane
attach|paperclip|paperclip|paperclip|paper-clip|paperclip|paperclip|attach|attach_file|paperclip
zoomin|zoom-in|zoom-in|zoom-in|magnifying-glass-plus|magnifying-glass-plus|zoom-in|-|zoom_in|zoom-in
zoomout|zoom-out|zoom-out|zoom-out|magnifying-glass-minus|magnifying-glass-minus|zoom-out|-|zoom_out|zoom-out
fullscreen|maximize|maximize|fullscreen|arrows-pointing-out|arrows-out|maximize|expand|fullscreen|screen-full
external|external-link|external-link|box-arrow-up-right|arrow-top-right-on-square|arrow-square-out|external-link|open|open_in_new|link-external
"""

# job | then for en, nl, de, fr, es: what the button is called | what it does
_WORDS = """
cart|shopping cart|Opens the shopping cart|winkelwagen|Opent de winkelwagen|Warenkorb|Öffnet den Warenkorb|panier|Ouvre le panier|carrito|Abre el carrito
search|search|Starts a search|zoeken|Start een zoekopdracht|Suche|Startet eine Suche|recherche|Lance une recherche|buscar|Inicia una búsqueda
close|close|Closes this window or panel|sluiten|Sluit dit venster of paneel|Schließen|Schließt dieses Fenster oder Panel|fermer|Ferme cette fenêtre ou ce panneau|cerrar|Cierra esta ventana o panel
menu|menu|Opens the menu|menu|Opent het menu|Menü|Öffnet das Menü|menu|Ouvre le menu|menú|Abre el menú
settings|settings|Opens the settings|instellingen|Opent de instellingen|Einstellungen|Öffnet die Einstellungen|paramètres|Ouvre les paramètres|ajustes|Abre los ajustes
account|account|Opens your account or profile|account|Opent je account of profiel|Konto|Öffnet dein Konto oder Profil|compte|Ouvre votre compte ou profil|cuenta|Abre tu cuenta o perfil
home|home|Goes to the home page|startpagina|Gaat naar de startpagina|Startseite|Geht zur Startseite|accueil|Va à la page d'accueil|inicio|Va a la página de inicio
favorite|favourite|Adds this to your favourites|favoriet|Voegt dit toe aan je favorieten|Favorit|Fügt dies zu deinen Favoriten hinzu|favori|Ajoute ceci à vos favoris|favorito|Añade esto a tus favoritos
star|star|Gives this a star or a rating|ster|Geeft dit een ster of een beoordeling|Stern|Gibt diesem einen Stern oder eine Bewertung|étoile|Donne une étoile ou une note|estrella|Da una estrella o una valoración
bookmark|bookmark|Saves this as a bookmark|bladwijzer|Bewaart dit als bladwijzer|Lesezeichen|Speichert dies als Lesezeichen|signet|Enregistre ceci comme signet|marcador|Guarda esto como marcador
share|share|Shares this with others|delen|Deelt dit met anderen|Teilen|Teilt dies mit anderen|partager|Partage ceci avec d'autres|compartir|Comparte esto con otros
download|download|Downloads a file|downloaden|Downloadt een bestand|Herunterladen|Lädt eine Datei herunter|télécharger|Télécharge un fichier|descargar|Descarga un archivo
upload|upload|Uploads a file|uploaden|Uploadt een bestand|Hochladen|Lädt eine Datei hoch|téléverser|Envoie un fichier depuis votre appareil|subir|Sube un archivo
delete|delete|Deletes this|verwijderen|Verwijdert dit|Löschen|Löscht dies|supprimer|Supprime ceci|eliminar|Elimina esto
edit|edit|Edits this|bewerken|Bewerkt dit|Bearbeiten|Bearbeitet dies|modifier|Modifie ceci|editar|Edita esto
add|add|Adds something new|toevoegen|Voegt iets nieuws toe|Hinzufügen|Fügt etwas Neues hinzu|ajouter|Ajoute un nouvel élément|añadir|Añade algo nuevo
remove|minus|Takes one away or makes it less|min|Haalt er één af of maakt het minder|Minus|Nimmt eins weg oder verringert es|moins|Enlève un élément ou diminue|menos|Quita uno o lo reduce
back|back|Goes back|terug|Gaat terug|Zurück|Geht zurück|retour|Revient en arrière|atrás|Vuelve atrás
forward|next|Goes on to the next one|volgende|Gaat verder naar de volgende|Weiter|Geht weiter zum Nächsten|suivant|Passe au suivant|siguiente|Avanza al siguiente
play|play|Starts playing|afspelen|Start het afspelen|Abspielen|Startet die Wiedergabe|lecture|Lance la lecture|reproducir|Inicia la reproducción
pause|pause|Pauses playing|pauzeren|Pauzeert het afspelen|Pause|Pausiert die Wiedergabe|pause|Met la lecture en pause|pausar|Pausa la reproducción
sound|sound|Changes the volume|geluid|Verandert het volume|Ton|Ändert die Lautstärke|son|Change le volume|sonido|Cambia el volumen
mute|mute|Turns the sound off|dempen|Zet het geluid uit|Stumm|Schaltet den Ton aus|muet|Coupe le son|silenciar|Apaga el sonido
microphone|microphone|Turns the microphone on or records your voice|microfoon|Zet de microfoon aan of neemt je stem op|Mikrofon|Schaltet das Mikrofon ein oder nimmt deine Stimme auf|microphone|Active le micro ou enregistre votre voix|micrófono|Activa el micrófono o graba tu voz
camera|camera|Takes a photo or turns the camera on|camera|Maakt een foto of zet de camera aan|Kamera|Macht ein Foto oder schaltet die Kamera ein|appareil photo|Prend une photo ou active la caméra|cámara|Hace una foto o activa la cámara
image|picture|Adds or shows a picture|afbeelding|Voegt een afbeelding toe of toont die|Bild|Fügt ein Bild hinzu oder zeigt es|image|Ajoute ou affiche une image|imagen|Añade o muestra una imagen
calendar|calendar|Opens the calendar or picks a date|kalender|Opent de kalender of kiest een datum|Kalender|Öffnet den Kalender oder wählt ein Datum|calendrier|Ouvre le calendrier ou choisit une date|calendario|Abre el calendario o elige una fecha
clock|time|Shows the time or the history|tijd|Toont de tijd of de geschiedenis|Uhrzeit|Zeigt die Uhrzeit oder den Verlauf|heure|Affiche l'heure ou l'historique|hora|Muestra la hora o el historial
location|location|Shows a place on the map|locatie|Toont een plek op de kaart|Standort|Zeigt einen Ort auf der Karte|lieu|Affiche un lieu sur la carte|ubicación|Muestra un lugar en el mapa
mail|email|Opens or sends an email|e-mail|Opent of verstuurt een e-mail|E-Mail|Öffnet oder sendet eine E-Mail|e-mail|Ouvre ou envoie un e-mail|correo|Abre o envía un correo
phone|phone call|Starts a phone call|bellen|Start een telefoongesprek|Anruf|Startet einen Anruf|appel|Lance un appel téléphonique|llamada|Inicia una llamada
chat|chat|Opens a chat or the comments|chat|Opent een chat of de reacties|Chat|Öffnet einen Chat oder die Kommentare|discussion|Ouvre une discussion ou les commentaires|chat|Abre un chat o los comentarios
notifications|notifications|Shows your notifications|meldingen|Toont je meldingen|Benachrichtigungen|Zeigt deine Benachrichtigungen|notifications|Affiche vos notifications|notificaciones|Muestra tus notificaciones
lock|lock|Locks this or shows it is secure|slot|Vergrendelt dit of toont dat het beveiligd is|Schloss|Sperrt dies oder zeigt, dass es sicher ist|verrou|Verrouille ceci ou indique que c'est sécurisé|candado|Bloquea esto o indica que es seguro
filter|filter|Filters the list|filter|Filtert de lijst|Filter|Filtert die Liste|filtre|Filtre la liste|filtro|Filtra la lista
refresh|refresh|Reloads or refreshes|vernieuwen|Laadt opnieuw of ververst|Aktualisieren|Lädt neu oder aktualisiert|actualiser|Recharge ou actualise|actualizar|Recarga o actualiza
print|print|Prints this page|afdrukken|Drukt deze pagina af|Drucken|Druckt diese Seite|imprimer|Imprime cette page|imprimir|Imprime esta página
copy|copy|Copies this|kopiëren|Kopieert dit|Kopieren|Kopiert dies|copier|Copie ceci|copiar|Copia esto
link|link|Copies or adds a link|link|Kopieert een link of voegt er een toe|Link|Kopiert einen Link oder fügt einen hinzu|lien|Copie ou ajoute un lien|enlace|Copia o añade un enlace
info|information|Shows more information|informatie|Toont meer informatie|Information|Zeigt mehr Informationen|information|Affiche plus d'informations|información|Muestra más información
help|help|Opens help|hulp|Opent de hulp|Hilfe|Öffnet die Hilfe|aide|Ouvre l'aide|ayuda|Abre la ayuda
warning|warning|Warns about a problem|waarschuwing|Waarschuwt voor een probleem|Warnung|Warnt vor einem Problem|avertissement|Signale un problème|advertencia|Avisa de un problema
confirm|confirm|Confirms this or marks it as done|bevestigen|Bevestigt dit of markeert het als klaar|Bestätigen|Bestätigt dies oder markiert es als erledigt|confirmer|Confirme ou marque comme terminé|confirmar|Confirma esto o lo marca como hecho
show|show|Shows what is hidden|tonen|Toont wat verborgen is|Anzeigen|Zeigt, was verborgen ist|afficher|Affiche ce qui est masqué|mostrar|Muestra lo que está oculto
hide|hide|Hides this|verbergen|Verbergt dit|Ausblenden|Blendet dies aus|masquer|Masque ceci|ocultar|Oculta esto
light|light mode|Switches to the light theme|lichte modus|Schakelt naar het lichte thema|heller Modus|Wechselt zum hellen Design|mode clair|Passe au thème clair|modo claro|Cambia al tema claro
dark|dark mode|Switches to the dark theme|donkere modus|Schakelt naar het donkere thema|dunkler Modus|Wechselt zum dunklen Design|mode sombre|Passe au thème sombre|modo oscuro|Cambia al tema oscuro
language|language|Changes the language|taal|Verandert de taal|Sprache|Ändert die Sprache|langue|Change la langue|idioma|Cambia el idioma
more|more options|Shows more options|meer opties|Toont meer opties|weitere Optionen|Zeigt weitere Optionen|plus d'options|Affiche plus d'options|más opciones|Muestra más opciones
grid|grid view|Shows the items as a grid|rasterweergave|Toont de items als raster|Rasteransicht|Zeigt die Elemente als Raster|vue en grille|Affiche les éléments en grille|vista de cuadrícula|Muestra los elementos en cuadrícula
list|list view|Shows the items as a list|lijstweergave|Toont de items als lijst|Listenansicht|Zeigt die Elemente als Liste|vue en liste|Affiche les éléments en liste|vista de lista|Muestra los elementos en lista
tag|tag|Adds a tag or label|label|Voegt een label toe|Schlagwort|Fügt ein Schlagwort hinzu|étiquette|Ajoute une étiquette|etiqueta|Añade una etiqueta
gift|gift|Sends this as a gift|cadeau|Verstuurt dit als cadeau|Geschenk|Sendet dies als Geschenk|cadeau|Envoie ceci en cadeau|regalo|Envía esto como regalo
payment|payment|Pays or adds a payment card|betalen|Betaalt of voegt een betaalkaart toe|Zahlung|Bezahlt oder fügt eine Zahlungskarte hinzu|paiement|Paie ou ajoute une carte de paiement|pago|Paga o añade una tarjeta de pago
logout|sign out|Signs you out|uitloggen|Logt je uit|Abmelden|Meldet dich ab|déconnexion|Vous déconnecte|cerrar sesión|Cierra tu sesión
login|sign in|Signs you in|inloggen|Logt je in|Anmelden|Meldet dich an|connexion|Vous connecte|iniciar sesión|Inicia tu sesión
save|save|Saves your changes|opslaan|Slaat je wijzigingen op|Speichern|Speichert deine Änderungen|enregistrer|Enregistre vos modifications|guardar|Guarda tus cambios
send|send|Sends the message|verzenden|Verstuurt het bericht|Senden|Sendet die Nachricht|envoyer|Envoie le message|enviar|Envía el mensaje
attach|attach|Attaches a file|bijlage|Voegt een bestand als bijlage toe|Anhängen|Hängt eine Datei an|joindre|Joint un fichier|adjuntar|Adjunta un archivo
zoomin|zoom in|Zooms in|inzoomen|Zoomt in|Vergrößern|Zoomt hinein|zoom avant|Agrandit l'affichage|acercar|Acerca la vista
zoomout|zoom out|Zooms out|uitzoomen|Zoomt uit|Verkleinern|Zoomt heraus|zoom arrière|Réduit l'affichage|alejar|Aleja la vista
fullscreen|full screen|Makes this fill the whole screen|volledig scherm|Maakt dit schermvullend|Vollbild|Zeigt dies im Vollbild|plein écran|Affiche en plein écran|pantalla completa|Muestra esto a pantalla completa
external|open in a new tab|Opens this in a new tab|openen in nieuw tabblad|Opent dit in een nieuw tabblad|in neuem Tab öffnen|Öffnet dies in einem neuen Tab|ouvrir dans un nouvel onglet|Ouvre ceci dans un nouvel onglet|abrir en una pestaña nueva|Abre esto en una pestaña nueva
"""


def words() -> dict[str, dict[str, tuple[str, str]]]:
    """job to {language: (what it is called, what it does)}."""
    out = {}
    for line in _WORDS.strip().splitlines():
        job, *cells = line.split("|")
        if len(cells) != 2 * len(LANGS):
            raise ValueError(f"bad line in the words table: {line}")
        out[job] = {lang: (cells[2 * i], cells[2 * i + 1]) for i, lang in enumerate(LANGS)}
    return out


def candidates() -> dict[str, dict[str, list[str]]]:
    """job to {set: icon names to try}."""
    out = {}
    for line in _ICONS.strip().splitlines():
        job, *cells = line.split("|")
        if len(cells) != len(ORDER):
            raise ValueError(f"bad line in the icon table: {line}")
        out[job] = {s: ([] if c == "-" else c.split(",")) for s, c in zip(ORDER, cells)}
    if set(out) != set(words()) or not HELD_OUT_JOBS <= set(out):
        raise ValueError("the icon table, the words table and HELD_OUT_JOBS name different jobs")
    return out


def files() -> dict[str, dict[str, list[tuple[str, Path]]]]:
    """job to {set: [(style, svg file)]} for the icons that are on disk."""
    out: dict[str, dict[str, list[tuple[str, Path]]]] = {}
    for job, per_set in candidates().items():
        out[job] = {}
        for s, names in per_set.items():
            found = []
            for style, (_, suffix) in SETS[s][1].items():
                for name in names:
                    path = ICONS / s / style / f"{name}{suffix}.svg"
                    if path.exists():
                        found.append((style, path))
                        break
            if found:
                out[job][s] = found
    return out


def fetch() -> None:
    for s, (tarball, styles) in SETS.items():
        if (ICONS / s / "LICENSE").exists():
            continue
        with urllib.request.urlopen(f"https://registry.npmjs.org/{tarball}", timeout=120) as r:
            data = r.read()
        kept = 0
        # Only the icons the table names are kept: a set has thousands, the table uses about sixty of each.
        named = {name for per_set in candidates().values() for name in per_set[s]}
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar:
                path = Path(member.name)
                if member.isfile() and path.parent.as_posix() == "package" and path.name.upper().startswith(("LICENSE", "LICENCE")):
                    # Each set's own licence goes with its icons, and with the pictures drawn from them.
                    (ICONS / s).mkdir(parents=True, exist_ok=True)
                    (ICONS / s / "LICENSE").write_bytes(tar.extractfile(member).read())
                if not member.isfile() or path.suffix != ".svg":
                    continue
                for style, (folder, suffix) in styles.items():
                    # A plain style ("") shares its folder with suffixed ones: the name decides.
                    if path.parent.as_posix() == f"package/{folder}" and path.stem.endswith(suffix) and path.stem[: len(path.stem) - len(suffix)] in named:
                        (ICONS / s / style).mkdir(parents=True, exist_ok=True)
                        (ICONS / s / style / path.name).write_bytes(tar.extractfile(member).read())
                        kept += 1
        print(f"{s}: {kept} icons ({len(data) / 1e6:.1f} MB fetched)", flush=True)


def job_words() -> set[str]:
    """Every word in the name of any icon any set uses for a job: an icon with one of them may mean that job too."""
    import re

    return {w for per_set in candidates().values() for names in per_set.values() for name in names for w in re.split(r"[-_]", name) if len(w) > 1} | {
        "bin", "cog", "magnifier", "loupe", "envelope", "inbox", "basket", "bag", "shop", "store", "avatar", "profile", "people", "person", "users", "delete", "remove", "erase",
        "cancel", "exit", "enter", "door", "key", "password", "shield", "security", "notification", "alarm", "time", "timer", "watch", "date", "schedule", "pin", "marker",
        "navigation", "compass", "back", "forward", "next", "previous", "prev", "chevron", "caret", "angle", "up", "down", "undo", "redo", "reload", "sync", "rotate", "repeat",
        "music", "speaker", "audio", "sound", "video", "film", "photo", "picture", "gallery", "album", "document", "file", "folder", "clipboard", "paste", "cut", "scissors",
        "edit", "write", "pen", "brush", "comment", "comments", "bubble", "speech", "talk", "question", "exclamation", "error", "danger", "caution", "success", "done", "ok",
        "view", "hide", "show", "visible", "invisible", "light", "dark", "night", "day", "brightness", "theme", "world", "earth", "translate", "language", "dots", "options",
        "table", "columns", "rows", "apps", "dashboard", "label", "price", "ticket", "present", "payment", "pay", "wallet", "money", "cash", "bank", "coin", "currency", "dollar",
        "euro", "sign", "account", "disk", "floppy", "plane", "airplane", "telegram", "attach", "attachment", "clip", "expand", "collapse", "minimize", "resize", "screen",
        "window", "open", "new", "tab", "add", "create", "close", "favorite", "like", "love", "rating", "rate", "bookmarks", "social", "network", "export", "import", "cloud",
        "save", "print", "duplicate", "chain", "url", "about", "faq", "support", "warning", "tick", "checkbox", "toggle", "switch", "sliders", "tune", "adjust", "control",
        "preferences", "tools", "wrench", "hamburger", "navicon", "sidebar", "drawer", "more", "search", "find", "zoom", "scan", "home", "house", "building",
    }


def others(per_set: int = 60) -> dict[str, list[Path]]:
    """set to the kept icons that are none of the jobs (fetched when they are not there yet)."""
    import random
    import re

    words = job_words()
    out = {}
    for s, (tarball, styles) in SETS.items():
        folder, suffix = styles["line"]
        target = ICONS / s / "other"
        if not target.exists():
            with urllib.request.urlopen(f"https://registry.npmjs.org/{tarball}", timeout=120) as r:
                data = r.read()
            found = {}
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                for member in tar:
                    path = Path(member.name)
                    if not member.isfile() or path.suffix != ".svg" or path.parent.as_posix() != f"package/{folder}" or not path.stem.endswith(suffix):
                        continue
                    name = path.stem[: len(path.stem) - len(suffix)] if suffix else path.stem
                    if suffix == "" and re.search(r"-(fill|outline|sharp|bold|16|24)$", path.stem):
                        continue  # another style of an icon, in a folder the styles share
                    if not any(w in words for w in re.split(r"[-_]", name)):
                        found[name] = tar.extractfile(member).read()
            picked = random.Random(f"others/{s}").sample(sorted(found), min(per_set, len(found)))
            target.mkdir(parents=True, exist_ok=True)
            for name in picked:
                (target / f"{name}.svg").write_bytes(found[name])
        out[s] = sorted(target.glob("*.svg"))
    return out


def coverage() -> None:
    found = files()
    for s in ORDER:
        missing = [job for job in found if s not in found[job] and candidates()[job][s]]
        absent = [job for job in found if not candidates()[job][s]]
        print(f"{s:10s} {sum(1 for job in found if s in found[job]):2d} of {len(found)} jobs" + (f"; not found: {missing}" if missing else "") + (f"; has none: {absent}" if absent else ""))
    print(f"{sum(len(v) for per in found.values() for v in per.values())} icon files for {len(found)} jobs ({len(HELD_OUT_JOBS)} jobs and {len(HELD_OUT_SETS)} sets held out)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["fetch", "others", "coverage"])
    cmd = parser.parse_args().cmd
    if cmd == "others":
        for name, paths in others().items():
            print(f"{name:10s} {len(paths)} other icons, e.g. {', '.join(p.stem for p in paths[:8])}")
    else:
        {"fetch": fetch, "coverage": coverage}[cmd]()
    sys.exit(0)
