/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// What a button with only an icon on it can be for, named in the five languages atelier-vision's questions were
// trained and measured in (docs/decisions.md › Pictures). Written by the atelier-decisions project's
// tools/vision_dist.py from its vision/icons.py: the names are part of what the model was measured on, so they
// change there, with a new version, not here. Jobs marked `unseen` were never trained on: answers about them
// are measured as a kind of button nobody showed the model.

export const CONTROL_LANGS = ['en', 'nl', 'de', 'fr', 'es'] as const
export type ControlLang = (typeof CONTROL_LANGS)[number]

export interface ControlJob {
  key: string
  names: Record<ControlLang, string>
  unseen?: true
}

export const CONTROL_JOBS: readonly ControlJob[] = [
  { key: 'account', names: { en: 'account', nl: 'account', de: 'Konto', fr: 'compte', es: 'cuenta' } },
  { key: 'add', names: { en: 'add', nl: 'toevoegen', de: 'Hinzufügen', fr: 'ajouter', es: 'añadir' } },
  { key: 'attach', names: { en: 'attach', nl: 'bijlage', de: 'Anhängen', fr: 'joindre', es: 'adjuntar' }, unseen: true },
  { key: 'back', names: { en: 'back', nl: 'terug', de: 'Zurück', fr: 'retour', es: 'atrás' } },
  { key: 'bookmark', names: { en: 'bookmark', nl: 'bladwijzer', de: 'Lesezeichen', fr: 'signet', es: 'marcador' } },
  { key: 'calendar', names: { en: 'calendar', nl: 'kalender', de: 'Kalender', fr: 'calendrier', es: 'calendario' } },
  { key: 'camera', names: { en: 'camera', nl: 'camera', de: 'Kamera', fr: 'appareil photo', es: 'cámara' } },
  { key: 'cart', names: { en: 'shopping cart', nl: 'winkelwagen', de: 'Warenkorb', fr: 'panier', es: 'carrito' } },
  { key: 'chat', names: { en: 'chat', nl: 'chat', de: 'Chat', fr: 'discussion', es: 'chat' } },
  { key: 'clock', names: { en: 'time', nl: 'tijd', de: 'Uhrzeit', fr: 'heure', es: 'hora' } },
  { key: 'close', names: { en: 'close', nl: 'sluiten', de: 'Schließen', fr: 'fermer', es: 'cerrar' } },
  { key: 'confirm', names: { en: 'confirm', nl: 'bevestigen', de: 'Bestätigen', fr: 'confirmer', es: 'confirmar' } },
  { key: 'copy', names: { en: 'copy', nl: 'kopiëren', de: 'Kopieren', fr: 'copier', es: 'copiar' } },
  { key: 'dark', names: { en: 'dark mode', nl: 'donkere modus', de: 'dunkler Modus', fr: 'mode sombre', es: 'modo oscuro' }, unseen: true },
  { key: 'delete', names: { en: 'delete', nl: 'verwijderen', de: 'Löschen', fr: 'supprimer', es: 'eliminar' } },
  { key: 'download', names: { en: 'download', nl: 'downloaden', de: 'Herunterladen', fr: 'télécharger', es: 'descargar' } },
  { key: 'edit', names: { en: 'edit', nl: 'bewerken', de: 'Bearbeiten', fr: 'modifier', es: 'editar' } },
  { key: 'external', names: { en: 'open in a new tab', nl: 'openen in nieuw tabblad', de: 'in neuem Tab öffnen', fr: 'ouvrir dans un nouvel onglet', es: 'abrir en una pestaña nueva' } },
  { key: 'favorite', names: { en: 'favourite', nl: 'favoriet', de: 'Favorit', fr: 'favori', es: 'favorito' } },
  { key: 'filter', names: { en: 'filter', nl: 'filter', de: 'Filter', fr: 'filtre', es: 'filtro' } },
  { key: 'forward', names: { en: 'next', nl: 'volgende', de: 'Weiter', fr: 'suivant', es: 'siguiente' } },
  { key: 'fullscreen', names: { en: 'full screen', nl: 'volledig scherm', de: 'Vollbild', fr: 'plein écran', es: 'pantalla completa' } },
  { key: 'gift', names: { en: 'gift', nl: 'cadeau', de: 'Geschenk', fr: 'cadeau', es: 'regalo' }, unseen: true },
  { key: 'grid', names: { en: 'grid view', nl: 'rasterweergave', de: 'Rasteransicht', fr: 'vue en grille', es: 'vista de cuadrícula' } },
  { key: 'help', names: { en: 'help', nl: 'hulp', de: 'Hilfe', fr: 'aide', es: 'ayuda' } },
  { key: 'hide', names: { en: 'hide', nl: 'verbergen', de: 'Ausblenden', fr: 'masquer', es: 'ocultar' } },
  { key: 'home', names: { en: 'home', nl: 'startpagina', de: 'Startseite', fr: 'accueil', es: 'inicio' } },
  { key: 'image', names: { en: 'picture', nl: 'afbeelding', de: 'Bild', fr: 'image', es: 'imagen' } },
  { key: 'info', names: { en: 'information', nl: 'informatie', de: 'Information', fr: 'information', es: 'información' } },
  { key: 'language', names: { en: 'language', nl: 'taal', de: 'Sprache', fr: 'langue', es: 'idioma' }, unseen: true },
  { key: 'light', names: { en: 'light mode', nl: 'lichte modus', de: 'heller Modus', fr: 'mode clair', es: 'modo claro' } },
  { key: 'link', names: { en: 'link', nl: 'link', de: 'Link', fr: 'lien', es: 'enlace' } },
  { key: 'list', names: { en: 'list view', nl: 'lijstweergave', de: 'Listenansicht', fr: 'vue en liste', es: 'vista de lista' } },
  { key: 'location', names: { en: 'location', nl: 'locatie', de: 'Standort', fr: 'lieu', es: 'ubicación' } },
  { key: 'lock', names: { en: 'lock', nl: 'slot', de: 'Schloss', fr: 'verrou', es: 'candado' } },
  { key: 'login', names: { en: 'sign in', nl: 'inloggen', de: 'Anmelden', fr: 'connexion', es: 'iniciar sesión' } },
  { key: 'logout', names: { en: 'sign out', nl: 'uitloggen', de: 'Abmelden', fr: 'déconnexion', es: 'cerrar sesión' }, unseen: true },
  { key: 'mail', names: { en: 'email', nl: 'e-mail', de: 'E-Mail', fr: 'e-mail', es: 'correo' } },
  { key: 'menu', names: { en: 'menu', nl: 'menu', de: 'Menü', fr: 'menu', es: 'menú' } },
  { key: 'microphone', names: { en: 'microphone', nl: 'microfoon', de: 'Mikrofon', fr: 'microphone', es: 'micrófono' } },
  { key: 'more', names: { en: 'more options', nl: 'meer opties', de: 'weitere Optionen', fr: 'plus d\'options', es: 'más opciones' } },
  { key: 'mute', names: { en: 'mute', nl: 'dempen', de: 'Stumm', fr: 'muet', es: 'silenciar' } },
  { key: 'notifications', names: { en: 'notifications', nl: 'meldingen', de: 'Benachrichtigungen', fr: 'notifications', es: 'notificaciones' } },
  { key: 'pause', names: { en: 'pause', nl: 'pauzeren', de: 'Pause', fr: 'pause', es: 'pausar' } },
  { key: 'payment', names: { en: 'payment', nl: 'betalen', de: 'Zahlung', fr: 'paiement', es: 'pago' } },
  { key: 'phone', names: { en: 'phone call', nl: 'bellen', de: 'Anruf', fr: 'appel', es: 'llamada' } },
  { key: 'play', names: { en: 'play', nl: 'afspelen', de: 'Abspielen', fr: 'lecture', es: 'reproducir' } },
  { key: 'print', names: { en: 'print', nl: 'afdrukken', de: 'Drucken', fr: 'imprimer', es: 'imprimir' }, unseen: true },
  { key: 'refresh', names: { en: 'refresh', nl: 'vernieuwen', de: 'Aktualisieren', fr: 'actualiser', es: 'actualizar' } },
  { key: 'remove', names: { en: 'minus', nl: 'min', de: 'Minus', fr: 'moins', es: 'menos' } },
  { key: 'save', names: { en: 'save', nl: 'opslaan', de: 'Speichern', fr: 'enregistrer', es: 'guardar' } },
  { key: 'search', names: { en: 'search', nl: 'zoeken', de: 'Suche', fr: 'recherche', es: 'buscar' } },
  { key: 'send', names: { en: 'send', nl: 'verzenden', de: 'Senden', fr: 'envoyer', es: 'enviar' } },
  { key: 'settings', names: { en: 'settings', nl: 'instellingen', de: 'Einstellungen', fr: 'paramètres', es: 'ajustes' } },
  { key: 'share', names: { en: 'share', nl: 'delen', de: 'Teilen', fr: 'partager', es: 'compartir' } },
  { key: 'show', names: { en: 'show', nl: 'tonen', de: 'Anzeigen', fr: 'afficher', es: 'mostrar' } },
  { key: 'sound', names: { en: 'sound', nl: 'geluid', de: 'Ton', fr: 'son', es: 'sonido' } },
  { key: 'star', names: { en: 'star', nl: 'ster', de: 'Stern', fr: 'étoile', es: 'estrella' } },
  { key: 'tag', names: { en: 'tag', nl: 'label', de: 'Schlagwort', fr: 'étiquette', es: 'etiqueta' }, unseen: true },
  { key: 'upload', names: { en: 'upload', nl: 'uploaden', de: 'Hochladen', fr: 'téléverser', es: 'subir' } },
  { key: 'warning', names: { en: 'warning', nl: 'waarschuwing', de: 'Warnung', fr: 'avertissement', es: 'advertencia' } },
  { key: 'zoomin', names: { en: 'zoom in', nl: 'inzoomen', de: 'Vergrößern', fr: 'zoom avant', es: 'acercar' }, unseen: true },
  { key: 'zoomout', names: { en: 'zoom out', nl: 'uitzoomen', de: 'Verkleinern', fr: 'zoom arrière', es: 'alejar' } },
]

// The second question about a button the model named (is it really a button for that job?), in two of the
// wordings it was trained on: how it is asked, and what a yes looks like. {name} is the job's name.
export const CONTROL_ASK: Record<ControlLang, string> = {
  en: 'Is this a button for {name}?',
  nl: 'Is dit een knop voor {name}?',
  de: 'Ist das eine Schaltfläche für {name}?',
  fr: 'Est-ce un bouton pour {name} ?',
  es: '¿Es un botón para {name}?',
}
export const CONTROL_IS: Record<ControlLang, string> = {
  en: 'A button for: {name}.',
  nl: 'Een knop voor: {name}.',
  de: 'Eine Schaltfläche für: {name}.',
  fr: 'Un bouton pour : {name}.',
  es: 'Un botón para: {name}.',
}
