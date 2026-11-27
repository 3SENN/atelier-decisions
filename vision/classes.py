# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""The things in photos atelier-vision is asked about, in the five languages its questions are written in.

Each line: an Open Images class (its display name there), the album it would go in, and how a person would name it
in "the photo shows ..." in English, Dutch, German (nominative), French and Spanish, with the article a sentence
needs ("a dog", "bread", "french fries"). bare() drops the article for a list of options.

The classes are the ones with enough human-checked labels in Open Images' validation set to measure on (15 or
more photos), that a person would sort photos by, and that are one clear thing: not "Clothing", "Wheel", "Person" or
"Tree", which are in half of all photos whatever they are of. The names and translations were written for this
project by Claude (Anthropic) and are part of its data (CC BY 4.0); the labels are Open Images' (CC BY 4.0).
"""

from __future__ import annotations

LANGS = ["en", "nl", "de", "fr", "es"]

_TABLE = """
Dog|animals|a dog|een hond|ein Hund|un chien|un perro
Cat|animals|a cat|een kat|eine Katze|un chat|un gato
Bird|animals|a bird|een vogel|ein Vogel|un oiseau|un pájaro
Horse|animals|a horse|een paard|ein Pferd|un cheval|un caballo
Fish|animals|a fish|een vis|ein Fisch|un poisson|un pez
Cattle|animals|cows|koeien|Kühe|des vaches|vacas
Sheep|animals|a sheep|een schaap|ein Schaf|un mouton|una oveja
Monkey|animals|a monkey|een aap|ein Affe|un singe|un mono
Lizard|animals|a lizard|een hagedis|eine Eidechse|un lézard|un lagarto
Snake|animals|a snake|een slang|eine Schlange|un serpent|una serpiente
Rabbit|animals|a rabbit|een konijn|ein Kaninchen|un lapin|un conejo
Butterfly|animals|a butterfly|een vlinder|ein Schmetterling|un papillon|una mariposa
Elephant|animals|an elephant|een olifant|ein Elefant|un éléphant|un elefante
Lion|animals|a lion|een leeuw|ein Löwe|un lion|un león
Bear|animals|a bear|een beer|ein Bär|un ours|un oso
Penguin|animals|a penguin|een pinguïn|ein Pinguin|un manchot|un pingüino
Parrot|animals|a parrot|een papegaai|ein Papagei|un perroquet|un loro
Squirrel|animals|a squirrel|een eekhoorn|ein Eichhörnchen|un écureuil|una ardilla
Frog|animals|a frog|een kikker|ein Frosch|une grenouille|una rana
Spider|animals|a spider|een spin|eine Spinne|une araignée|una araña
Dolphin|animals|a dolphin|een dolfijn|ein Delfin|un dauphin|un delfín
Whale|animals|a whale|een walvis|ein Wal|une baleine|una ballena
Shark|animals|a shark|een haai|ein Hai|un requin|un tiburón
Goat|animals|a goat|een geit|eine Ziege|une chèvre|una cabra
Pig|animals|a pig|een varken|ein Schwein|un cochon|un cerdo
Chicken|animals|a chicken|een kip|ein Huhn|une poule|una gallina
Owl|animals|an owl|een uil|eine Eule|un hibou|un búho
Eagle|animals|an eagle|een adelaar|ein Adler|un aigle|un águila
Tortoise|animals|a tortoise|een schildpad|eine Schildkröte|une tortue|una tortuga
Crab|animals|a crab|een krab|eine Krabbe|un crabe|un cangrejo
Jellyfish|animals|a jellyfish|een kwal|eine Qualle|une méduse|una medusa
Swan|animals|a swan|een zwaan|ein Schwan|un cygne|un cisne
Goose|animals|a goose|een gans|eine Gans|une oie|un ganso
Camel|animals|a camel|een kameel|ein Kamel|un chameau|un camello
Zebra|animals|a zebra|een zebra|ein Zebra|un zèbre|una cebra
Giraffe|animals|a giraffe|een giraffe|eine Giraffe|une girafe|una jirafa
Tiger|animals|a tiger|een tijger|ein Tiger|un tigre|un tigre
Deer|animals|a deer|een hert|ein Hirsch|un cerf|un ciervo
Duck|animals|a duck|een eend|eine Ente|un canard|un pato
Crocodile|animals|a crocodile|een krokodil|ein Krokodil|un crocodile|un cocodrilo
Fox|animals|a fox|een vos|ein Fuchs|un renard|un zorro
Ostrich|animals|an ostrich|een struisvogel|ein Strauß|une autruche|un avestruz
Kangaroo|animals|a kangaroo|een kangoeroe|ein Känguru|un kangourou|un canguro
Hedgehog|animals|a hedgehog|een egel|ein Igel|un hérisson|un erizo
Starfish|animals|a starfish|een zeester|ein Seestern|une étoile de mer|una estrella de mar
Cake|food|a cake|een taart|ein Kuchen|un gâteau|una tarta
Pizza|food|a pizza|een pizza|eine Pizza|une pizza|una pizza
Hamburger|food|a hamburger|een hamburger|ein Hamburger|un hamburger|una hamburguesa
Bread|food|bread|brood|Brot|du pain|pan
Sandwich|food|a sandwich|een sandwich|ein Sandwich|un sandwich|un sándwich
Ice cream|food|ice cream|ijs|Eis|une glace|helado
Cheese|food|cheese|kaas|Käse|du fromage|queso
Cookie|food|cookies|koekjes|Kekse|des biscuits|galletas
Pancake|food|pancakes|pannenkoeken|Pfannkuchen|des crêpes|tortitas
Doughnut|food|a doughnut|een donut|ein Donut|un beignet|una rosquilla
Cupcake|food|a cupcake|een cupcake|ein Cupcake|un cupcake|una magdalena decorada
French fries|food|french fries|friet|Pommes frites|des frites|patatas fritas
Salad|food|a salad|een salade|ein Salat|une salade|una ensalada
Soup|food|soup|soep|Suppe|de la soupe|sopa
Noodle|food|noodles|noedels|Nudeln|des nouilles|fideos
Rice|food|rice|rijst|Reis|du riz|arroz
Egg|food|an egg|een ei|ein Ei|un œuf|un huevo
Hot dog|food|a hot dog|een hotdog|ein Hotdog|un hot-dog|un perrito caliente
Croissant|food|a croissant|een croissant|ein Croissant|un croissant|un cruasán
Apple|food|an apple|een appel|ein Apfel|une pomme|una manzana
Orange (fruit)|food|an orange|een sinaasappel|eine Orange|une orange|una naranja
Strawberry|food|strawberries|aardbeien|Erdbeeren|des fraises|fresas
Lemon (plant)|food|a lemon|een citroen|eine Zitrone|un citron|un limón
Tomato|food|tomatoes|tomaten|Tomaten|des tomates|tomates
Banana|food|a banana|een banaan|eine Banane|une banane|un plátano
Watermelon|food|a watermelon|een watermeloen|eine Wassermelone|une pastèque|una sandía
Grape|food|grapes|druiven|Weintrauben|du raisin|uvas
Carrot|food|carrots|wortels|Karotten|des carottes|zanahorias
Broccoli|food|broccoli|broccoli|Brokkoli|du brocoli|brócoli
Mushroom|food|mushrooms|paddenstoelen|Pilze|des champignons|setas
Pumpkin|food|a pumpkin|een pompoen|ein Kürbis|une citrouille|una calabaza
Coffee cup|drinks|a cup of coffee|een kop koffie|eine Tasse Kaffee|une tasse de café|una taza de café
Wine glass|drinks|a glass of wine|een glas wijn|ein Glas Wein|un verre de vin|una copa de vino
Beer|drinks|beer|bier|Bier|de la bière|cerveza
Cocktail|drinks|a cocktail|een cocktail|ein Cocktail|un cocktail|un cóctel
Juice|drinks|juice|sap|Saft|du jus|zumo
Tea|drinks|tea|thee|Tee|du thé|té
Car|vehicles|a car|een auto|ein Auto|une voiture|un coche
Bicycle|vehicles|a bicycle|een fiets|ein Fahrrad|un vélo|una bicicleta
Motorcycle|vehicles|a motorcycle|een motor|ein Motorrad|une moto|una moto
Bus|vehicles|a bus|een bus|ein Bus|un bus|un autobús
Train|vehicles|a train|een trein|ein Zug|un train|un tren
Truck|vehicles|a truck|een vrachtwagen|ein Lastwagen|un camion|un camión
Boat|vehicles|a boat|een boot|ein Boot|un bateau|un barco
Airplane|vehicles|an airplane|een vliegtuig|ein Flugzeug|un avion|un avión
Helicopter|vehicles|a helicopter|een helikopter|ein Hubschrauber|un hélicoptère|un helicóptero
Taxi|vehicles|a taxi|een taxi|ein Taxi|un taxi|un taxi
Van|vehicles|a van|een busje|ein Lieferwagen|une camionnette|una furgoneta
Canoe|vehicles|a canoe|een kano|ein Kanu|un canoë|una canoa
Tank|vehicles|a tank|een tank|ein Panzer|un char d'assaut|un tanque
Ambulance|vehicles|an ambulance|een ambulance|ein Krankenwagen|une ambulance|una ambulancia
Tram|vehicles|a tram|een tram|eine Straßenbahn|un tramway|un tranvía
Rocket|vehicles|a rocket|een raket|eine Rakete|une fusée|un cohete
Mountain|landscape|mountains|bergen|Berge|des montagnes|montañas
Snow|landscape|snow|sneeuw|Schnee|de la neige|nieve
Beach|landscape|a beach|een strand|ein Strand|une plage|una playa
Sea|landscape|the sea|de zee|das Meer|la mer|el mar
River|landscape|a river|een rivier|ein Fluss|une rivière|un río
Waterfall|landscape|a waterfall|een waterval|ein Wasserfall|une cascade|una cascada
Lake|landscape|a lake|een meer|ein See|un lac|un lago
Cave|landscape|a cave|een grot|eine Höhle|une grotte|una cueva
Desert|landscape|a desert|een woestijn|eine Wüste|un désert|un desierto
Fire|landscape|fire|vuur|Feuer|du feu|fuego
Fireworks|landscape|fireworks|vuurwerk|Feuerwerk|un feu d'artifice|fuegos artificiales
Moon|landscape|the moon|de maan|der Mond|la lune|la luna
Flower|plants|flowers|bloemen|Blumen|des fleurs|flores
Rose|plants|a rose|een roos|eine Rose|une rose|una rosa
Cactus|plants|a cactus|een cactus|ein Kaktus|un cactus|un cactus
Houseplant|plants|a houseplant|een kamerplant|eine Zimmerpflanze|une plante d'intérieur|una planta de interior
Common sunflower|plants|a sunflower|een zonnebloem|eine Sonnenblume|un tournesol|un girasol
Christmas tree|plants|a Christmas tree|een kerstboom|ein Weihnachtsbaum|un sapin de Noël|un árbol de Navidad
Skyscraper|buildings|a skyscraper|een wolkenkrabber|ein Wolkenkratzer|un gratte-ciel|un rascacielos
Tower|buildings|a tower|een toren|ein Turm|une tour|una torre
Bridge (structure)|buildings|a bridge|een brug|eine Brücke|un pont|un puente
Stadium|buildings|a stadium|een stadion|ein Stadion|un stade|un estadio
Lighthouse|buildings|a lighthouse|een vuurtoren|ein Leuchtturm|un phare|un faro
Stairs|buildings|stairs|een trap|eine Treppe|un escalier|una escalera
Tent|buildings|a tent|een tent|ein Zelt|une tente|una tienda de campaña
Fountain|buildings|a fountain|een fontein|ein Springbrunnen|une fontaine|una fuente
Guitar|music|a guitar|een gitaar|eine Gitarre|une guitare|una guitarra
Piano|music|a piano|een piano|ein Klavier|un piano|un piano
Violin|music|a violin|een viool|eine Geige|un violon|un violín
Saxophone|music|a saxophone|een saxofoon|ein Saxophon|un saxophone|un saxofón
Cello|music|a cello|een cello|ein Cello|un violoncelle|un violonchelo
Drums (Music)|music|drums|een drumstel|ein Schlagzeug|une batterie|una batería
Microphone|music|a microphone|een microfoon|ein Mikrofon|un microphone|un micrófono
Headphones|music|headphones|een koptelefoon|Kopfhörer|un casque audio|auriculares
Football|sports|a football|een voetbal|ein Fußball|un ballon de football|un balón de fútbol
Tennis racket|sports|a tennis racket|een tennisracket|ein Tennisschläger|une raquette de tennis|una raqueta de tenis
Surfboard|sports|a surfboard|een surfplank|ein Surfbrett|une planche de surf|una tabla de surf
Ski|sports|skis|ski's|Skier|des skis|esquís
Skateboard|sports|a skateboard|een skateboard|ein Skateboard|un skateboard|un monopatín
Parachute|sports|a parachute|een parachute|ein Fallschirm|un parachute|un paracaídas
Kite|sports|a kite|een vlieger|ein Drachen|un cerf-volant|una cometa
Billiard table|sports|a pool table|een biljarttafel|ein Billardtisch|une table de billard|una mesa de billar
Chess|sports|a chess game|een schaakspel|ein Schachspiel|un jeu d'échecs|un ajedrez
Dice|sports|dice|dobbelstenen|Würfel|des dés|dados
Chair|home|a chair|een stoel|ein Stuhl|une chaise|una silla
Couch|home|a couch|een bank|ein Sofa|un canapé|un sofá
Bed|home|a bed|een bed|ein Bett|un lit|una cama
Toilet|home|a toilet|een toilet|eine Toilette|des toilettes|un inodoro
Sink|home|a sink|een wastafel|ein Waschbecken|un lavabo|un lavabo
Refrigerator|home|a refrigerator|een koelkast|ein Kühlschrank|un réfrigérateur|un frigorífico
Washing machine|home|a washing machine|een wasmachine|eine Waschmaschine|un lave-linge|una lavadora
Bathtub|home|a bathtub|een badkuip|eine Badewanne|une baignoire|una bañera
Fireplace|home|a fireplace|een open haard|ein Kamin|une cheminée|una chimenea
Lamp|home|a lamp|een lamp|eine Lampe|une lampe|una lámpara
Candle|home|a candle|een kaars|eine Kerze|une bougie|una vela
Mirror|home|a mirror|een spiegel|ein Spiegel|un miroir|un espejo
Clock|home|a clock|een klok|eine Uhr|une horloge|un reloj de pared
Vase|home|a vase|een vaas|eine Vase|un vase|un jarrón
Frying pan|home|a frying pan|een koekenpan|eine Bratpfanne|une poêle|una sartén
Laptop|tech|a laptop|een laptop|ein Laptop|un ordinateur portable|un portátil
Mobile phone|tech|a mobile phone|een mobiele telefoon|ein Handy|un téléphone portable|un teléfono móvil
Computer keyboard|tech|a computer keyboard|een toetsenbord|eine Computertastatur|un clavier d'ordinateur|un teclado de ordenador
Television|tech|a television|een televisie|ein Fernseher|une télévision|un televisor
Camera|tech|a camera|een camera|eine Kamera|un appareil photo|una cámara
Computer monitor|tech|a computer monitor|een beeldscherm|ein Computermonitor|un écran d'ordinateur|un monitor de ordenador
Computer mouse|tech|a computer mouse|een computermuis|eine Computermaus|une souris d'ordinateur|un ratón de ordenador
Printer|tech|a printer|een printer|ein Drucker|une imprimante|una impresora
Calculator|tech|a calculator|een rekenmachine|ein Taschenrechner|une calculatrice|una calculadora
Robot|tech|a robot|een robot|ein Roboter|un robot|un robot
Typewriter|tech|a typewriter|een typemachine|eine Schreibmaschine|une machine à écrire|una máquina de escribir
Hat|wear|a hat|een hoed|ein Hut|un chapeau|un sombrero
Glasses|wear|glasses|een bril|eine Brille|des lunettes|gafas
Boot|wear|boots|laarzen|Stiefel|des bottes|botas
High heels|wear|high heels|hoge hakken|Stöckelschuhe|des talons hauts|zapatos de tacón
Tie|wear|a tie|een stropdas|eine Krawatte|une cravate|una corbata
Handbag|wear|a handbag|een handtas|eine Handtasche|un sac à main|un bolso
Backpack|wear|a backpack|een rugzak|ein Rucksack|un sac à dos|una mochila
Umbrella|wear|an umbrella|een paraplu|ein Regenschirm|un parapluie|un paraguas
Watch|wear|a wristwatch|een horloge|eine Armbanduhr|une montre|un reloj de pulsera
Helmet|wear|a helmet|een helm|ein Helm|un casque|un casco
Suitcase|wear|a suitcase|een koffer|ein Koffer|une valise|una maleta
Poster|pictures|a poster|een poster|ein Plakat|une affiche|un cartel
Painting|pictures|a painting|een schilderij|ein Gemälde|un tableau|un cuadro
Drawing|pictures|a drawing|een tekening|eine Zeichnung|un dessin|un dibujo
Cartoon|pictures|a cartoon|een tekenfilmfiguur|eine Zeichentrickfigur|un dessin animé|un dibujo animado
Sculpture|pictures|a sculpture|een beeldhouwwerk|eine Skulptur|une sculpture|una escultura
Graffiti|pictures|graffiti|graffiti|Graffiti|des graffitis|grafitis
Logo|pictures|a logo|een logo|ein Logo|un logo|un logotipo
Map|pictures|a map|een kaart|eine Landkarte|une carte géographique|un mapa
Screenshot|pictures|a screenshot|een schermafbeelding|ein Bildschirmfoto|une capture d'écran|una captura de pantalla
Book|pictures|a book|een boek|ein Buch|un livre|un libro
Newspaper|pictures|a newspaper|een krant|eine Zeitung|un journal|un periódico
Calendar|pictures|a calendar|een kalender|ein Kalender|un calendrier|un calendario
Banknote|pictures|banknotes|bankbiljetten|Geldscheine|des billets de banque|billetes
Coin|pictures|coins|munten|Münzen|des pièces de monnaie|monedas
Whiteboard|pictures|a whiteboard|een whiteboard|ein Whiteboard|un tableau blanc|una pizarra blanca
Envelope|pictures|an envelope|een envelop|ein Briefumschlag|une enveloppe|un sobre
Flag|pictures|a flag|een vlag|eine Flagge|un drapeau|una bandera
Traffic sign|pictures|a traffic sign|een verkeersbord|ein Verkehrsschild|un panneau de signalisation|una señal de tráfico
Balloon|pictures|balloons|ballonnen|Luftballons|des ballons|globos
Teddy bear|pictures|a teddy bear|een teddybeer|ein Teddybär|un ours en peluche|un oso de peluche
Doll|pictures|a doll|een pop|eine Puppe|une poupée|una muñeca
X-ray|pictures|an X-ray|een röntgenfoto|ein Röntgenbild|une radiographie|una radiografía
Globe|pictures|a globe|een wereldbol|ein Globus|un globe terrestre|un globo terráqueo
Playing card|pictures|playing cards|speelkaarten|Spielkarten|des cartes à jouer|naipes
"""

# What each album holds, as a person would describe it to someone sorting photos.
ALBUMS = {
    "animals": ("Animals and pets", "Dieren en huisdieren", "Tiere und Haustiere", "Animaux", "Animales y mascotas"),
    "food": ("Food and meals", "Eten en maaltijden", "Essen und Mahlzeiten", "Nourriture et repas", "Comida y platos"),
    "drinks": ("Drinks", "Drankjes", "Getränke", "Boissons", "Bebidas"),
    "vehicles": ("Cars, trains, boats and other vehicles", "Auto's, treinen, boten en andere voertuigen", "Autos, Züge, Boote und andere Fahrzeuge",
                 "Voitures, trains, bateaux et autres véhicules", "Coches, trenes, barcos y otros vehículos"),
    "landscape": ("Landscapes and nature", "Landschappen en natuur", "Landschaften und Natur", "Paysages et nature", "Paisajes y naturaleza"),
    "plants": ("Flowers and plants", "Bloemen en planten", "Blumen und Pflanzen", "Fleurs et plantes", "Flores y plantas"),
    "buildings": ("Buildings and landmarks", "Gebouwen en bezienswaardigheden", "Gebäude und Sehenswürdigkeiten", "Bâtiments et monuments", "Edificios y monumentos"),
    "music": ("Music and instruments", "Muziek en instrumenten", "Musik und Instrumente", "Musique et instruments", "Música e instrumentos"),
    "sports": ("Sports and games", "Sport en spel", "Sport und Spiele", "Sports et jeux", "Deportes y juegos"),
    "home": ("Home and furniture", "Huis en meubels", "Zuhause und Möbel", "Maison et meubles", "Casa y muebles"),
    "tech": ("Computers, phones and other devices", "Computers, telefoons en andere apparaten", "Computer, Handys und andere Geräte",
             "Ordinateurs, téléphones et autres appareils", "Ordenadores, teléfonos y otros aparatos"),
    "wear": ("Clothes and accessories", "Kleding en accessoires", "Kleidung und Accessoires", "Vêtements et accessoires", "Ropa y accesorios"),
    "pictures": ("Pictures, papers and signs", "Afbeeldingen, papieren en borden", "Bilder, Papiere und Schilder", "Images, papiers et panneaux", "Imágenes, papeles y señales"),
}

# Classes never trained on: the measure of things nobody taught it. Chosen before any training, a few from every album.
HELD_OUT = {
    "Horse", "Penguin", "Frog", "Shark", "Owl", "Giraffe", "Hedgehog", "Pizza", "Pancake", "Egg", "Banana", "Mushroom", "Cocktail", "Motorcycle", "Helicopter", "Tram",
    "Waterfall", "Desert", "Fireworks", "Cactus", "Lighthouse", "Tent", "Violin", "Drums (Music)", "Surfboard", "Chess", "Bathtub", "Candle", "Camera", "Printer",
    "Umbrella", "Helmet", "Painting", "Map", "Banknote", "Envelope", "Teddy bear",
}

_ARTICLES = {
    "en": ("a ", "an ", "the "),
    "nl": ("een ", "de ", "het "),
    "de": ("ein ", "eine ", "der ", "die ", "das "),
    "fr": ("un ", "une ", "du ", "de la ", "de l'", "des ", "le ", "la ", "les ", "l'"),
    "es": ("un ", "una ", "unos ", "unas ", "el ", "la ", "los ", "las "),
}


def bare(phrase: str, lang: str) -> str:
    """The phrase without its article: "a dog" to "dog", "du pain" to "pain"."""
    for article in _ARTICLES[lang]:
        if phrase.startswith(article):
            return phrase[len(article):]
    return phrase


def classes() -> dict[str, dict]:
    """Open Images display name to {"album", "en", "nl", "de", "fr", "es"}."""
    out = {}
    for line in _TABLE.strip().splitlines():
        name, album, *phrases = line.split("|")
        if len(phrases) != len(LANGS) or album not in ALBUMS or name in out:
            raise ValueError(f"bad line in the class table: {line}")
        out[name] = {"album": album, **dict(zip(LANGS, phrases))}
    missing = HELD_OUT - set(out)
    if missing:
        raise ValueError(f"held-out classes not in the table: {sorted(missing)}")
    return out


if __name__ == "__main__":
    table = classes()
    by_album: dict[str, int] = {}
    for c in table.values():
        by_album[c["album"]] = by_album.get(c["album"], 0) + 1
    print(f"{len(table)} classes, {len(HELD_OUT)} held out; per album: {by_album}")
