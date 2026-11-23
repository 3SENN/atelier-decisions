# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pool v4: thousands of distinct "does this request fit this skill?" pairs, from MASSIVE's 60 intents written up as
skills. Real, varied utterances in five languages teach matching by meaning; intents from the same scenario
(alarm_set vs alarm_query) are the hard negatives. The teacher labels.

    python data/build_skills_massive.py --out data/pool/pool-v4.jsonl --n 3000
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atelier_uses as U  # noqa: E402

ROOT = Path(__file__).resolve().parent
LOCALES = {"en": "en-US", "nl": "nl-NL", "de": "de-DE", "fr": "fr-FR", "es": "es-ES"}
# name, description: written the way people describe their own skills, not the way the intents are labelled.
SKILLS = {
    "alarm_query": ("my-alarms", "Tells me which alarms I have set and when they go off."),
    "alarm_remove": ("clear-alarm", "Cancels or deletes alarms I no longer need."),
    "alarm_set": ("wake-me", "Sets an alarm or wake-up call for a time I say."),
    "audio_volume_down": ("quieter", "Turns the sound down."),
    "audio_volume_mute": ("mute", "Silences the speaker completely."),
    "audio_volume_other": ("volume", "Changes or reports the volume level."),
    "audio_volume_up": ("louder", "Turns the sound up."),
    "calendar_query": ("whats-on", "Looks up what is in my agenda: meetings, appointments, birthdays."),
    "calendar_remove": ("cancel-event", "Removes an appointment or meeting from my agenda."),
    "calendar_set": ("add-event", "Puts a new appointment, meeting or reminder in my agenda."),
    "cooking_query": ("kitchen-help", "Answers cooking questions: temperatures, timings, substitutes."),
    "cooking_recipe": ("recipes", "Finds a recipe and walks me through making it."),
    "datetime_convert": ("time-zones", "Converts times and dates between time zones."),
    "datetime_query": ("date-time", "Tells the current time, date or day of the week."),
    "email_addcontact": ("save-contact", "Adds a person's email address to my contacts."),
    "email_query": ("check-mail", "Checks my inbox for new or specific emails."),
    "email_querycontact": ("contact-info", "Looks up details of someone in my contacts."),
    "email_sendemail": ("send-mail", "Writes and sends an email for me."),
    "general_greet": ("chit-chat", "Small talk: greetings and how-are-you."),
    "general_joke": ("jokes", "Tells a joke when I need a laugh."),
    "general_quirky": ("random-questions", "Handles odd, playful or off-topic questions."),
    "iot_cleaning": ("robot-vacuum", "Starts or stops the robot vacuum cleaner."),
    "iot_coffee": ("coffee-maker", "Makes coffee on the smart coffee machine."),
    "iot_hue_lightchange": ("light-colour", "Changes the colour of the smart lights."),
    "iot_hue_lightdim": ("dim-lights", "Dims the smart lights."),
    "iot_hue_lightoff": ("lights-off", "Switches the lights off."),
    "iot_hue_lighton": ("lights-on", "Switches the lights on."),
    "iot_hue_lightup": ("brighter", "Makes the lights brighter."),
    "iot_wemo_off": ("plug-off", "Switches a smart plug off."),
    "iot_wemo_on": ("plug-on", "Switches a smart plug on."),
    "lists_createoradd": ("add-to-list", "Adds items to a list, like groceries or to-dos, or starts a new list."),
    "lists_query": ("read-list", "Reads out what is on one of my lists."),
    "lists_remove": ("remove-from-list", "Takes items off a list or deletes a list."),
    "music_dislikeness": ("skip-dislikes", "Remembers songs or artists I don't like."),
    "music_likeness": ("music-taste", "Remembers music I like."),
    "music_query": ("whats-playing", "Tells me which song or artist is playing."),
    "music_settings": ("playback", "Shuffle, repeat and other playback settings."),
    "news_query": ("news", "Gives me the latest news and headlines."),
    "play_audiobook": ("audiobooks", "Plays or resumes an audiobook."),
    "play_game": ("games", "Starts a game to play together."),
    "play_music": ("play-music", "Plays songs, artists, albums or playlists."),
    "play_podcasts": ("podcasts", "Plays podcast episodes."),
    "play_radio": ("radio", "Tunes in to a radio station."),
    "qa_currency": ("exchange-rates", "Converts money between currencies."),
    "qa_definition": ("dictionary", "Explains what a word means."),
    "qa_factoid": ("quick-facts", "Answers general knowledge questions."),
    "qa_maths": ("calculator", "Does sums and calculations."),
    "qa_stock": ("stocks", "Gives share prices and stock market news."),
    "recommendation_events": ("events-nearby", "Suggests events and things to do."),
    "recommendation_locations": ("places-nearby", "Recommends shops, restaurants and places near me."),
    "recommendation_movies": ("movie-picks", "Recommends films to watch."),
    "social_post": ("post-social", "Posts updates or complaints on social media for me."),
    "social_query": ("social-updates", "Checks what friends posted on social media."),
    "takeaway_order": ("order-food", "Orders takeaway food for delivery."),
    "takeaway_query": ("takeaway-info", "Checks takeaway menus, delivery times or order status."),
    "transport_query": ("journey-planner", "Looks up train and bus times and routes."),
    "transport_taxi": ("taxi", "Books a taxi or ride."),
    "transport_ticket": ("tickets", "Buys train or bus tickets."),
    "transport_traffic": ("traffic", "Tells me about traffic and jams on my route."),
    "weather_query": ("weather", "Gives the weather forecast."),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    by_intent: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for code, locale in LOCALES.items():
        with (ROOT / "cache" / "massive" / "1.1" / "data" / f"{locale}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                by_intent[row["intent"]][code].append(row["utt"])
    intents = sorted(SKILLS)
    scenario = lambda i: i.split("_")[0]  # noqa: E731
    rows, seen = [], set()
    while len(rows) < args.n:
        code = rng.choices(list(LOCALES), [0.4, 0.3, 0.1, 0.1, 0.1])[0]
        target = rng.choice(intents)
        message = rng.choice(by_intent[target][code])
        siblings = [i for i in intents if scenario(i) == scenario(target) and i != target]
        k = rng.choice([3, 5, 8])
        pool = rng.sample(siblings, min(len(siblings), rng.randint(0, 2))) + rng.sample(intents, k)
        chunk = list(dict.fromkeys([i for i in pool if i != target]))[: k - 1]
        if rng.random() < 0.6:  # the fitting skill is offered
            chunk.insert(rng.randrange(len(chunk) + 1), target)
        skills = [{"name": SKILLS[i][0], "description": SKILLS[i][1]} for i in chunk]
        request = U.skills(message, skills)
        key = json.dumps(request, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"meta": {"use": "skills", "source": "massive-intents", "lang": code}, "request": request})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} requests, {sum(len(r['request']['questions']) for r in rows)} questions -> {args.out}")


if __name__ == "__main__":
    main()
