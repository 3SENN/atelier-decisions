# atelier-vision v0.1: a decision model that looks

The first saved version of atelier-vision, 2026-11-27. Atelier's decision models read text; a browser is full of
things that are not text: a button that is only an icon, a photo, a scanned receipt, a page with a notice over it.
atelier-vision answers the same System One questions about a picture, on the device.

It writes nothing. Every answer of a question is a short text ("A receipt from a shop or a restaurant", "Opens the
shopping cart"), and the model says how well the picture matches each one:

| Question | The texts it matches | Answer |
|---|---|---|
| choice | one per option: its description, or the option's own name | a probability per option |
| noul | what a yes looks like (`criteria.true`, else the question itself); what a no looks like when `criteria.false` says | the probability of yes |
| score | | declined: no version has been measured on levels yet |

A picture is looked at once: its vector answers every later question about it, and finds it by a description. What
it isn't sure of, Atelier's rule or the companion decides.

| | atelier-vision v0.1 (`model/`) |
|---|---|
| Architecture | SigLIP 2's two towers (`google/siglip2-base-patch16-256`, Apache-2.0): an image tower of 93M parameters (256 patches of 16 x 16 pixels), a text tower of 86M with an embedding table cut from 256,000 tokens to 118,947 |
| Training | LoRA of rank 16 on the linear layers of each tower's last six blocks (2.65M parameters), merged into the weights. Two passes on an M2 (`eval/vision-v1-train.txt`, `eval/vision-v2-train.txt`): the first on 22,466 pictures (53 minutes), the second on 23,906 with the table and tokenizer as shipped, buttons that are none of the jobs, and the question Atelier asks (all jobs at once), 74 minutes. Photos: Open Images V7; buttons, documents and pages: drawn by a browser engine from nine open icon sets and generated content in five languages. No teacher: every answer was checked by people or known by construction |
| Shipped files | `image.q8.onnx` 105.8 MB and `text.q8.emb4.onnx` 147.9 MB (every layer's weights at 8 bits, the embedding table at 4), `tokenizer.json` 6.3 MB, `vision.json` (the calibration): 260 MB |
| Per picture, M2 | about 0.3 s on 2 CPU threads, 0.2 s on 4 (ONNX Runtime; `eval/vision-bench-v0.1.txt`). A text not read before 0.1 s; a question whose texts were read before, under a millisecond |
| Memory | about 0.6 GB once both towers are loaded, up to about 1 GB while many new texts are read at once |
| To train on from | `vision.pt` (the adapters; the base is named with its revision in `vision.json`): `vision/train.py --init releases/vision-v0.1/model` |

**Checks:**
- **The two runtimes:** `runtime/node/src/vision` makes the training code's tower input (to 6e-15), the same token
  ids on 39 of 39 texts, declines the same ones, and gives the same vectors (cosine 0.999999) and the same 12 of 12
  answers (`model/parity.jsonl`, `npm run vision:parity`).
- **The shrunk files against full precision** (`model/vision.json`): pictures 0.9996 (lowest 0.9987 of 96), texts
  0.9999 (lowest 0.9995) read through the table as shipped.
- **How a picture arrives** (`eval/vision-robust-v0.1.txt`, 100 photos from their originals and 100 buttons): made
  smaller first, as Atelier does above 768 pixels, changes 0.6% of the photos' readings and none of the buttons';
  another resize filter 1-2.5%; a rough JPEG (quality 60) 1% for photos and 5% for buttons, which are small and
  sharp.

## Results on held-out questions (`eval/vision-v0.1.json`)

Through the shipped files, read the way Atelier reads answers: a choice when its option has 0.75, a yes at 0.7, a no
at 0.3. Half of the held-out pictures are what the calibration was fitted on; these are the other half. The base is
SigLIP 2 as it comes, calibrated the same way (`eval/vision-base.json`).

| Held-out questions | Questions | Right | Decides | Right when deciding | Sure and wrong |
|---|---|---|---|---|---|
| **All** | 22,625 | **91.2%** | **89.6%** | **95.4%** | **4.1%** |
| SigLIP 2 as it comes, calibrated the same way | 22,625 | 74.2% | 56.4% | 88.5% | 6.5% |
| All, with a wording never trained on | 9,410 | 88.0% | 86.3% | 93.3% | 5.8% |

By kind of picture ("drawn a way never seen" is three whole icon sets, two page layouts, a language a kind's text
was never in; "never trained on" is a thing that was never an answer or a wrong option in training). The base's
numbers are in brackets.

| Picture | Question | Right | Decides | Right when deciding | Sure and wrong |
|---|---|---|---|---|---|
| Photo never seen | which of these things (choice) | 94.7% (91.2) | 91.0% (86.7) | 97.7% (95.3) | 2.1% (4.1) |
| | is there a ... (yes/no) | 87.6% (72.6) | 88.3% (57.9) | 91.9% (81.6) | 7.1% (10.7) |
| | which album (13) | 95.7% (85.3) | 92.3% (65.3) | 98.2% (94.2) | 1.6% (3.8) |
| Photo of a thing never trained on | choice | 91.8% (90.8) | 88.5% (90.5) | 95.2% (93.8) | 4.3% (5.6) |
| | yes/no | 82.4% (80.5) | 86.3% (64.1) | 88.4% (86.5) | 10.0% (8.7) |
| Button, icon set never seen | which job (choice among 3-8) | 97.8% (70.5) | 96.4% (38.6) | 99.4% (95.2) | 0.6% (1.9) |
| | is it a button for ... (yes/no) | 96.6% (69.4) | 94.8% (42.3) | 98.3% (79.6) | 1.6% (8.6) |
| Button with a job never trained on | choice | 87.7% (68.9) | 78.5% (46.1) | 96.7% (92.4) | 2.6% (3.5) |
| | yes/no | 80.3% (64.5) | 86.6% (43.4) | 84.8% (77.3) | 13.2% (9.9) |
| Document, in a language its kind was never in | which kind (choice) | 100% (94.5) | 100% (83.5) | 100% (99.6) | 0.0% (0.4) |
| | yes/no | 99.5% (87.8) | 99.1% (73.2) | 100% (95.1) | 0.0% (3.6) |
| Document of a kind never trained on | choice | 98.6% (98.4) | 96.9% (92.0) | 99.4% (100) | 0.5% (0.0) |
| | yes/no | 96.9% (91.9) | 96.2% (77.5) | 98.4% (97.3) | 1.5% (2.1) |
| Page, layout never seen | which state (choice) | 88.4% (42.8) | 82.5% (16.9) | 96.1% (78.0) | 3.2% (3.7) |
| | yes/no | 86.5% (56.2) | 85.0% (39.1) | 91.8% (61.1) | 7.0% (15.2) |
| Page in a state never trained on (a robot check, a paywall) | choice | 54.2% (49.2) | 49.2% (28.8) | 67.2% (81.3) | **16.2%** (5.4) |
| | yes/no | 62.1% (52.3) | 67.3% (49.0) | 68.9% (55.7) | **21.0%** (21.7) |

By the language of the question: English 91.8% right, Spanish 91.7%, French 91.4%, Dutch 90.9%, German 89.7%.

**Calibration** (`vision/calibrate.py`, in `model/vision.json`): a temperature for a choice (2.54; 1.99 from 16
options), and a weight and an offset for a yes/no (0.32 and 2.93 with only a "true" text; 0.34, 0.00 and 3.13 with a
"false" text too: what a no looks like turns out to add nothing). Fitted on one half of the held-out pictures; on
that half the same reading gives 92.1% right, 89.9% decides, 96.0% right when deciding
(`eval/vision-v0.1-half-a.txt`), so the numbers above are not an artefact of the fit.

**Which file read what:** every text went through the shipped text file. The 3,486 held-out button pictures went
through the shipped image file; the photos, documents and pages went through the same merged weights in PyTorch at
full precision, which the shipped image file reproduces to a cosine of 0.9996 (lowest 0.9987, on 96 pictures of
every kind): running all 20,000 through the shipped file takes hours on this Mac. On the buttons, where both were run, the two paths' vectors agree to 0.9997 (lowest 0.9978 of 3,486).

## A button the page gave no name (`eval/vision-buttons-v0.1.txt`)

This is the use Atelier ships with (`browser_steps`): a link or button whose page gives it no name is asked two
questions, exactly as measured here. Which of all 63 jobs is it, by name, in the page's language; and then, is it
really a button for that job. Every held-out button in each of five languages:

| Cutoffs (job, yes) | Icon set never seen: named right / wrong / left alone | Job never trained on | None of the 63 jobs (a logo, an anchor): named anyway |
|---|---|---|---|
| 0.75, 0.70 (the readers' usual) | 75.9% / 2.4% / 21.8% | 33.7% / 11.3% / 55.0% | 19.6% |
| 0.90, no second question | 66.3% / 0.9% / 32.7% | 31.7% / 8.1% / 60.2% | 14.0% |
| **0.90, 0.90 (what Atelier uses)** | **64.1% / 0.9% / 35.0%** | 29.0% / 6.3% / 64.7% | **9.9%** |
| 0.95, 0.90 | 57.0% / 0.6% / 42.4% | 28.2% / 4.7% / 67.2% | 7.8% |

Before, every one of these buttons was left out of a step. With Atelier's cutoffs two in three icon buttons drawn a
way it never saw get their right name, and one in a hundred a wrong one. The base, asked the first question, names
5% of them: among 63 options it is almost never sure.

The weak side is a button that is none of the jobs: one in ten is still named as the job it looks most like. A name
is not a click: atelier-step still has to choose it for the task, and the click guard and the approvals check every
click. But it is the number to bring down in the next version.

## Search (`eval/vision-search-v0.1.txt`)

A picture's vector also finds it by a description. On Crossmodal-3600 (3,600 photos, none trained on): a caption
in, and how often its photo is the nearest, among the nearest 5 and 10 of all 3,600; and how often it comes first
among itself and 39 others (a page's worth of pictures, what `label_pictures` does with `find`).

| Captions in | Nearest | In 5 | In 10 | First among 40 | SigLIP 2 as it comes: nearest |
|---|---|---|---|---|---|
| English | 58.2% | 82.1% | 89.0% | 95.7% | 57.7% |
| Dutch | 53.2% | 78.8% | 86.2% | 94.1% | 54.8% |
| German | 69.1% | 89.1% | 93.5% | 97.3% | 70.4% |
| French | 67.9% | 89.3% | 93.9% | 97.6% | 68.4% |
| Spanish | 64.0% | 86.6% | 92.4% | 96.8% | 64.6% |
| Mean | 62.5% | 85.2% | 91.0% | 96.3% | 63.2% |

Training was held to what the base could already do here (the photos' vectors and ordinary captions kept in place),
and shrinking the files and the vocabulary comes on top: together they cost 0.7 points of "nearest".

Languages it was not trained to answer in still search, through the trimmed vocabulary
(`eval/vision-search-v0.1-more.txt`, one caption a photo): Italian 65.6% nearest, Portuguese 58.8%, Indonesian
58.6%, Polish 52.7%, Danish 46.4%, Swedish 44.6%, Turkish 44.0%. Japanese, Russian and Arabic captions are declined,
all of them: the vocabulary keeps Latin script only.

## Use

```bash
hf download 3SENN/atelier-decisions --include "releases/vision-v0.1/model/*" --local-dir .
```

From a clone of the GitHub repository ([3SENN/atelier-decisions](https://github.com/3SENN/atelier-decisions)),
`python tools/hf_release.py fetch vision-v0.1 --only model` does the same and checks every file against
`MANIFEST.json`. Then, in `runtime/node`:

```bash
npm run vision:parity     # the runtime's tower input, tokens, vectors and answers against the export's
npm run start:vision      # POST /v1/look, POST /v1/embed on 127.0.0.1:8011, and /look to try it in a browser
```

```json
{"picture": {"png": "<base64>"},
 "questions": {"kind": {"type": "choice", "instructions": "What is this a picture of?",
                        "criteria": {"receipt": "A till receipt from a shop or a restaurant",
                                     "invoice": "An invoice from a company", "photo": "An ordinary photo"}}}}
```

An option has to say what to look for: for a choice the `instructions` are not read, only the options. With
`keep: true` the picture's vector comes back, and `{vector, questions}` asks again without looking again.
In Atelier it is an optional download (`integration/atelier-vision.patch` in the GitHub repository).

## Data (`data/`)

- `pictures.zip`: every picture as the towers read it. 13,050 Open Images V7 validation photos (256 x 256, each
  with its author, licence and page: CC BY 2.0 by their authors, labels CC BY 4.0 by Google), and the drawn buttons
  (11,814), documents (8,240) and pages (4,240), with the licences of the nine icon sets the buttons are drawn from.
- `questions.zip`: the 42,452 held-out questions every number here is measured on, and the COCO-35L dev captions
  (CC BY 4.0) used as text only to hold the text tower in place.
- Crossmodal-3600 (CC BY 4.0), for search, is fetched by `vision/photos.py xm3600`.

The generated pictures and the questions are CC BY 4.0 like the project's other data. The weights are Apache-2.0,
built from SigLIP 2 (`google/siglip2-base-patch16-256`, Apache-2.0).

## What it can't do

- **A state of a page it never trained on.** A robot check or a paywall, neither ever shown: right 54% of the time
  as a choice, and sure and wrong on 16% (21% as a yes/no). The six states it trained on hold on layouts it never
  saw (88% right, 3% sure and wrong); a new state needs examples, not only a new question.
- **A button that is none of its 63 jobs.** One in ten is still named as the job it looks most like, with Atelier's
  cutoffs. And a job it never trained on is named wrong 6% of the time, right 29%.
- **Yes/no about what is in a photo.** Sure and wrong on 7% (10% for things never trained on): Open Images' people
  confirmed "a dog" but nobody listed everything else in the photo, and a small or half-hidden thing is easy to say
  no to. A choice among things is the safer question (2% sure and wrong).
- **Reading.** It sees that a picture is a receipt; it does not read the total. No text recognition, no faces (who
  is this), no counting, no "the third one": an answer has to say what to look for.
- **Scripts other than Latin.** The vocabulary is trimmed, so Japanese, Russian, Arabic and the like are declined,
  not answered badly. Languages in Latin script it wasn't trained to answer in (Swedish, Danish, Turkish) search at
  about 45% nearest against 58% for English, and were not measured on questions.
- **Levels on a scale** (`score`): declined. No version has been measured on them.
- **Speed and memory.** A picture takes about 0.3 s on two CPU threads of an M2 (0.2 s on four), far from atelier-nano's 10 ms: fine for a dozen buttons or forty thumbnails, not for a library of photos in one go. It is the heaviest of the on-device models to hold:
  260 MB to download and about 0.6 GB of memory while loaded (up to 1 GB while it reads many new texts at once).
- **Drawn data.** Buttons, documents and pages are generated: real icon sets and a real browser engine, but made-up
  sites. Photos and search are real pictures. No number here is from pictures of real websites yet.
- **The shipped image file was only run on the buttons** for these numbers (see "Which file read what").

## Found along the way

- **A decision is a match.** Nothing was added to SigLIP 2 to make it answer System One questions: every answer is a
  text, and three or four fitted numbers turn how well the picture matches each into a calibrated probability. That
  is why a new use needs no new head and no new training, only a new question.
- **Uncalibrated, it is certain about everything.** SigLIP's scale is 113, so a difference of 0.02 in cosine is
  already ten to one. And one temperature does not fit a question with 63 options: sixty wrong ones that each look a
  little possible add up to more than the right one. A choice from 16 options has its own.
- **What a no looks like adds nothing.** Fitted, the weight of a yes/no question's "false" text is 0.00.
- **Trained weights are touchier than the base's.** The base reads through a 4-bit embedding table almost unchanged
  (text cosine 0.988); after the first training run the same table moved texts to 0.96, and to 0.88 on exactly the
  wordings it had been trained on. The second run trains through the table and the tokenizer as shipped, and what
  is left between PyTorch and the shipped text file is 0.9999.
- **int8 the usual way breaks the image tower** (cosine 0.83 to full precision). Weight-only 8 bits keeps 0.9996
  at a quarter of the size; 4 bits does not (0.97).
- **Lowercase.** SigLIP 2 was trained on lowercased text; in a quick check on the base, Dutch captions with
  their capitals found the right photo 43% of the time instead of 57%. The format lowercases, in both runtimes.
- **The base is already good at photos and bad at interfaces.** As it comes it sorts photos into albums 85% right
  and names a document's kind 94-98% right, but tells the state of a page 43% right and is sure of a button's job
  among 63 for one in fourteen. Training went where the gap was, with two terms holding photos and captions where
  the base had them: search lost 0.7 points.

## Next

- Buttons that are none of the jobs: more of them in training and a "none of these" the first question can choose.
- Pages from real sites: the lab's flows with atelier-step, counting the steps an icon button's name made possible.
- More page states, and screenshots of real notices and walls, so a new state is not a guess.
- Levels on a scale, measured (how full, how blurry, how finished), before `score` is answered.
- A lighter image tower: at this size the picture is the cost. A smaller SigLIP 2, or fewer patches, measured the
  same way.
- The shipped image file on every held-out picture, on a machine where that is minutes.
