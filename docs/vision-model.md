# atelier-vision: a decision model that looks

Atelier's decision models read text. A browser is full of things that are not text: a button that is only an icon,
a photo, a scanned receipt, a page with a notice over it. atelier-vision answers the same System One questions about
a picture, on the device, and like the others it writes nothing: when it is not sure, the rule or the companion
decides.

## What it answers

A decision model chooses among the answers it is given. For a picture, every answer is a short text, and the model
says how well the picture matches each one:

| Question | The texts | Answer |
|---|---|---|
| choice | one per option: its description, or the option's own name when it has none | a probability per option |
| noul | what a yes looks like (`criteria.true`, or else the question itself), and what a no looks like when `criteria.false` says | the probability of yes |
| score | one per level | declined: no version has been measured on levels yet |

So a new use is a new question, not a new model. "Which album does this photo belong in", "is this attachment a
receipt", "what does this button do", "is a cookie notice covering the page" are the same model and the same call:

```json
{"picture": {"png": "..."},
 "questions": {"kind": {"type": "choice", "instructions": "What is this a picture of?",
                        "criteria": {"receipt": "A till receipt from a shop or a restaurant", "invoice": "An invoice from a company",
                                     "photo": "An ordinary photo"}}}}
```

Two things follow from how it works (`vision/format.py`, ported line for line to `runtime/node/src/vision/format.ts`):

- **An option has to say what to look for.** For a choice the `instructions` are not read, only the options: "the
  first one" means nothing to it.
- **A picture is looked at once.** Its vector (768 numbers) answers every later question about it and finds it in a
  search: the nearest pictures to a description's vector are the ones it describes.

It declines (`refusal`, which Atelier reads as "the rule decides") what it can't answer honestly: a text it can't
read (its vocabulary is trimmed to Latin-script languages), a text longer than its 63 tokens, an option without a
text, and levels on a scale.

## The model

Two towers that put a picture and a text in the same space, from SigLIP 2 (`google/siglip2-base-patch16-256`,
Apache-2.0): an image tower of 93M parameters that reads 256 patches of 16 x 16 pixels, and a text tower of 86M with
an embedding table. SigLIP trains them so that sigmoid(scale x cos(picture, text) + bias) is the chance that the text
describes the picture. That one number per answer is all a decision needs (`vision/decide.py`):

- a choice is a softmax over its options' numbers, with one temperature;
- a yes/no question is a sigmoid of the "true" text's number (and the "false" text's, when there is one), with
  fitted weights.

Those few numbers are the calibration (`vision/calibrate.py`, in `vision.json`): one set for every kind of picture
on purpose, so a question nobody measured gets the same reading.

| | atelier-vision v0.1 |
|---|---|
| Image tower | 93M parameters, `image.q8.onnx`, 105.8 MB |
| Text tower | 86M parameters and an embedding table cut from 256,000 tokens to 118,947, `text.q8.emb4.onnx`, 147.9 MB; `tokenizer.json`, 6.3 MB |
| Download | 260 MB |
| A picture, M2 | about 0.3 s on 2 CPU threads, 0.2 s on 4 (`analysis/results/vision-bench-v0.1.txt`) |
| A text not read before | 0.1 s; a question whose texts were read before is a dot product |
| Memory | about 0.6 GB loaded, up to 1 GB while many new texts are read at once |

## Data: answers people checked, and answers known by construction

No teacher labels anything.

- **Photos** (`vision/photos.py`, `vision/classes.py`): Open Images V7's validation photos (labels CC BY 4.0, photos
  CC BY 2.0 by their authors). People confirmed some labels on each photo ("a dog") and rejected others ("no cat"),
  so a yes/no question has both answers and a choice has hard wrong options. 13,050 photos of 203 things a person
  would sort photos by, grouped into 13 albums, named in five languages.
- **Buttons** (`vision/icons.py`, `vision/synth.py`): 63 jobs a button can have (cart, search, close, ...), each
  drawn by nine open icon sets, as buttons in the sizes, shapes, colours and themes sites use, at 1x, 2x and 3x.
  The picture is the button's own box, as Atelier cuts it out of a page. And icons that are none of the jobs (a
  logo, an anchor; 1,440 pictures to train on, 540 held out from the icon sets never seen): in training only ever
  a "no".
- **Documents** (`vision/synth.py`): 22 kinds of paper and screen (a receipt, an invoice, a ticket, a letter, an
  identity card marked as a specimen, a menu, a calendar, a spreadsheet, a chat, a handwritten note, ...) with
  made-up content in five languages, as a flat scan or screenshot, or as a photo taken at an angle on a table.
- **Pages** (`vision/synth.py`): whole pages in one of 8 states (ready, a cookie notice, a pop-up, a sign-in wall,
  an error, still loading, a robot check, a paywall) on six layouts of site, at desktop and phone sizes, with real
  photos where a site has pictures.

Everything drawn is drawn by a real browser engine (`vision/render.mjs`, any Chromium over the DevTools protocol),
the one Atelier shows pages with, so a generated button looks as it would when Atelier takes its picture.

A question is worded the ways a caller would word it (`vision/questions.py`): options as bare names, with their
article, as sentences, or as a job's description; a yes/no with what yes and no look like, with only the yes, or as
the bare question; in English, Dutch, German, French or Spanish, whatever the picture's own language.

**Held out, never trained on:**

| Split | What |
|---|---|
| items | trained things on pictures never seen: 30% of the photos, a tenth of the drawn pictures |
| look | trained things drawn a way never seen: three whole icon sets, two page layouts, and for every kind of document and page state one language its text was never in |
| kinds | things never trained on, as an answer or as a wrong option: 37 photo classes, 8 button jobs, 5 kinds of document, 2 page states |
| none | buttons from the held-out icon sets that are none of the 63 jobs |
| wording | the last wording of every list of templates is only ever used in the held-out questions |

Half of the held-out pictures (by their id) are what the calibration is fitted on; every number below is from the
other half.

**Search** is measured on Crossmodal-3600 (CC BY 4.0): 3,600 photos with captions in 36 languages, never trained on.

## Training (on this Mac)

SigLIP 2 already does much of this. What training adds, and what it must not cost:

- **Added** (LoRA of rank 16 on the linear layers of each tower's last six blocks, 2.65M parameters):
  telling apart what SigLIP was never shown much of (icon buttons, the state a page is in), and reading an answer
  however it is worded, questions and all.
- **Kept**: the base finds a photo by a caption in dozens of languages. Two terms hold that in place while it
  trains: every photo's vector stays where the base put it, and the text tower keeps reading ordinary captions as
  the base did (COCO-35L's dev captions in 21 languages, as text only).

```bash
python3.12 -m venv .venv && .venv/bin/pip install torch "transformers==5.17.*" tokenizers onnx onnxruntime pillow
hf download google/siglip2-base-patch16-256 --revision 3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab --local-dir models/siglip2-base-patch16-256
python3 -I vision/photos.py fetch && python3 -I vision/photos.py xm3600 && python3 -I vision/photos.py coco35   # about 6 minutes, 350 MB
python3 -I vision/icons.py fetch && python3 -I vision/icons.py others
.venv/bin/python -I vision/synth.py render                       # draws 24,000 pictures with a Chromium: about 40 minutes
.venv/bin/python -I vision/data.py eval                           # the held-out questions, fixed
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.42 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.34 PYTORCH_ENABLE_MPS_FALLBACK=1
.venv/bin/python -I vision/embed.py --packs photos --out runs/vision/emb/base      # where the base puts every photo
.venv/bin/python -I vision/train.py --out runs/vision-v1 --epochs 1
.venv/bin/python -I vision/vocab.py build --out runs/vision/vocab-12 --langs en nl de fr es it pt sv da pl id tr
.venv/bin/python -I vision/train.py --init runs/vision-v1 --out runs/vision-v2 --epochs 1 --lr 5e-5 --buttons --vocab runs/vision/vocab-12 --batch 14
.venv/bin/python -I vision/export.py --adapters runs/vision-v2/vision.pt --vocab runs/vision/vocab-12 --out runs/vision-v2/onnx
.venv/bin/python -I vision/embed.py --packs controls --runtime runs/vision-v2/onnx --only-held-out --out runs/vision/emb/v2      # the shipped image file: add photos documents pages xm3600 where there is time
.venv/bin/python -I vision/embed.py --packs photos documents pages xm3600 --adapters runs/vision-v2/vision.pt --only-held-out --out runs/vision/emb/v2
.venv/bin/python -I vision/calibrate.py --emb runs/vision/emb/v2 --runtime runs/vision-v2/onnx --out runs/vision-v2/onnx/vision.json
.venv/bin/python -I vision/measure.py --emb runs/vision/emb/v2 --runtime runs/vision-v2/onnx --name vision-v0.1
.venv/bin/python -I vision/buttons.py --emb runs/vision/emb/v2 --runtime runs/vision-v2/onnx --name vision-buttons-v0.1
.venv/bin/python -I vision/search.py --emb runs/vision/emb/v2 --runtime runs/vision-v2/onnx --langs en nl de fr es --name vision-search-v0.1
.venv/bin/python -I vision/export.py --adapters runs/vision-v2/vision.pt --vocab runs/vision/vocab-12 --calibration runs/vision-v2/onnx/vision.json --out runs/vision-v2/onnx   # the same files, with calibrated parity fixtures
.venv/bin/python -I tools/make_vision_release.py --version vision-v0.1 --run runs/vision-v2 --results vision-v0.1.json ...
```

Two runs, both one pass over the training pictures on an M2 with 24 GB (`releases/vision-v0.1/eval/vision-v1-train.txt`,
`vision-v2-train.txt`):

1. **v1**, 53 minutes: 1,404 updates of 16 pictures. On a monitor of 2,400 held-out questions it went from 74% right
   to 93%.
2. **v2**, 74 minutes, from v1 at half the learning rate, with three things v1 showed were needed: the embedding
   table and tokenizer as they ship (below), the buttons that are none of the jobs, and the question Atelier really
   asks (every job at once). This is the saved version.

It trains in half precision on the Mac's GPU. Memory is the limit, not time: a batch is 14-16 pictures and at most
112 texts, and the run stops itself when the disk or the swap file runs out of room.

## Results

Every held-out question, read the way Atelier reads answers (a choice at 0.75, a yes at 0.7, a no at 0.3), through
the shipped files (`analysis/results/vision-v0.1.json`). "Decides" is how often it is that sure; the base is SigLIP 2
as it comes, calibrated the same way.

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

**Calibration** (`vision/calibrate.py`, in `vision.json`): a temperature for a choice (2.54; 1.99 from 16
options), and a weight and an offset for a yes/no (0.32 and 2.93 with only a "true" text; 0.34, 0.00 and 3.13 with a
"false" text too: what a no looks like turns out to add nothing). Fitted on one half of the held-out pictures; on
that half the same reading gives 92.1% right, 89.9% decides, 96.0% right when deciding
(`analysis/results/vision-v0.1-half-a.txt`), so the numbers above are not an artefact of the fit.

**Which file read what:** every text went through the shipped text file. The 3,486 held-out button pictures went
through the shipped image file; the photos, documents and pages went through the same merged weights in PyTorch at
full precision, which the shipped image file reproduces to a cosine of 0.9996 (lowest 0.9987, on 96 pictures of
every kind): running all 20,000 through the shipped file takes hours on this Mac. On the buttons, where both were run, the two paths' vectors agree to 0.9997 (lowest 0.9978 of 3,486).

### A button the page gave no name (`analysis/results/vision-buttons-v0.1.txt`)

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

### On live websites, a first look (`analysis/results/vision-real-sites-v0.1.txt`)

After the version was saved: icon-only links and buttons from live pages (`vision/real.mjs`, a headless Chromium;
`vision/real.py` asks the two questions). Twelve sites were tried and seven gave buttons (the others answer a
headless browser with a wall or have no icon-only buttons in view): GitHub, Wikipedia, NOS, YouTube, Stack Overflow,
IKEA and Booking.com, 21 buttons, each with the name the page's own code gives it to read the answer against.

- **10 named, 9 of them right**: YouTube's menu, search, microphone and three dots, GitHub's three dots, Stack
  Overflow's help and copy, IKEA's chat, Booking.com's close.
- **1 named wrong**: a user's generated avatar (a pattern of squares) as "grid view". It is none of the 63 jobs,
  the known weak side.
- **11 left alone**: GitHub's logo, a padlock, three users' avatars, previous and next arrows (the right job on
  top, at 0.45-0.66 of the weight), a filter icon.

Twenty-one buttons are a look, not a number: it says the drawn buttons were not a world of their own.

### How often a page needs it (`analysis/results/vision-sites-v0.1.txt`)

The question behind the button numbers: how many buttons does a real page leave without a name? `vision/sites.mjs`
opens the first screen of 60 well-known sites (`vision/sites.txt`) and splits every link and button the way
Atelier's page bridge does. 39 loaded; the others answer a headless browser with a wall.

| | |
|---|---|
| Links and buttons the page's own code names | 6,305 |
| With no name anywhere | 175 (2.7%) |
| Of those, whole in view and the size of a control: what Atelier asks the model about | 9, on 7 of the 39 pages |
| Real buttons among the 9 (two searches, a settings gear, an account button) | 4, all named right |
| Site logos among the 9 | 5, all left alone |
| A picture, as the helper runs it | 0.3 s |

So websites name nearly all their buttons themselves, as accessibility asks of them. For browser tasks the model
adds a button on about one first screen in ten, got those right, and named nothing wrongly here. That is a small
gain, and it is the honest size of this use: nobody should download 260 MB for it alone. What the download is for
is pictures themselves, which no page's code describes: sorting and finding photos, telling a receipt from an
invoice (`label_pictures`, the results above).

### Search (`analysis/results/vision-search-v0.1.txt`)

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
(`analysis/results/vision-search-v0.1-more.txt`, one caption a photo): Italian 65.6% nearest, Portuguese 58.8%, Indonesian
58.6%, Polish 52.7%, Danish 46.4%, Swedish 44.6%, Turkish 44.0%. Japanese, Russian and Arabic captions are declined,
all of them: the vocabulary keeps Latin script only.

## Shrinking it

What ships is 260 MB instead of the base's 1.5 GB, by three cuts, each measured (`vision/export.py` writes the
numbers into `vision.json`):

- **Layers at 8 bits, weights only.** ONNX Runtime's usual int8 (dynamic quantization) breaks the image tower: its
  vectors fall to a cosine of 0.83 against full precision. Storing each weight matrix at 8 bits and computing in
  floats keeps 0.9996 (lowest 0.9987) at a quarter of the size. 4 bits does not (0.97).
- **A vocabulary of 118,947 tokens instead of 256,000** (`vision/vocab.py`, the way atelier-nano's is trimmed): the
  tokens that captions and everyday sentences in twelve Latin-script languages use, closed under the tokenizer's
  merges, each word in both its positions. Text in other scripts would fall apart into bytes, so the format declines
  it (under 70% Latin letters, or over 10% raw bytes).
- **The embedding table at 4 bits.** The base reads through it almost unchanged, but after training the same table
  moved texts to a cosine of 0.96, and 0.88 on exactly the wordings training had sharpened. So v2 trains through the
  table and tokenizer as shipped: what is left between PyTorch and the shipped text file is 0.9999 (lowest 0.9995).

Search is where shrinking would show: through the shipped files the caption finds its photo first 62.5% of the
time against the untouched base's 63.2% (`analysis/results/vision-search-v0.1.txt`, `vision-search-base.txt`).

**Parity.** `vision/export.py` writes `parity.jsonl`: pictures with the tower input's checksum and vector, texts
with their token ids and vectors, questions with their answers. `runtime/node/src/vision/parity.ts` must reproduce
all of it from the same bytes, and does: 39 of 39 texts the same ids, vectors to 0.999999, 12 of 12 answers.

**How a picture arrives** (`vision/robust.py`, `analysis/results/vision-robust-v0.1.txt`): made smaller first, as
Atelier does above 768 pixels, changes 0.6% of a photo's readings and none of a button's; another resize filter
1-2.5%; a rough JPEG 1% for photos, 5% for buttons.

## Running it

Saved as `releases/vision-v0.1` ([MODEL_CARD.md](../releases/vision-v0.1/MODEL_CARD.md)). Its weights and pictures
are on Hugging Face once the version is published there (`python tools/hf_release.py publish vision-v0.1`).

```bash
python tools/hf_release.py fetch vision-v0.1 --only model   # the model, checked against MANIFEST.json
cd runtime/node
npm run vision:parity     # the runtime's tower input, tokens, vectors and answers against the export's
npm run start:vision      # the decision server with atelier-vision: POST /v1/look, POST /v1/embed, and /look to try it
```

`POST /v1/look` takes `{picture, questions}` and answers each question; with `keep: true` it gives the picture's
vector back, and `{vector, questions}` asks again without looking again. `POST /v1/embed` takes `{texts, picture}`
and gives vectors to search by. A picture is `{png}` or `{pixels, width, height, channels}`, base64. The page at
`/look` is for trying it: drop pictures, write the answers a question may have, see what it picks and how sure it
is, and find a picture by describing it.

**In Atelier** (`integration/atelier-vision.patch`, docs/decisions.md › Pictures there): an optional download in
Settings › Quick decisions › On this computer, run in the same helper as the other models. Two uses come with it:

- **Browser tasks: buttons with only an icon.** A page's code often gives such a button no name, so atelier-step
  could never choose it: the page bridge left it out. Now up to twelve a page come along with their box, the browser
  takes each one's picture, and `browser_steps` asks two questions about it: which of the 63 jobs is this, by name,
  in the page's language; and then whether it really is a button for that job. It is named only when both answers
  are sure (one job with 0.90 of the weight, and a yes at 0.90: two in three icon buttons drawn a way it never saw get their right name, one in a hundred a wrong one); otherwise it stays out of the step, as before. The second question is there for
  buttons that are none of the jobs: among 63 answers something always looks likeliest.
- **The companion's `label_pictures`**: `label_items` for pictures. Up to 40 pictures on a page, each named by the
  ref of its element, and one to five questions in the System One shape, read with the same cutoffs; "unsure" comes
  back with what the model leaned towards. With `find`, a description, every picture also gets how likely the
  description fits it, and the pictures come back best first: search among a page's pictures.

Both run in the helper that already runs the other models, and nothing leaves the computer. Elsewhere in Atelier
`look(picture or vector, questions)` and `embed(texts, picture)` are there for the next use: a new one is a
question, not a model.

## What it can't do yet

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
  sites. Photos and search are real pictures. From real websites there are the two looks above: 21 icon buttons, and the 9 that 39 first screens left unnamed.
- **The shipped image file was only run on the buttons** for these numbers (see "Which file read what").
