## What changes

<!-- What changes, and why. Link the issue it fixes. -->

## How it was measured

<!-- For a model or data change: which reports you ran (capabilities, cascade, numeric, Atelier's eval), with
     decides / right when deciding / sure and wrong next to the version before, and the files they are in.
     For code that doesn't touch a model: how you tested it. -->

## Checklist

- [ ] `python tools/check_headers.py`, `python tools/hf_release.py verify --all --present` and `npm run typecheck` (in `runtime/node`) pass
- [ ] Nothing trains on Atelier's eval cases or on held-out families
- [ ] Every new number in the README or a model card names the results file it comes from
- [ ] New data names its source and licence in its builder and in NOTICE
- [ ] New source files carry the MPL-2.0 notice
