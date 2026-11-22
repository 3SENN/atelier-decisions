# Capabilities: mini-0.8b-150k

`runs\mini-l24-150k` (mini-l24-150k-Q5_K_M-e4.decider.gguf, 515.2 MB), measured 2026-11-22 10:52.

## Atelier's own eval

| Decision | Rules alone | With the model | Right when sure | Unsure |
|---|---|---|---|---|
| needsYou | 5/10 | 10/10 | 10/10 | 0 |
| dictation | 4/8 | 6/8 | 2/2 | 6 |
| agentClicks | 5/12 | 12/12 | 8/8 | 4 |
| searchOrAsk | 5/10 | 10/10 | 10/10 | 0 |
| voiceTarget | 3/6 | 6/6 | 3/3 | 3 |
| effort | 3/12 | 11/12 | 10/10 | 2 |
| memory | 5/10 | 10/10 | 5/5 | 5 |
| tabs | 0/12 | 10/12 | 10/10 | 2 |
| skills | 2/10 | 9/10 | 9/9 | 1 |

Tests: 10 passed, 0 failed.

## Held out, per decision

| Decision | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| agentClicks | 516 | 93.8% | 50.4% | 92.3% | 0.0% | 3.9% | 0.0% |
| dictation | 167 | 77.2% | 59.3% | 96.0% | 0.0% | 2.4% | 0.0% |
| effort | 739 | 77.0% | 36.5% | 86.7% | 0.1% | 4.7% | 0.0% |
| labeling | 34 | 91.2% | 50.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| memory | 591 | 88.2% | 40.8% | 97.5% | 0.0% | 1.0% | 0.0% |
| needsYou | 568 | 93.1% | 93.3% | 97.2% | 0.2% | 2.5% | 0.0% |
| searchOrAsk | 664 | 82.4% | 66.4% | 94.6% | 0.3% | 3.3% | 0.0% |
| skills | 2215 | 95.3% | 95.8% | 98.4% | 0.7% | 0.9% | 0.0% |
| tabs | 425 | 95.1% | 75.1% | 96.5% | 0.0% | 2.6% | 0.0% |
| voiceTarget | 384 | 77.3% | 36.7% | 95.0% | 0.0% | 1.8% | 0.0% |
| ALL | 6303 | 89.2% | 70.4% | 96.4% | 0.3% | 2.2% | 0.0% |

## Held out, per language

| Language | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| de | 407 | 86.5% | 69.3% | 95.0% | 1.0% | 2.5% | 0.0% |
| en | 3860 | 90.1% | 72.3% | 96.9% | 0.3% | 2.0% | 0.0% |
| es | 503 | 88.9% | 69.0% | 96.2% | 0.4% | 2.2% | 0.0% |
| fr | 475 | 88.2% | 65.9% | 95.5% | 0.2% | 2.7% | 0.0% |
| nl | 1058 | 87.6% | 66.7% | 95.9% | 0.0% | 2.7% | 0.0% |
| ALL | 6303 | 89.2% | 70.4% | 96.4% | 0.3% | 2.2% | 0.0% |

## Held out, open questions per source

| Source | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| gen2-pages | 35 | 51.4% | 45.7% | 87.5% | 2.9% | 2.9% | 0.0% |
| gen2-routines | 27 | 33.3% | 51.8% | 64.3% | 11.1% | 7.4% | 0.0% |
| gen2-tools | 23 | 60.9% | 52.2% | 83.3% | 8.7% | 0.0% | 0.0% |
| gliclass | 408 | 74.3% | 69.1% | 90.1% | 2.5% | 4.4% | 0.0% |
| zeroshot | 284 | 76.4% | 63.7% | 96.1% | 0.0% | 2.5% | 0.0% |
| ALL | 777 | 72.2% | 65.0% | 91.3% | 2.1% | 3.6% | 0.0% |

## Probe: 20 languages (never seen)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| ar | 150 | 71.3% | 17.3% | 76.9% | 0.0% | 4.0% | 0.0% |
| de | 150 | 76.0% | 27.3% | 92.7% | 0.0% | 2.0% | 0.0% |
| el | 150 | 61.3% | 17.3% | 73.1% | 0.0% | 4.7% | 0.0% |
| en | 150 | 72.0% | 20.0% | 90.0% | 0.0% | 2.0% | 0.0% |
| es | 150 | 76.7% | 20.7% | 100.0% | 0.0% | 0.0% | 0.0% |
| fr | 150 | 76.0% | 22.0% | 87.9% | 0.0% | 2.7% | 0.0% |
| he | 150 | 59.3% | 12.7% | 57.9% | 0.7% | 4.7% | 0.0% |
| hi | 150 | 56.0% | 13.3% | 55.0% | 1.3% | 4.7% | 0.0% |
| id | 150 | 75.3% | 18.7% | 85.7% | 0.0% | 2.7% | 0.0% |
| it | 150 | 66.0% | 14.7% | 81.8% | 0.0% | 2.7% | 0.0% |
| ja | 150 | 70.0% | 10.7% | 87.5% | 0.0% | 1.3% | 0.0% |
| ko | 150 | 59.3% | 13.3% | 70.0% | 1.3% | 2.7% | 0.0% |
| nl | 150 | 68.7% | 30.0% | 95.6% | 0.0% | 1.3% | 0.0% |
| pl | 150 | 72.0% | 20.7% | 83.9% | 0.0% | 3.3% | 0.0% |
| pt | 150 | 74.7% | 17.3% | 88.5% | 0.0% | 2.0% | 0.0% |
| ru | 150 | 76.7% | 24.0% | 97.2% | 0.0% | 0.7% | 0.0% |
| sv | 150 | 72.7% | 16.7% | 80.0% | 0.0% | 3.3% | 0.0% |
| th | 150 | 70.7% | 8.7% | 46.2% | 0.0% | 4.7% | 0.0% |
| tr | 150 | 51.3% | 10.7% | 75.0% | 0.7% | 2.0% | 0.0% |
| zh | 150 | 70.0% | 17.3% | 100.0% | 0.0% | 0.0% | 0.0% |
| ALL | 3000 | 68.8% | 17.7% | 84.3% | 0.2% | 2.6% | 0.0% |

## Probe: new kinds of questions about new text

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| probe-areas | 200 | 72.5% | 67.5% | 99.3% | 0.0% | 0.5% | 0.0% |
| probe-family | 100 | 88.0% | 86.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| probe-money | 47 | 74.5% | 87.2% | 85.4% | 10.6% | 2.1% | 0.0% |
| probe-purchase | 53 | 96.2% | 100.0% | 96.2% | 1.9% | 1.9% | 0.0% |
| probe-urgency | 50 | 86.0% | 84.0% | 92.9% | 0.0% | 6.0% | 0.0% |
| ALL | 450 | 80.4% | 79.3% | 96.6% | 1.3% | 1.3% | 0.0% |

## Probe: held-out browser question families (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| chat-sincere-or-sarcastic | 10 | 30.0% | 60.0% | 50.0% | 30.0% | 0.0% | 0.0% |
| contract-type | 10 | 50.0% | 80.0% | 62.5% | 20.0% | 10.0% | 0.0% |
| destructive-action | 10 | 60.0% | 80.0% | 75.0% | 10.0% | 10.0% | 0.0% |
| energy-rating-meets-minimum | 10 | 30.0% | 50.0% | 60.0% | 20.0% | 0.0% | 0.0% |
| message-contains-credentials | 10 | 40.0% | 80.0% | 50.0% | 40.0% | 0.0% | 0.0% |
| summary-faithful | 10 | 50.0% | 80.0% | 62.5% | 30.0% | 0.0% | 0.0% |
| support-ticket-escalate | 10 | 50.0% | 70.0% | 71.4% | 20.0% | 0.0% | 0.0% |
| ticket-availability | 10 | 20.0% | 40.0% | 25.0% | 20.0% | 10.0% | 0.0% |
| tool-output-prompt-injection | 10 | 50.0% | 70.0% | 71.4% | 20.0% | 0.0% | 0.0% |
| total-cost-within-budget | 10 | 20.0% | 30.0% | 0.0% | 20.0% | 10.0% | 0.0% |
| ALL | 480 | 44.2% | 43.3% | 80.3% | 6.2% | 2.3% | 0.0% |

## Probe: held-out browser question families, second set (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| cancel-window-open | 8 | 12.5% | 25.0% | 0.0% | 12.5% | 12.5% | 0.0% |
| click-accepts-legal-terms | 8 | 62.5% | 75.0% | 83.3% | 12.5% | 0.0% | 0.0% |
| deductible-remaining | 8 | 12.5% | 25.0% | 50.0% | 12.5% | 0.0% | 0.0% |
| mortgage-rate-change-direction | 8 | 62.5% | 62.5% | 80.0% | 12.5% | 0.0% | 0.0% |
| moving-checklist-item-done | 8 | 75.0% | 75.0% | 83.3% | 12.5% | 0.0% | 0.0% |
| open-reveals-seen-status | 8 | 50.0% | 62.5% | 80.0% | 12.5% | 0.0% | 0.0% |
| outage-affects-address | 8 | 37.5% | 62.5% | 40.0% | 25.0% | 12.5% | 0.0% |
| regenerate-invalidates-old | 8 | 87.5% | 100.0% | 87.5% | 12.5% | 0.0% | 0.0% |
| sick-note-required | 8 | 25.0% | 50.0% | 50.0% | 12.5% | 12.5% | 0.0% |
| sql-statement-effect | 8 | 50.0% | 75.0% | 66.7% | 25.0% | 0.0% | 0.0% |
| ALL | 576 | 39.6% | 38.5% | 84.7% | 3.8% | 2.1% | 0.0% |

## Probe: 18-28 tab groups

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| probe-many-groups | 150 | 50.0% | 21.3% | 65.6% | 0.0% | 7.3% | 0.0% |
| ALL | 150 | 50.0% | 21.3% | 65.6% | 0.0% | 7.3% | 0.0% |
