# Capabilities: mini-0.8b-40k

`runs\mini-l24` (mini-l24-Q8_0.decider.gguf, 811.8 MB), measured 2026-11-22 06:34.

## Atelier's own eval

| Decision | Rules alone | With the model | Right when sure | Unsure |
|---|---|---|---|---|
| needsYou | 5/10 | 9/10 | 9/9 | 1 |
| dictation | 4/8 | 8/8 | 4/4 | 4 |
| agentClicks | 5/12 | 11/12 | 7/7 | 5 |
| searchOrAsk | 5/10 | 10/10 | 10/10 | 0 |
| voiceTarget | 3/6 | 6/6 | 3/3 | 3 |
| effort | 3/12 | 8/12 | 7/7 | 5 |
| memory | 5/10 | 10/10 | 5/5 | 5 |
| tabs | 0/12 | 10/12 | 10/10 | 2 |
| skills | 2/10 | 10/10 | 10/10 | 0 |

Tests: 10 passed, 0 failed.

## Held out, per decision

| Decision | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| agentClicks | 516 | 91.3% | 54.5% | 86.5% | 0.4% | 7.0% | 0.0% |
| dictation | 167 | 83.8% | 71.9% | 92.5% | 0.0% | 5.4% | 0.0% |
| effort | 739 | 71.7% | 34.5% | 81.2% | 0.1% | 6.4% | 0.0% |
| labeling | 34 | 79.4% | 38.2% | 100.0% | 0.0% | 0.0% | 0.0% |
| memory | 591 | 83.4% | 35.4% | 98.1% | 0.0% | 0.7% | 0.0% |
| needsYou | 568 | 93.3% | 95.2% | 96.3% | 0.2% | 3.4% | 0.0% |
| searchOrAsk | 664 | 79.4% | 64.3% | 93.9% | 0.0% | 3.9% | 0.0% |
| skills | 2215 | 95.5% | 98.4% | 96.8% | 1.4% | 1.7% | 0.0% |
| tabs | 425 | 88.5% | 72.2% | 93.8% | 0.0% | 4.5% | 0.0% |
| voiceTarget | 384 | 69.5% | 28.9% | 93.7% | 0.0% | 1.8% | 0.0% |
| ALL | 6303 | 86.9% | 70.5% | 94.6% | 0.6% | 3.2% | 0.0% |

## Held out, per language

| Language | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| de | 407 | 83.0% | 67.8% | 93.1% | 1.5% | 3.2% | 0.0% |
| en | 3860 | 88.3% | 72.4% | 95.5% | 0.4% | 2.9% | 0.0% |
| es | 503 | 87.1% | 71.8% | 92.8% | 1.0% | 4.2% | 0.0% |
| fr | 475 | 84.2% | 65.5% | 92.6% | 0.4% | 4.4% | 0.0% |
| nl | 1058 | 84.3% | 66.1% | 93.3% | 0.8% | 3.7% | 0.0% |
| ALL | 6303 | 86.9% | 70.5% | 94.6% | 0.6% | 3.2% | 0.0% |

## Held out, open questions per source

| Source | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| gen2-pages | 35 | 54.3% | 60.0% | 76.2% | 8.6% | 5.7% | 0.0% |
| gen2-routines | 27 | 44.4% | 48.1% | 84.6% | 3.7% | 3.7% | 0.0% |
| gen2-tools | 23 | 56.5% | 39.1% | 100.0% | 0.0% | 0.0% | 0.0% |
| gliclass | 408 | 74.3% | 72.3% | 88.1% | 2.7% | 5.9% | 0.0% |
| zeroshot | 284 | 77.1% | 68.7% | 92.8% | 0.7% | 4.2% | 0.0% |
| ALL | 777 | 72.8% | 68.6% | 89.5% | 2.2% | 5.0% | 0.0% |

## Probe: 20 languages (never seen)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| ar | 150 | 74.7% | 23.3% | 77.1% | 0.0% | 5.3% | 0.0% |
| de | 150 | 73.3% | 31.3% | 83.0% | 0.0% | 5.3% | 0.0% |
| el | 150 | 59.3% | 19.3% | 65.5% | 0.0% | 6.7% | 0.0% |
| en | 150 | 68.0% | 36.0% | 66.7% | 0.0% | 12.0% | 0.0% |
| es | 150 | 70.7% | 29.3% | 75.0% | 0.0% | 7.3% | 0.0% |
| fr | 150 | 73.3% | 30.0% | 73.3% | 0.0% | 8.0% | 0.0% |
| he | 150 | 63.3% | 20.0% | 63.3% | 1.3% | 6.0% | 0.0% |
| hi | 150 | 55.3% | 3.3% | 80.0% | 0.0% | 0.7% | 0.0% |
| id | 150 | 66.7% | 34.0% | 56.9% | 0.0% | 14.7% | 0.0% |
| it | 150 | 68.0% | 31.3% | 68.1% | 0.0% | 10.0% | 0.0% |
| ja | 150 | 76.7% | 20.0% | 86.7% | 0.0% | 2.7% | 0.0% |
| ko | 150 | 61.3% | 14.0% | 81.0% | 0.0% | 2.7% | 0.0% |
| nl | 150 | 66.0% | 40.7% | 80.3% | 0.0% | 8.0% | 0.0% |
| pl | 150 | 70.0% | 34.7% | 67.3% | 0.0% | 11.3% | 0.0% |
| pt | 150 | 73.3% | 26.7% | 72.5% | 0.0% | 7.3% | 0.0% |
| ru | 150 | 75.3% | 36.0% | 79.6% | 0.0% | 7.3% | 0.0% |
| sv | 150 | 66.7% | 18.7% | 60.7% | 0.0% | 7.3% | 0.0% |
| th | 150 | 70.0% | 12.0% | 44.4% | 0.0% | 6.7% | 0.0% |
| tr | 150 | 62.7% | 22.7% | 88.2% | 0.0% | 2.7% | 0.0% |
| zh | 150 | 72.0% | 28.7% | 83.7% | 0.0% | 4.7% | 0.0% |
| ALL | 3000 | 68.3% | 25.6% | 73.0% | 0.1% | 6.8% | 0.0% |

## Probe: new kinds of questions about new text

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| probe-areas | 200 | 79.0% | 77.0% | 97.4% | 0.0% | 2.0% | 0.0% |
| probe-family | 100 | 93.0% | 93.0% | 98.9% | 0.0% | 1.0% | 0.0% |
| probe-money | 47 | 74.5% | 80.8% | 89.5% | 8.5% | 0.0% | 0.0% |
| probe-purchase | 53 | 96.2% | 100.0% | 96.2% | 1.9% | 1.9% | 0.0% |
| probe-urgency | 50 | 82.0% | 94.0% | 85.1% | 2.0% | 12.0% | 0.0% |
| ALL | 450 | 84.0% | 85.6% | 95.3% | 1.3% | 2.7% | 0.0% |

## Probe: held-out browser question families (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| chat-sincere-or-sarcastic | 10 | 10.0% | 30.0% | 33.3% | 20.0% | 0.0% | 0.0% |
| dark-pattern-type | 10 | 30.0% | 30.0% | 66.7% | 10.0% | 0.0% | 0.0% |
| destructive-action | 10 | 70.0% | 90.0% | 77.8% | 10.0% | 10.0% | 0.0% |
| free-returns | 10 | 50.0% | 80.0% | 62.5% | 20.0% | 10.0% | 0.0% |
| pre-ticked-optional-box | 10 | 50.0% | 70.0% | 71.4% | 20.0% | 0.0% | 0.0% |
| price-currency-as-expected | 10 | 40.0% | 70.0% | 57.1% | 30.0% | 0.0% | 0.0% |
| summary-faithful | 10 | 40.0% | 60.0% | 66.7% | 20.0% | 0.0% | 0.0% |
| ticket-availability | 10 | 40.0% | 60.0% | 33.3% | 40.0% | 0.0% | 0.0% |
| tool-output-prompt-injection | 10 | 10.0% | 20.0% | 50.0% | 10.0% | 0.0% | 0.0% |
| total-cost-within-budget | 10 | 40.0% | 90.0% | 44.4% | 20.0% | 30.0% | 0.0% |
| ALL | 480 | 47.7% | 46.2% | 83.3% | 5.2% | 2.5% | 0.0% |

## Probe: held-out browser question families, second set (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| agent-proceed-or-ask | 8 | 37.5% | 12.5% | 0.0% | 12.5% | 0.0% | 0.0% |
| applicable-tariff | 8 | 12.5% | 12.5% | 0.0% | 12.5% | 0.0% | 0.0% |
| cancel-window-open | 8 | 37.5% | 25.0% | 50.0% | 12.5% | 0.0% | 0.0% |
| jobseeker-obligation-met | 8 | 50.0% | 100.0% | 50.0% | 37.5% | 12.5% | 0.0% |
| maintenance-visit-due | 8 | 25.0% | 12.5% | 0.0% | 12.5% | 0.0% | 0.0% |
| open-reveals-seen-status | 8 | 62.5% | 75.0% | 83.3% | 12.5% | 0.0% | 0.0% |
| outage-affects-address | 8 | 37.5% | 100.0% | 37.5% | 37.5% | 25.0% | 0.0% |
| payment-method-matches-request | 8 | 37.5% | 62.5% | 60.0% | 25.0% | 0.0% | 0.0% |
| sick-note-required | 8 | 50.0% | 87.5% | 57.1% | 25.0% | 12.5% | 0.0% |
| vignette-valid-for-trip | 8 | 87.5% | 87.5% | 85.7% | 12.5% | 0.0% | 0.0% |
| ALL | 576 | 43.8% | 43.4% | 86.0% | 3.5% | 2.6% | 0.0% |

## Probe: 18-28 tab groups

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| probe-many-groups | 150 | 52.7% | 14.7% | 81.8% | 0.0% | 2.7% | 0.0% |
| ALL | 150 | 52.7% | 14.7% | 81.8% | 0.0% | 2.7% | 0.0% |
