# Capabilities: mini-0.8b-num

`runs\mini-l24-num` (mini-l24-num-Q5_K_M-e4.decider.gguf, 515.2 MB), measured 2026-11-22 12:05.

## Atelier's own eval

| Decision | Rules alone | With the model | Right when sure | Unsure |
|---|---|---|---|---|
| needsYou | 5/10 | 10/10 | 9/9 | 1 |
| dictation | 4/8 | 6/8 | 2/2 | 6 |
| agentClicks | 5/12 | 12/12 | 8/8 | 4 |
| searchOrAsk | 5/10 | 10/10 | 10/10 | 0 |
| voiceTarget | 3/6 | 6/6 | 3/3 | 3 |
| effort | 3/12 | 9/12 | 8/8 | 4 |
| memory | 5/10 | 10/10 | 5/5 | 5 |
| tabs | 0/12 | 10/12 | 10/10 | 2 |
| skills | 2/10 | 9/10 | 9/10 | 0 |

Tests: 10 passed, 0 failed.

## Held out, per decision

| Decision | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| agentClicks | 516 | 94.6% | 52.7% | 91.2% | 0.0% | 4.7% | 0.0% |
| dictation | 167 | 83.2% | 64.1% | 97.2% | 0.0% | 1.8% | 0.0% |
| effort | 739 | 75.8% | 31.8% | 90.2% | 0.1% | 3.0% | 0.0% |
| labeling | 34 | 91.2% | 50.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| memory | 591 | 87.0% | 39.9% | 97.0% | 0.0% | 1.2% | 0.0% |
| needsYou | 568 | 93.8% | 93.8% | 97.2% | 0.4% | 2.3% | 0.0% |
| searchOrAsk | 664 | 83.3% | 69.1% | 93.5% | 0.3% | 4.2% | 0.0% |
| skills | 2215 | 95.9% | 97.2% | 98.0% | 0.6% | 1.3% | 0.0% |
| tabs | 425 | 93.4% | 74.8% | 95.6% | 0.0% | 3.3% | 0.0% |
| voiceTarget | 384 | 76.0% | 34.9% | 95.5% | 0.0% | 1.6% | 0.0% |
| ALL | 6303 | 89.3% | 70.8% | 96.3% | 0.3% | 2.3% | 0.0% |

## Held out, per language

| Language | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| de | 407 | 86.2% | 69.0% | 95.4% | 0.5% | 2.7% | 0.0% |
| en | 3860 | 90.2% | 72.4% | 96.9% | 0.3% | 1.9% | 0.0% |
| es | 503 | 88.7% | 70.2% | 95.2% | 0.6% | 2.8% | 0.0% |
| fr | 475 | 90.1% | 66.7% | 96.5% | 0.0% | 2.3% | 0.0% |
| nl | 1058 | 87.5% | 67.9% | 95.0% | 0.1% | 3.3% | 0.0% |
| ALL | 6303 | 89.3% | 70.8% | 96.3% | 0.3% | 2.3% | 0.0% |

## Held out, open questions per source

| Source | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| gen2-pages | 35 | 51.4% | 45.7% | 87.5% | 2.9% | 2.9% | 0.0% |
| gen2-routines | 27 | 29.6% | 44.4% | 58.3% | 14.8% | 3.7% | 0.0% |
| gen2-tools | 23 | 52.2% | 52.2% | 75.0% | 8.7% | 4.3% | 0.0% |
| gliclass | 408 | 72.5% | 66.9% | 89.0% | 3.9% | 3.4% | 0.0% |
| zeroshot | 284 | 74.7% | 62.0% | 96.0% | 0.0% | 2.5% | 0.0% |
| ALL | 777 | 70.3% | 62.9% | 90.4% | 3.0% | 3.1% | 0.0% |

## Probe: 20 languages (never seen)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| ar | 150 | 71.3% | 16.0% | 79.2% | 0.0% | 3.3% | 0.0% |
| de | 150 | 75.3% | 24.0% | 97.2% | 0.0% | 0.7% | 0.0% |
| el | 150 | 65.3% | 18.0% | 81.5% | 0.7% | 2.7% | 0.0% |
| en | 150 | 77.3% | 29.3% | 86.4% | 0.0% | 4.0% | 0.0% |
| es | 150 | 74.7% | 24.0% | 88.9% | 0.0% | 2.7% | 0.0% |
| fr | 150 | 74.7% | 26.0% | 79.5% | 0.0% | 5.3% | 0.0% |
| he | 150 | 59.3% | 9.3% | 64.3% | 0.0% | 3.3% | 0.0% |
| hi | 150 | 60.7% | 12.0% | 77.8% | 0.7% | 2.0% | 0.0% |
| id | 150 | 74.7% | 20.7% | 80.7% | 0.0% | 4.0% | 0.0% |
| it | 150 | 71.3% | 22.0% | 81.8% | 0.7% | 3.3% | 0.0% |
| ja | 150 | 71.3% | 14.7% | 81.8% | 0.0% | 2.7% | 0.0% |
| ko | 150 | 65.3% | 14.0% | 95.2% | 0.0% | 0.7% | 0.0% |
| nl | 150 | 65.3% | 29.3% | 90.9% | 0.0% | 2.7% | 0.0% |
| pl | 150 | 72.0% | 20.0% | 83.3% | 0.7% | 2.7% | 0.0% |
| pt | 150 | 76.7% | 22.0% | 84.9% | 0.0% | 3.3% | 0.0% |
| ru | 150 | 78.0% | 28.0% | 92.9% | 0.0% | 2.0% | 0.0% |
| sv | 150 | 72.7% | 12.7% | 89.5% | 0.0% | 1.3% | 0.0% |
| th | 150 | 76.7% | 13.3% | 70.0% | 0.0% | 4.0% | 0.0% |
| tr | 150 | 50.7% | 8.7% | 76.9% | 0.7% | 1.3% | 0.0% |
| zh | 150 | 70.0% | 18.7% | 96.4% | 0.0% | 0.7% | 0.0% |
| ALL | 3000 | 70.2% | 19.1% | 85.4% | 0.2% | 2.6% | 0.0% |

## Probe: new kinds of questions about new text

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| probe-areas | 200 | 77.5% | 71.5% | 100.0% | 0.0% | 0.0% | 0.0% |
| probe-family | 100 | 69.0% | 67.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| probe-money | 47 | 80.8% | 85.1% | 95.0% | 2.1% | 2.1% | 0.0% |
| probe-purchase | 53 | 96.2% | 100.0% | 96.2% | 1.9% | 1.9% | 0.0% |
| probe-urgency | 50 | 80.0% | 74.0% | 94.6% | 0.0% | 4.0% | 0.0% |
| ALL | 450 | 78.4% | 75.6% | 98.2% | 0.4% | 0.9% | 0.0% |

## Probe: held-out browser question families (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| contract-type | 10 | 40.0% | 60.0% | 66.7% | 10.0% | 10.0% | 0.0% |
| customer-intent | 10 | 60.0% | 70.0% | 85.7% | 10.0% | 0.0% | 0.0% |
| destructive-action | 10 | 60.0% | 60.0% | 83.3% | 10.0% | 0.0% | 0.0% |
| email-meeting-request | 10 | 70.0% | 80.0% | 87.5% | 10.0% | 0.0% | 0.0% |
| free-returns | 10 | 40.0% | 70.0% | 57.1% | 20.0% | 10.0% | 0.0% |
| message-contains-credentials | 10 | 50.0% | 90.0% | 55.6% | 40.0% | 0.0% | 0.0% |
| oauth-scope-excessive | 10 | 10.0% | 20.0% | 50.0% | 10.0% | 0.0% | 0.0% |
| summary-faithful | 10 | 60.0% | 90.0% | 66.7% | 30.0% | 0.0% | 0.0% |
| ticket-availability | 10 | 40.0% | 60.0% | 50.0% | 20.0% | 10.0% | 0.0% |
| tool-output-prompt-injection | 10 | 50.0% | 70.0% | 71.4% | 20.0% | 0.0% | 0.0% |
| ALL | 480 | 45.0% | 44.4% | 82.6% | 4.4% | 3.3% | 0.0% |

## Probe: held-out browser question families, second set (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| applicable-tariff | 8 | 12.5% | 12.5% | 0.0% | 12.5% | 0.0% | 0.0% |
| click-accepts-legal-terms | 8 | 50.0% | 62.5% | 80.0% | 12.5% | 0.0% | 0.0% |
| contact-field-misplaced | 8 | 0.0% | 37.5% | 0.0% | 37.5% | 0.0% | 0.0% |
| deductible-remaining | 8 | 12.5% | 25.0% | 50.0% | 12.5% | 0.0% | 0.0% |
| open-reveals-seen-status | 8 | 62.5% | 75.0% | 83.3% | 12.5% | 0.0% | 0.0% |
| outage-affects-address | 8 | 12.5% | 75.0% | 16.7% | 37.5% | 25.0% | 0.0% |
| payment-method-matches-request | 8 | 50.0% | 62.5% | 80.0% | 12.5% | 0.0% | 0.0% |
| sick-note-required | 8 | 37.5% | 62.5% | 60.0% | 12.5% | 12.5% | 0.0% |
| sql-statement-effect | 8 | 25.0% | 37.5% | 66.7% | 12.5% | 0.0% | 0.0% |
| weekly-habit-target-met | 8 | 50.0% | 75.0% | 66.7% | 25.0% | 0.0% | 0.0% |
| ALL | 576 | 39.6% | 38.7% | 83.9% | 4.3% | 1.9% | 0.0% |

## Probe: 18-28 tab groups

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| probe-many-groups | 150 | 57.3% | 26.0% | 76.9% | 0.0% | 6.0% | 0.0% |
| ALL | 150 | 57.3% | 26.0% | 76.9% | 0.0% | 6.0% | 0.0% |
