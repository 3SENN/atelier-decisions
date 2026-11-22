# Capabilities: mini-0.8b-v05

`runs\mini-v05` (mini-v05-Q5_K_M-e4.decider.gguf, 515.2 MB), measured 2026-11-22 23:06.

## Atelier's own eval

| Decision | Rules alone | With the model | Right when sure | Unsure |
|---|---|---|---|---|
| needsYou | 5/10 | 9/10 | 8/8 | 2 |
| dictation | 4/8 | 6/8 | 2/2 | 6 |
| agentClicks | 5/12 | 12/12 | 8/8 | 4 |
| searchOrAsk | 5/10 | 10/10 | 10/10 | 0 |
| voiceTarget | 3/6 | 6/6 | 3/3 | 3 |
| effort | 3/12 | 11/12 | 11/11 | 1 |
| memory | 5/10 | 10/10 | 5/5 | 5 |
| tabs | 0/12 | 9/12 | 9/10 | 2 |
| skills | 2/10 | 10/10 | 10/10 | 0 |

Tests: 10 passed, 0 failed.

## Held out, per decision

| Decision | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| agentClicks | 516 | 94.8% | 52.1% | 91.8% | 0.0% | 4.3% | 0.0% |
| dictation | 167 | 85.0% | 68.3% | 95.6% | 0.0% | 3.0% | 0.0% |
| effort | 739 | 75.4% | 33.7% | 87.5% | 0.0% | 4.2% | 0.0% |
| labeling | 34 | 94.1% | 58.8% | 95.0% | 0.0% | 2.9% | 0.0% |
| memory | 591 | 89.8% | 42.8% | 97.2% | 0.0% | 1.2% | 0.0% |
| needsYou | 568 | 93.7% | 95.2% | 96.5% | 0.2% | 3.2% | 0.0% |
| searchOrAsk | 664 | 82.8% | 68.7% | 93.6% | 0.0% | 4.4% | 0.0% |
| skills | 2215 | 95.5% | 96.7% | 97.9% | 0.8% | 1.2% | 0.0% |
| tabs | 425 | 95.5% | 73.7% | 97.8% | 0.0% | 1.7% | 0.0% |
| voiceTarget | 384 | 81.0% | 41.4% | 94.3% | 0.0% | 2.3% | 0.0% |
| ALL | 6303 | 89.9% | 71.6% | 96.2% | 0.3% | 2.5% | 0.0% |

## Held out, per language

| Language | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| de | 407 | 87.5% | 71.0% | 95.2% | 0.2% | 3.2% | 0.0% |
| en | 3860 | 90.5% | 73.0% | 96.7% | 0.3% | 2.1% | 0.0% |
| es | 503 | 87.9% | 71.2% | 94.1% | 0.4% | 3.8% | 0.0% |
| fr | 475 | 89.3% | 68.0% | 95.0% | 0.0% | 3.4% | 0.0% |
| nl | 1058 | 89.8% | 68.8% | 95.7% | 0.5% | 2.5% | 0.0% |
| ALL | 6303 | 89.9% | 71.6% | 96.2% | 0.3% | 2.5% | 0.0% |

## Held out, open questions per source

| Source | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| gen2-pages | 35 | 62.9% | 54.3% | 89.5% | 5.7% | 0.0% | 0.0% |
| gen2-routines | 27 | 22.2% | 29.6% | 50.0% | 14.8% | 0.0% | 0.0% |
| gen2-tools | 23 | 52.2% | 47.8% | 81.8% | 4.3% | 4.3% | 0.0% |
| gliclass | 408 | 73.0% | 67.9% | 88.8% | 3.9% | 3.7% | 0.0% |
| zeroshot | 284 | 78.2% | 64.8% | 96.7% | 0.0% | 2.1% | 0.0% |
| ALL | 777 | 72.1% | 64.2% | 91.0% | 3.0% | 2.8% | 0.0% |

## Probe: 20 languages (never seen)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| ar | 150 | 73.3% | 20.7% | 77.4% | 0.0% | 4.7% | 0.0% |
| de | 150 | 76.7% | 33.3% | 86.0% | 0.0% | 4.7% | 0.0% |
| el | 150 | 60.7% | 18.7% | 67.9% | 0.7% | 5.3% | 0.0% |
| en | 150 | 76.7% | 27.3% | 87.8% | 0.0% | 3.3% | 0.0% |
| es | 150 | 76.0% | 24.7% | 89.2% | 0.7% | 2.0% | 0.0% |
| fr | 150 | 77.3% | 30.0% | 80.0% | 0.0% | 6.0% | 0.0% |
| he | 150 | 55.3% | 10.0% | 40.0% | 0.7% | 5.3% | 0.0% |
| hi | 150 | 65.3% | 21.3% | 75.0% | 1.3% | 4.0% | 0.0% |
| id | 150 | 70.7% | 26.0% | 66.7% | 0.0% | 8.7% | 0.0% |
| it | 150 | 65.3% | 18.0% | 74.1% | 0.0% | 4.7% | 0.0% |
| ja | 150 | 74.7% | 18.0% | 85.2% | 0.0% | 2.7% | 0.0% |
| ko | 150 | 60.7% | 13.3% | 80.0% | 0.0% | 2.7% | 0.0% |
| nl | 150 | 73.3% | 36.0% | 94.4% | 0.0% | 2.0% | 0.0% |
| pl | 150 | 72.0% | 28.0% | 73.8% | 0.7% | 6.7% | 0.0% |
| pt | 150 | 76.7% | 23.3% | 82.9% | 0.0% | 4.0% | 0.0% |
| ru | 150 | 80.0% | 32.7% | 89.8% | 0.0% | 3.3% | 0.0% |
| sv | 150 | 74.7% | 17.3% | 80.8% | 1.3% | 2.0% | 0.0% |
| th | 150 | 77.3% | 12.7% | 73.7% | 0.0% | 3.3% | 0.0% |
| tr | 150 | 55.3% | 15.3% | 82.6% | 0.0% | 2.7% | 0.0% |
| zh | 150 | 71.3% | 22.7% | 91.2% | 0.0% | 2.0% | 0.0% |
| ALL | 3000 | 70.7% | 22.5% | 81.0% | 0.3% | 4.0% | 0.0% |

## Probe: new kinds of questions about new text

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| probe-areas | 200 | 71.5% | 67.5% | 98.5% | 0.0% | 1.0% | 0.0% |
| probe-family | 100 | 78.0% | 76.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| probe-money | 47 | 72.3% | 80.8% | 89.5% | 6.4% | 2.1% | 0.0% |
| probe-purchase | 53 | 94.3% | 96.2% | 98.0% | 0.0% | 1.9% | 0.0% |
| probe-urgency | 50 | 88.0% | 74.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| ALL | 450 | 77.6% | 74.9% | 97.9% | 0.7% | 0.9% | 0.0% |

## Probe: held-out browser question families (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| chat-directed-at-user | 10 | 60.0% | 80.0% | 75.0% | 20.0% | 0.0% | 0.0% |
| destructive-action | 10 | 70.0% | 90.0% | 77.8% | 10.0% | 10.0% | 0.0% |
| energy-rating-meets-minimum | 10 | 10.0% | 20.0% | 50.0% | 10.0% | 0.0% | 0.0% |
| form-submission-result | 10 | 80.0% | 100.0% | 80.0% | 10.0% | 10.0% | 0.0% |
| invite-conflicts-with-agenda | 10 | 10.0% | 20.0% | 50.0% | 10.0% | 0.0% | 0.0% |
| message-contains-credentials | 10 | 80.0% | 90.0% | 88.9% | 10.0% | 0.0% | 0.0% |
| ticket-availability | 10 | 0.0% | 40.0% | 0.0% | 20.0% | 20.0% | 0.0% |
| tool-output-prompt-injection | 10 | 80.0% | 100.0% | 80.0% | 20.0% | 0.0% | 0.0% |
| total-cost-within-budget | 10 | 50.0% | 80.0% | 50.0% | 20.0% | 20.0% | 0.0% |
| work-mode | 10 | 40.0% | 60.0% | 66.7% | 10.0% | 10.0% | 0.0% |
| ALL | 480 | 44.0% | 43.8% | 81.4% | 4.8% | 3.3% | 0.0% |

## Probe: held-out browser question families, second set (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| contact-field-misplaced | 8 | 50.0% | 100.0% | 50.0% | 50.0% | 0.0% | 0.0% |
| dialog-choice-for-intent | 8 | 25.0% | 37.5% | 66.7% | 12.5% | 0.0% | 0.0% |
| outage-affects-address | 8 | 37.5% | 62.5% | 40.0% | 25.0% | 12.5% | 0.0% |
| payment-method-matches-request | 8 | 37.5% | 62.5% | 60.0% | 25.0% | 0.0% | 0.0% |
| recurring-edit-scope | 8 | 25.0% | 37.5% | 66.7% | 12.5% | 0.0% | 0.0% |
| refund-method-outcome | 8 | 50.0% | 62.5% | 60.0% | 12.5% | 12.5% | 0.0% |
| regenerate-invalidates-old | 8 | 87.5% | 100.0% | 87.5% | 12.5% | 0.0% | 0.0% |
| sql-statement-effect | 8 | 37.5% | 62.5% | 60.0% | 25.0% | 0.0% | 0.0% |
| tax-letter-kind | 8 | 25.0% | 50.0% | 50.0% | 25.0% | 0.0% | 0.0% |
| trial-ends-before-date | 8 | 37.5% | 62.5% | 40.0% | 25.0% | 12.5% | 0.0% |
| ALL | 576 | 41.1% | 41.0% | 82.6% | 5.4% | 1.7% | 0.0% |

## Probe: held-out browser question families, third set (gen4) (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| ab-test-result-conclusive | 8 | 0.0% | 25.0% | 0.0% | 25.0% | 0.0% | 0.0% |
| alcohol-sale-allowed-now | 8 | 25.0% | 50.0% | 50.0% | 25.0% | 0.0% | 0.0% |
| driver-covered-by-policy | 8 | 12.5% | 25.0% | 0.0% | 25.0% | 0.0% | 0.0% |
| field-unit-matches-value | 8 | 50.0% | 62.5% | 60.0% | 25.0% | 0.0% | 0.0% |
| marketplace-post-intent | 8 | 50.0% | 62.5% | 60.0% | 25.0% | 0.0% | 0.0% |
| message-send-outcome | 8 | 37.5% | 50.0% | 75.0% | 12.5% | 0.0% | 0.0% |
| rail-booking-state | 8 | 37.5% | 62.5% | 60.0% | 25.0% | 0.0% | 0.0% |
| signup-email-already-registered | 8 | 62.5% | 87.5% | 71.4% | 25.0% | 0.0% | 0.0% |
| visitor-entry-permitted | 8 | 0.0% | 37.5% | 0.0% | 37.5% | 0.0% | 0.0% |
| withdrawal-right-applies | 8 | 0.0% | 25.0% | 0.0% | 25.0% | 0.0% | 0.0% |
| ALL | 576 | 37.7% | 38.0% | 82.2% | 5.9% | 0.9% | 0.0% |

## Probe: 18-28 tab groups

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| probe-many-groups | 150 | 56.0% | 16.7% | 88.0% | 0.0% | 2.0% | 0.0% |
| ALL | 150 | 56.0% | 16.7% | 88.0% | 0.0% | 2.0% | 0.0% |
