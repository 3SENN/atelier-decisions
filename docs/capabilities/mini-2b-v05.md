# Capabilities: mini-2b-v05

`runs\mini-2b-v05` (mini-2b-v05-Q5_K_M-e4.decider.gguf, 1285.6 MB), measured 2026-11-23 09:27.

## Atelier's own eval

| Decision | Rules alone | With the model | Right when sure | Unsure |
|---|---|---|---|---|
| needsYou | 5/10 | 9/10 | 8/8 | 2 |
| dictation | 4/8 | 6/8 | 2/2 | 6 |
| agentClicks | 5/12 | 12/12 | 8/8 | 4 |
| searchOrAsk | 5/10 | 10/10 | 10/10 | 0 |
| voiceTarget | 3/6 | 6/6 | 3/3 | 3 |
| effort | 3/12 | 11/12 | 10/10 | 2 |
| memory | 5/10 | 10/10 | 5/5 | 5 |
| tabs | 0/12 | 10/12 | 10/10 | 2 |
| skills | 2/10 | 10/10 | 10/10 | 0 |

Tests: 10 passed, 0 failed.

## Held out, per decision

| Decision | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| agentClicks | 516 | 94.4% | 51.4% | 92.1% | 0.0% | 4.1% | 0.0% |
| dictation | 167 | 68.9% | 47.3% | 98.7% | 0.0% | 0.6% | 0.0% |
| effort | 739 | 81.9% | 44.5% | 85.7% | 0.0% | 6.4% | 0.0% |
| labeling | 34 | 91.2% | 50.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| memory | 591 | 89.8% | 42.8% | 97.2% | 0.0% | 1.2% | 0.0% |
| needsYou | 568 | 94.9% | 94.0% | 97.8% | 0.2% | 1.9% | 0.0% |
| searchOrAsk | 664 | 84.5% | 68.1% | 95.1% | 0.1% | 3.2% | 0.0% |
| skills | 2215 | 96.1% | 96.8% | 98.5% | 0.3% | 1.2% | 0.0% |
| tabs | 425 | 97.2% | 72.5% | 99.7% | 0.0% | 0.2% | 0.0% |
| voiceTarget | 384 | 81.0% | 39.8% | 96.1% | 0.0% | 1.6% | 0.0% |
| ALL | 6303 | 90.8% | 71.9% | 96.7% | 0.1% | 2.2% | 0.0% |

## Held out, per language

| Language | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| de | 407 | 87.2% | 70.3% | 95.5% | 0.2% | 2.9% | 0.0% |
| en | 3860 | 91.8% | 74.0% | 97.1% | 0.1% | 2.1% | 0.0% |
| es | 503 | 89.3% | 71.4% | 95.0% | 0.4% | 3.2% | 0.0% |
| fr | 475 | 88.8% | 65.7% | 96.2% | 0.2% | 2.3% | 0.0% |
| nl | 1058 | 90.1% | 68.0% | 96.8% | 0.0% | 2.2% | 0.0% |
| ALL | 6303 | 90.8% | 71.9% | 96.7% | 0.1% | 2.2% | 0.0% |

## Held out, open questions per source

| Source | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| gen2-pages | 35 | 74.3% | 65.7% | 95.7% | 0.0% | 2.9% | 0.0% |
| gen2-routines | 27 | 55.6% | 74.1% | 75.0% | 11.1% | 7.4% | 0.0% |
| gen2-tools | 23 | 52.2% | 47.8% | 81.8% | 4.3% | 4.3% | 0.0% |
| gliclass | 408 | 79.7% | 70.8% | 93.1% | 2.2% | 2.7% | 0.0% |
| zeroshot | 284 | 81.0% | 66.2% | 97.9% | 0.0% | 1.4% | 0.0% |
| ALL | 777 | 78.2% | 68.3% | 94.0% | 1.7% | 2.5% | 0.0% |

## Probe: 20 languages (never seen)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| ar | 150 | 71.3% | 24.0% | 69.4% | 0.0% | 7.3% | 0.0% |
| de | 150 | 76.7% | 34.7% | 84.6% | 0.0% | 5.3% | 0.0% |
| el | 150 | 62.0% | 21.3% | 68.8% | 0.7% | 6.0% | 0.0% |
| en | 150 | 81.3% | 30.7% | 91.3% | 0.0% | 2.7% | 0.0% |
| es | 150 | 76.0% | 38.7% | 75.9% | 0.0% | 9.3% | 0.0% |
| fr | 150 | 81.3% | 38.0% | 79.0% | 0.0% | 8.0% | 0.0% |
| he | 150 | 67.3% | 16.7% | 80.0% | 0.7% | 2.7% | 0.0% |
| hi | 150 | 70.0% | 23.3% | 85.7% | 0.0% | 3.3% | 0.0% |
| id | 150 | 75.3% | 22.7% | 79.4% | 0.0% | 4.7% | 0.0% |
| it | 150 | 77.3% | 27.3% | 87.8% | 0.0% | 3.3% | 0.0% |
| ja | 150 | 74.0% | 17.3% | 84.6% | 0.0% | 2.7% | 0.0% |
| ko | 150 | 66.7% | 19.3% | 86.2% | 0.0% | 2.7% | 0.0% |
| nl | 150 | 78.0% | 43.3% | 92.3% | 0.0% | 3.3% | 0.0% |
| pl | 150 | 78.7% | 27.3% | 87.8% | 0.0% | 3.3% | 0.0% |
| pt | 150 | 80.7% | 28.7% | 83.7% | 0.0% | 4.7% | 0.0% |
| ru | 150 | 81.3% | 39.3% | 84.8% | 0.0% | 6.0% | 0.0% |
| sv | 150 | 73.3% | 17.3% | 80.8% | 0.0% | 3.3% | 0.0% |
| th | 150 | 75.3% | 14.7% | 63.6% | 0.0% | 5.3% | 0.0% |
| tr | 150 | 56.7% | 18.7% | 78.6% | 0.7% | 3.3% | 0.0% |
| zh | 150 | 78.0% | 28.0% | 95.2% | 0.0% | 1.3% | 0.0% |
| ALL | 3000 | 74.1% | 26.6% | 82.9% | 0.1% | 4.4% | 0.0% |

## Probe: new kinds of questions about new text

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| probe-areas | 200 | 78.0% | 75.0% | 98.0% | 0.0% | 1.5% | 0.0% |
| probe-family | 100 | 89.0% | 87.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| probe-money | 47 | 93.6% | 91.5% | 100.0% | 0.0% | 0.0% | 0.0% |
| probe-purchase | 53 | 96.2% | 98.1% | 98.1% | 0.0% | 1.9% | 0.0% |
| probe-urgency | 50 | 70.0% | 56.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| ALL | 450 | 83.3% | 80.0% | 98.9% | 0.0% | 0.9% | 0.0% |

## Probe: held-out browser question families (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| brand-impersonating-domain | 10 | 60.0% | 70.0% | 85.7% | 10.0% | 0.0% | 0.0% |
| commute-within-limit | 10 | 10.0% | 30.0% | 33.3% | 20.0% | 0.0% | 0.0% |
| email-thread-resolved | 10 | 70.0% | 70.0% | 85.7% | 10.0% | 0.0% | 0.0% |
| free-returns | 10 | 70.0% | 100.0% | 70.0% | 20.0% | 10.0% | 0.0% |
| invite-conflicts-with-agenda | 10 | 70.0% | 80.0% | 87.5% | 10.0% | 0.0% | 0.0% |
| message-contains-credentials | 10 | 80.0% | 90.0% | 88.9% | 10.0% | 0.0% | 0.0% |
| result-is-ad | 10 | 60.0% | 80.0% | 75.0% | 20.0% | 0.0% | 0.0% |
| ticket-availability | 10 | 10.0% | 40.0% | 25.0% | 10.0% | 20.0% | 0.0% |
| tool-output-prompt-injection | 10 | 80.0% | 90.0% | 88.9% | 10.0% | 0.0% | 0.0% |
| work-mode | 10 | 60.0% | 80.0% | 75.0% | 10.0% | 10.0% | 0.0% |
| ALL | 480 | 57.7% | 57.3% | 86.2% | 4.6% | 3.3% | 0.0% |

## Probe: held-out browser question families, second set (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| attempts-remaining | 8 | 62.5% | 62.5% | 80.0% | 12.5% | 0.0% | 0.0% |
| deductible-remaining | 8 | 37.5% | 50.0% | 75.0% | 12.5% | 0.0% | 0.0% |
| deposit-return-outcome | 8 | 37.5% | 50.0% | 75.0% | 12.5% | 0.0% | 0.0% |
| energy-bill-estimated-or-actual | 8 | 37.5% | 62.5% | 60.0% | 25.0% | 0.0% | 0.0% |
| number-format-locale | 8 | 25.0% | 62.5% | 40.0% | 37.5% | 0.0% | 0.0% |
| outage-affects-address | 8 | 50.0% | 75.0% | 33.3% | 50.0% | 0.0% | 0.0% |
| payment-method-matches-request | 8 | 75.0% | 87.5% | 85.7% | 12.5% | 0.0% | 0.0% |
| regenerate-invalidates-old | 8 | 75.0% | 87.5% | 85.7% | 12.5% | 0.0% | 0.0% |
| sick-note-required | 8 | 50.0% | 62.5% | 60.0% | 25.0% | 0.0% | 0.0% |
| tax-letter-kind | 8 | 75.0% | 87.5% | 85.7% | 12.5% | 0.0% | 0.0% |
| ALL | 576 | 53.8% | 52.1% | 88.7% | 4.5% | 1.4% | 0.0% |

## Probe: held-out browser question families, third set (gen4) (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| added-item-appears-in-list | 8 | 75.0% | 100.0% | 75.0% | 25.0% | 0.0% | 0.0% |
| compose-mode-open | 8 | 25.0% | 37.5% | 33.3% | 12.5% | 12.5% | 0.0% |
| field-unit-matches-value | 8 | 50.0% | 87.5% | 57.1% | 25.0% | 12.5% | 0.0% |
| file-upload-progress-state | 8 | 37.5% | 50.0% | 75.0% | 12.5% | 0.0% | 0.0% |
| listing-sale-format | 8 | 62.5% | 75.0% | 66.7% | 12.5% | 12.5% | 0.0% |
| out-of-office-covers-date | 8 | 37.5% | 62.5% | 60.0% | 25.0% | 0.0% | 0.0% |
| search-scope-restricted | 8 | 75.0% | 87.5% | 85.7% | 12.5% | 0.0% | 0.0% |
| signup-email-already-registered | 8 | 75.0% | 100.0% | 75.0% | 25.0% | 0.0% | 0.0% |
| visitor-entry-permitted | 8 | 62.5% | 87.5% | 71.4% | 25.0% | 0.0% | 0.0% |
| withdrawal-right-applies | 8 | 12.5% | 50.0% | 25.0% | 37.5% | 0.0% | 0.0% |
| ALL | 576 | 54.7% | 59.2% | 85.3% | 5.6% | 3.1% | 0.0% |

## Probe: 18-28 tab groups

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| probe-many-groups | 150 | 55.3% | 19.3% | 79.3% | 0.7% | 3.3% | 0.0% |
| ALL | 150 | 55.3% | 19.3% | 79.3% | 0.7% | 3.3% | 0.0% |
