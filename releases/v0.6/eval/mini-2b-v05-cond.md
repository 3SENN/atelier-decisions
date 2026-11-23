# Capabilities: mini-2b-v05-cond

`runs\mini-2b-v05-cond` (mini-2b-v05-cond-Q5_K_M-e4.decider.gguf, 1285.6 MB), measured 2026-11-23 11:44.

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
| skills | 2/10 | 10/10 | 10/10 | 0 |

Tests: 10 passed, 0 failed.

## Held out, per decision

| Decision | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| agentClicks | 516 | 93.8% | 51.5% | 91.3% | 0.0% | 4.5% | 0.0% |
| dictation | 167 | 82.6% | 64.7% | 96.3% | 0.0% | 2.4% | 0.0% |
| effort | 739 | 80.1% | 41.1% | 86.5% | 0.0% | 5.5% | 0.0% |
| labeling | 34 | 91.2% | 50.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| memory | 591 | 88.8% | 41.1% | 97.9% | 0.0% | 0.9% | 0.0% |
| needsYou | 568 | 94.2% | 95.6% | 96.5% | 0.4% | 3.0% | 0.0% |
| searchOrAsk | 664 | 84.5% | 69.3% | 94.3% | 0.1% | 3.8% | 0.0% |
| skills | 2215 | 95.9% | 96.2% | 98.6% | 0.4% | 0.9% | 0.0% |
| tabs | 425 | 94.6% | 72.7% | 97.7% | 0.0% | 1.7% | 0.0% |
| voiceTarget | 384 | 85.7% | 49.7% | 91.6% | 0.0% | 4.2% | 0.0% |
| ALL | 6303 | 90.8% | 72.5% | 96.3% | 0.2% | 2.5% | 0.0% |

## Held out, per language

| Language | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| de | 407 | 86.7% | 68.8% | 96.1% | 0.2% | 2.5% | 0.0% |
| en | 3860 | 91.6% | 74.4% | 96.7% | 0.2% | 2.3% | 0.0% |
| es | 503 | 88.9% | 70.2% | 95.5% | 0.4% | 2.8% | 0.0% |
| fr | 475 | 89.7% | 69.0% | 94.5% | 0.2% | 3.6% | 0.0% |
| nl | 1058 | 90.6% | 69.8% | 95.9% | 0.1% | 2.7% | 0.0% |
| ALL | 6303 | 90.8% | 72.5% | 96.3% | 0.2% | 2.5% | 0.0% |

## Held out, open questions per source

| Source | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| gen2-pages | 35 | 68.6% | 60.0% | 95.2% | 0.0% | 2.9% | 0.0% |
| gen2-routines | 27 | 40.7% | 48.1% | 76.9% | 7.4% | 3.7% | 0.0% |
| gen2-tools | 23 | 52.2% | 52.2% | 83.3% | 0.0% | 8.7% | 0.0% |
| gliclass | 408 | 76.0% | 66.2% | 94.1% | 1.2% | 2.7% | 0.0% |
| zeroshot | 284 | 77.8% | 61.6% | 98.9% | 0.0% | 0.7% | 0.0% |
| ALL | 777 | 74.4% | 63.2% | 95.1% | 0.9% | 2.2% | 0.0% |

## Probe: 20 languages (never seen)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| ar | 150 | 76.0% | 23.3% | 80.0% | 0.0% | 4.7% | 0.0% |
| de | 150 | 79.3% | 36.0% | 87.0% | 0.0% | 4.7% | 0.0% |
| el | 150 | 63.3% | 17.3% | 76.9% | 0.7% | 3.3% | 0.0% |
| en | 150 | 82.0% | 35.3% | 86.8% | 0.0% | 4.7% | 0.0% |
| es | 150 | 80.0% | 33.3% | 86.0% | 0.0% | 4.7% | 0.0% |
| fr | 150 | 79.3% | 37.3% | 76.8% | 0.0% | 8.7% | 0.0% |
| he | 150 | 65.3% | 17.3% | 73.1% | 0.7% | 4.0% | 0.0% |
| hi | 150 | 72.7% | 26.0% | 87.2% | 0.0% | 3.3% | 0.0% |
| id | 150 | 77.3% | 23.3% | 82.9% | 0.0% | 4.0% | 0.0% |
| it | 150 | 73.3% | 27.3% | 80.5% | 0.0% | 5.3% | 0.0% |
| ja | 150 | 74.0% | 16.7% | 84.0% | 0.7% | 2.0% | 0.0% |
| ko | 150 | 64.0% | 16.0% | 83.3% | 0.7% | 2.0% | 0.0% |
| nl | 150 | 76.7% | 42.0% | 92.1% | 0.0% | 3.3% | 0.0% |
| pl | 150 | 76.0% | 28.0% | 81.0% | 0.7% | 4.7% | 0.0% |
| pt | 150 | 77.3% | 26.7% | 80.0% | 0.0% | 5.3% | 0.0% |
| ru | 150 | 78.7% | 34.0% | 86.3% | 0.0% | 4.7% | 0.0% |
| sv | 150 | 74.7% | 17.3% | 84.6% | 0.0% | 2.7% | 0.0% |
| th | 150 | 80.0% | 16.7% | 76.0% | 0.0% | 4.0% | 0.0% |
| tr | 150 | 57.3% | 18.7% | 82.1% | 0.0% | 3.3% | 0.0% |
| zh | 150 | 75.3% | 24.0% | 97.2% | 0.0% | 0.7% | 0.0% |
| ALL | 3000 | 74.1% | 25.8% | 83.9% | 0.2% | 4.0% | 0.0% |

## Probe: new kinds of questions about new text

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| probe-areas | 200 | 72.0% | 69.0% | 97.8% | 0.0% | 1.5% | 0.0% |
| probe-family | 100 | 86.0% | 86.0% | 98.8% | 0.0% | 1.0% | 0.0% |
| probe-money | 47 | 83.0% | 85.1% | 97.5% | 0.0% | 2.1% | 0.0% |
| probe-purchase | 53 | 94.3% | 96.2% | 98.0% | 0.0% | 1.9% | 0.0% |
| probe-urgency | 50 | 76.0% | 70.0% | 94.3% | 0.0% | 4.0% | 0.0% |
| ALL | 450 | 79.3% | 77.8% | 97.7% | 0.0% | 1.8% | 0.0% |

## Probe: held-out browser question families (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| brand-impersonating-domain | 10 | 20.0% | 50.0% | 40.0% | 30.0% | 0.0% | 0.0% |
| destructive-action | 10 | 80.0% | 80.0% | 87.5% | 10.0% | 0.0% | 0.0% |
| free-returns | 10 | 80.0% | 90.0% | 77.8% | 20.0% | 0.0% | 0.0% |
| invite-conflicts-with-agenda | 10 | 20.0% | 30.0% | 66.7% | 10.0% | 0.0% | 0.0% |
| message-contains-credentials | 10 | 80.0% | 90.0% | 88.9% | 10.0% | 0.0% | 0.0% |
| price-currency-as-expected | 10 | 40.0% | 60.0% | 66.7% | 20.0% | 0.0% | 0.0% |
| result-is-ad | 10 | 60.0% | 80.0% | 75.0% | 20.0% | 0.0% | 0.0% |
| support-ticket-escalate | 10 | 80.0% | 90.0% | 88.9% | 10.0% | 0.0% | 0.0% |
| tool-output-prompt-injection | 10 | 80.0% | 90.0% | 88.9% | 10.0% | 0.0% | 0.0% |
| total-cost-within-budget | 10 | 30.0% | 50.0% | 40.0% | 10.0% | 20.0% | 0.0% |
| ALL | 480 | 53.3% | 51.2% | 86.6% | 4.2% | 2.7% | 0.0% |

## Probe: held-out browser question families, second set (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| attempts-remaining | 8 | 62.5% | 62.5% | 80.0% | 12.5% | 0.0% | 0.0% |
| contact-field-misplaced | 8 | 12.5% | 37.5% | 33.3% | 25.0% | 0.0% | 0.0% |
| energy-bill-estimated-or-actual | 8 | 37.5% | 50.0% | 75.0% | 12.5% | 0.0% | 0.0% |
| mortgage-rate-change-direction | 8 | 50.0% | 75.0% | 66.7% | 12.5% | 12.5% | 0.0% |
| outage-affects-address | 8 | 37.5% | 75.0% | 33.3% | 37.5% | 12.5% | 0.0% |
| regenerate-invalidates-old | 8 | 75.0% | 87.5% | 85.7% | 12.5% | 0.0% | 0.0% |
| sick-note-required | 8 | 25.0% | 50.0% | 50.0% | 12.5% | 12.5% | 0.0% |
| study-credit-requirement-met | 8 | 50.0% | 62.5% | 80.0% | 12.5% | 0.0% | 0.0% |
| target-reachability | 8 | 50.0% | 62.5% | 80.0% | 12.5% | 0.0% | 0.0% |
| tax-letter-kind | 8 | 37.5% | 50.0% | 75.0% | 12.5% | 0.0% | 0.0% |
| ALL | 576 | 50.9% | 49.8% | 88.8% | 3.1% | 2.4% | 0.0% |

## Probe: held-out browser question families, third set (gen4) (the 10 with most contradictions, and all)

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| added-item-appears-in-list | 8 | 87.5% | 100.0% | 87.5% | 12.5% | 0.0% | 0.0% |
| cashback-claim-conditions-met | 8 | 62.5% | 87.5% | 71.4% | 25.0% | 0.0% | 0.0% |
| compose-mode-open | 8 | 25.0% | 37.5% | 33.3% | 12.5% | 12.5% | 0.0% |
| field-unit-matches-value | 8 | 62.5% | 87.5% | 57.1% | 37.5% | 0.0% | 0.0% |
| masked-contact-matches-known | 8 | 37.5% | 87.5% | 42.9% | 37.5% | 12.5% | 0.0% |
| out-of-office-covers-date | 8 | 25.0% | 62.5% | 40.0% | 37.5% | 0.0% | 0.0% |
| regex-matches-sample | 8 | 37.5% | 75.0% | 50.0% | 25.0% | 12.5% | 0.0% |
| route-direction-matches-request | 8 | 62.5% | 87.5% | 71.4% | 25.0% | 0.0% | 0.0% |
| signup-email-already-registered | 8 | 75.0% | 100.0% | 75.0% | 25.0% | 0.0% | 0.0% |
| withdrawal-right-applies | 8 | 37.5% | 62.5% | 60.0% | 25.0% | 0.0% | 0.0% |
| ALL | 576 | 58.0% | 60.2% | 87.6% | 5.4% | 2.1% | 0.0% |

## Probe: 18-28 tab groups

| Group | n | Same verdict as teacher | Decides (coverage) | Right when it decides (precision) | Sure and contradicts | Sure where teacher unsure | Declined |
|---|---|---|---|---|---|---|---|
| probe-many-groups | 150 | 55.3% | 20.0% | 80.0% | 0.0% | 4.0% | 0.0% |
| ALL | 150 | 55.3% | 20.0% | 80.0% | 0.0% | 4.0% | 0.0% |
