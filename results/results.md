| faults | policy | task success | duplicate charges | silent failures | unconfirmed | pass^8 |
|---|---|---|---|---|---|---|
| none | naive-retry | 100.0% | 0.0% (0.0% to 0.1%) | 0.0% | 0.0% | 100.0% |
| none | verify-before-retry | 100.0% | 0.0% (0.0% to 0.1%) | 0.0% | 0.0% | 100.0% |
| none | idempotency-key | 100.0% | 0.0% (0.0% to 0.1%) | 0.0% | 0.0% | 100.0% |
| none | idempotency-key+verify | 100.0% | 0.0% (0.0% to 0.1%) | 0.0% | 0.0% | 100.0% |
| low | naive-retry | 93.9% | 3.2% (2.6% to 3.9%) | 6.1% | 0.0% | 59.9% |
| low | verify-before-retry | 95.9% | 0.2% (0.1% to 0.4%) | 4.1% | 0.0% | 71.0% |
| low | idempotency-key | 97.1% | 0.0% (0.0% to 0.1%) | 2.9% | 0.0% | 78.2% |
| low | idempotency-key+verify | 100.0% | 0.0% (0.0% to 0.1%) | 0.0% | 0.0% | 100.0% |
| medium | naive-retry | 80.6% | 10.2% (9.2% to 11.3%) | 18.3% | 0.0% | 18.8% |
| medium | verify-before-retry | 88.2% | 1.8% (1.4% to 2.3%) | 11.7% | 0.0% | 36.9% |
| medium | idempotency-key | 89.3% | 0.0% (0.0% to 0.1%) | 9.9% | 0.0% | 38.9% |
| medium | idempotency-key+verify | 99.8% | 0.0% (0.0% to 0.1%) | 0.0% | 0.2% | 98.5% |
| high | naive-retry | 66.2% | 19.8% (18.5% to 21.3%) | 29.1% | 0.0% | 3.8% |
| high | verify-before-retry | 79.1% | 5.4% (4.6% to 6.2%) | 20.2% | 0.0% | 14.3% |
| high | idempotency-key | 81.1% | 0.0% (0.0% to 0.1%) | 14.7% | 0.0% | 19.3% |
| high | idempotency-key+verify | 99.2% | 0.0% (0.0% to 0.1%) | 0.0% | 0.8% | 94.1% |
