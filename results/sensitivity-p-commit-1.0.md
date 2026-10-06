# Sensitivity: every lost response had already committed (p_commit_given_timeout = 1.0)

Worst case for the caller: whenever a response is lost, the write had happened. Same 200 tasks x 16 trials per cell, seed 0.

| faults | policy | task success | duplicate charges | silent failures | unconfirmed | pass^8 |
|---|---|---|---|---|---|---|
| none | naive-retry | 100.0% | 0.0% (0.0% to 0.1%) | 0.0% | 0.0% | 100.0% |
| none | verify-before-retry | 100.0% | 0.0% (0.0% to 0.1%) | 0.0% | 0.0% | 100.0% |
| none | idempotency-key | 100.0% | 0.0% (0.0% to 0.1%) | 0.0% | 0.0% | 100.0% |
| none | idempotency-key+verify | 100.0% | 0.0% (0.0% to 0.1%) | 0.0% | 0.0% | 100.0% |
| low | naive-retry | 91.4% | 5.7% (5.0% to 6.6%) | 8.6% | 0.0% | 47.8% |
| low | verify-before-retry | 95.8% | 0.4% (0.2% to 0.7%) | 4.2% | 0.0% | 70.0% |
| low | idempotency-key | 96.9% | 0.0% (0.0% to 0.1%) | 3.0% | 0.0% | 77.4% |
| low | idempotency-key+verify | 100.0% | 0.0% (0.0% to 0.1%) | 0.0% | 0.0% | 100.0% |
| medium | naive-retry | 72.5% | 19.5% (18.2% to 20.9%) | 26.7% | 0.0% | 8.1% |
| medium | verify-before-retry | 87.4% | 3.1% (2.5% to 3.7%) | 12.6% | 0.0% | 35.0% |
| medium | idempotency-key | 89.4% | 0.0% (0.0% to 0.1%) | 9.9% | 0.0% | 39.7% |
| medium | idempotency-key+verify | 100.0% | 0.0% (0.0% to 0.1%) | 0.0% | 0.0% | 100.0% |
| high | naive-retry | 54.8% | 35.4% (33.8% to 37.1%) | 40.7% | 0.0% | 0.8% |
| high | verify-before-retry | 77.1% | 8.6% (7.7% to 9.6%) | 22.9% | 0.0% | 11.3% |
| high | idempotency-key | 81.5% | 0.0% (0.0% to 0.1%) | 14.5% | 0.0% | 20.6% |
| high | idempotency-key+verify | 100.0% | 0.0% (0.0% to 0.1%) | 0.0% | 0.0% | 100.0% |
