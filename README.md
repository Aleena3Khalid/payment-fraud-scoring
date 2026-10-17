# Pre-Payment Fraud Scoring with Cost-Based Alert Thresholds

Fraud scoring on PaySim synthetic transaction data. The model scores TRANSFER and CASH_OUT payments using only information available before the payment is made, and alert thresholds are chosen from explicit assumptions about fraud losses and review costs.

**All results come from synthetic data (PaySim) and are not evidence of performance on real bank transactions.**

## Summary

- Scope: 2.77 million TRANSFER and CASH_OUT transactions, 8,213 frauds (0.30%).
- Chronological split: train, validation and test periods, plus a later stress period with a much higher fraud rate.
- Main model: XGBoost without the account-emptying features (Model B). Test PR-AUC 0.749.
- At the main threshold (review cost 1,000 per alert), the model caught 93.7% of test fraud at 9.0% precision, about 121 alerts per hour. Total cost fell by 99.0% compared with raising no alerts.
- With a capacity of about 20 alerts per hour, it caught 78.8% of fraud at 41.0% precision.
- The 6% of alerts scoring 0.5 or above contain 61% of caught fraud at 88% precision, which supports a tiered review process.

## Data

Source: [Synthetic Financial Datasets For Fraud Detection (PaySim)](https://www.kaggle.com/datasets/ealaxi/paysim1), E. Lopez-Rojas. One month of simulated mobile money transactions (743 hourly steps, 6,362,620 rows).

The data is not included in this repository. Download the CSV, save it as `data/raw/paysim.csv`, and run the notebooks in order.

## Data checks and design decisions

These came from `notebooks/01_data_checks.ipynb`.

| Finding | Decision |
|---|---|
| Fraud is 0.13% of all transactions. Predicting "not fraud" for everything scores 99.87% accuracy. | Evaluate with precision, recall and PR-AUC, not accuracy. |
| Fraud only occurs in TRANSFER and CASH_OUT. | Model these two types only (2,770,409 rows, 0.30% fraud). |
| `newbalanceOrig` and `newbalanceDest` are only known after the payment. | Excluded, so the model can score a payment before it is made. |
| Fraud is generated at about 11 cases per hour regardless of normal activity. Between hours 2 and 5 of each day, 10 to 22% of transactions are fraud. | `step` and hour of day excluded, so the model does not learn this simulator effect. See `reports/fraud_rate_by_hour.png`. |
| 99.85% of sending accounts appear only once. | Account IDs not used. Customer history features are not possible with this dataset. |
| PaySim's built-in rule (`isFlaggedFraud`) flags 16 transactions, all fraud: 100% precision, 0.19% recall. | Used as a baseline only. |
| "Payment equals the sender's full non-zero balance" identifies 97.6% of fraud with zero false positives. No legit transaction matches it. | Treated as a simulator pattern. Reported as a baseline, and the main model is trained without it (Model B). |
| Steps 401 to 743 hold 9% of transactions but 45% of fraud. | Main split uses steps 1 to 400. Steps 401 to 743 are kept as a separate stress test. |

Other simulator effects to be aware of: 47% of legit transfers come from accounts with a zero balance, most transactions send more than the sender's balance, and no fraud payment exceeds 10,000,000 (PaySim caps fraud payments at that amount).

## Features

Defined once in `src/features.py` and used by every notebook.

| Feature | Model A | Model B |
|---|---|---|
| `is_transfer` | yes | yes |
| `amount` | yes | yes |
| `oldbalanceOrg`, `oldbalanceDest` | yes | yes |
| `orig_balance_zero`, `dest_balance_zero` | yes | yes |
| `drains_account` (payment equals full non-zero balance) | yes | no |
| `amount_to_balance` (amount divided by sender balance) | yes | no |

## Split

| Set | Steps | Rows | Frauds | Fraud rate |
|---|---|---|---|---|
| Train | 1 to 299 | 1,766,373 | 3,393 | 0.192% |
| Validation | 300 to 349 | 381,950 | 494 | 0.129% |
| Test | 350 to 400 | 381,213 | 590 | 0.155% |
| Stress | 401 to 743 | 240,873 | 3,736 | 1.551% |

Cut-offs fall on whole steps, so each set comes strictly after the previous one. The model, features and thresholds were all chosen on validation. The test and stress sets were used once, at the end.

## Model comparison (validation)

| Model | PR-AUC |
|---|---|
| Rule: `isFlaggedFraud` | 0.001 (no alerts in this period) |
| Rule: payment empties the account | 0.990 |
| Logistic regression A | 1.000 |
| XGBoost A | 1.000 |
| Logistic regression B | 0.030 |
| XGBoost B | 0.717 |

With the account-emptying features, both models rank fraud perfectly, which confirms how much this one pattern dominates PaySim. Without them, logistic regression drops to 0.030 and XGBoost to 0.717. XGBoost B goes forward because it is the setting where the trade-off between catching fraud and raising false alarms is real.

## Choosing the threshold

Total cost for a threshold = amount lost to missed fraud + (number of alerts x cost per review).

PaySim has no currency and the real review cost is unknown, so three review costs were tested. With no alerts at all, the validation fraud loss is 656.7 million.

| Cost per review | Best threshold | Alerts | Frauds caught (of 494) | Precision | Recall | Total cost |
|---|---|---|---|---|---|---|
| 100 | 0.005 | 10,875 | 490 | 4.5% | 99.2% | 1.23M |
| 1,000 | 0.010 | 5,936 | 470 | 7.9% | 95.1% | 8.33M |
| 10,000 | 0.060 | 968 | 386 | 39.9% | 78.1% | 23.80M |

The cheaper a review, the more alerts are worth raising. Missed frauds at the main threshold are mostly small: about 100,000 each, against an average fraud of 1.33 million.

The same question from a capacity point of view (validation):

| Alerts per hour | Recall | Precision |
|---|---|---|
| 10 | 68.8% | 68.0% |
| 20 | 78.1% | 38.6% |
| 50 | 86.6% | 17.1% |
| 100 | 93.7% | 9.3% |

Two settings were fixed before testing: the main setting (threshold 0.010, review cost 1,000) and a capacity setting (threshold 0.0565, about 20 alerts per hour).

## Test results

Test PR-AUC: 0.749 (validation: 0.717).

| Setting | Alerts per hour | Frauds caught (of 590) | Precision | Recall | Total cost | Reduction vs no alerts |
|---|---|---|---|---|---|---|
| Main (0.010) | 121 | 553 | 9.0% | 93.7% | 10.27M | 99.0% |
| Capacity (0.0565) | 22 | 465 | 41.0% | 78.8% | 18.35M | 98.3% |
| Rule: `isFlaggedFraud` | 0 | 1 | 100% | 0.2% | 1,074M | 0.5% |

Results were close to validation on every measure, so the thresholds carried over to unseen data.

## Subgroup results (test, main setting)

These are operational subgroups. PaySim has no customer attributes, so fairness across customer groups cannot be assessed.

| Group | Frauds | Recall | False alarm rate |
|---|---|---|---|
| TRANSFER | 295 | 100% | 1.1% |
| CASH_OUT | 295 | 87.5% | 1.6% |
| Amount Q1 (lowest) | 95 | 85.3% | 2.0% |
| Amount Q2 | 66 | 78.8% | 1.5% |
| Amount Q3 | 84 | 89.3% | 1.4% |
| Amount Q4 (highest) | 345 | 100% | 1.0% |
| Sender balance above 0 | 589 | 93.9% | 2.8% |
| Sender balance zero | 1 | 0% | 0.01% |

All missed frauds are CASH_OUT. Every high-value fraud is caught. The model almost never flags zero-balance senders, which in PaySim are nearly always legit (a simulator effect). On transactions where the sender has a positive balance, the false alarm rate is 2.8%, about double the overall 1.5%. In a real setting, the higher figure is the more realistic one.

## Stress test (steps 401 to 743, model and threshold unchanged)

| Steps | Fraud rate | Alerts per hour | Precision | Recall |
|---|---|---|---|---|
| 401 to 500 | 0.93% | 28.7 | 36.2% | 96.0% |
| 501 to 600 | 1.30% | 23.0 | 43.4% | 94.9% |
| 601 to 700 | 2.90% | 17.2 | 61.8% | 95.6% |
| 701 to 743 | 9.39% | 14.1 | 76.6% | 95.1% |

Recall stayed at 95 to 96% while the fraud rate rose more than tenfold. Precision and alert volume changed a lot, because false alarms scale with the volume of legit transactions. A fixed threshold can therefore produce very different workloads from one period to the next, which is why alert volume and precision should be monitored over time.

This tests a change in the fraud rate, not in fraud behaviour. PaySim generates fraud the same way throughout.

## Which transactions to review, and why

Explanations use SHAP values from XGBoost (`pred_contribs`), checked to add up exactly to each transaction's model output (maximum difference 0.00003).

**What drives alerts.** On alerted test transactions, sender balance and amount have the largest effect (mean absolute contribution 2.0 and 1.0 log-odds), followed by receiving balance (0.6) and transaction type (0.4). XGBoost's built-in importance ranks a zero receiving balance first, but that measure reflects training splits across all rows. SHAP reflects each alert's actual score, so it is used here. See `reports/shap_alerts.png`.

**Fraud vs false alarms.** Among alerts, caught frauds receive two to four times stronger contributions from sender balance, amount, receiving balance and transfer type than false alarms do. False alarms are mostly driven by sender balance alone.

**Priority by score band.**

| Score | Alerts | Frauds | Precision | Share of caught fraud |
|---|---|---|---|---|
| 0.5 to 1.0 | 385 | 338 | 87.8% | 61% |
| 0.2 to 0.5 | 258 | 86 | 33.3% | 16% |
| 0.05 to 0.2 | 572 | 41 | 7.2% | 7% |
| 0.01 to 0.05 | 4,963 | 88 | 1.8% | 16% |

A suggested tiered process based on these results:

- Score 0.5 or above: review first, and consider holding the payment until checked.
- 0.2 to 0.5: standard review.
- Below 0.2: too many alerts for manual review at this precision. A lighter check, such as asking the customer to confirm the payment, would fit better. These alerts still contain 16% of caught fraud, so they should not simply be dropped.

The actual policy would depend on the bank's processes, costs and customer impact.

**Example alerts.** Each alert lists up to three features that raised its score, in plain language. In the test data, the highest-scoring alerts are real frauds where the payment equals the sender's full balance, even though Model B never saw the account-emptying feature. The highest-scoring false alarms are transfers to empty receiving accounts where the amount is two to three times the sender's balance, which would not be possible in a real account. See `reports/review_queue_example.csv`.

## Limitations

- Synthetic data. PaySim fraud is designed around emptying accounts, so even Model B partly relies on a simulator pattern.
- No customer history, device, location or payee information, which real fraud models depend on.
- No authorised push payment (APP) scams, where the customer is tricked into sending money themselves. These are a major fraud type in UK banking.
- Review costs are assumptions. A bank would supply real figures. The method would stay the same.
- The stress test covers a change in fraud rate only, not fraudsters changing their behaviour.

## How to run

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Download the dataset to `data/raw/paysim.csv`, then run the notebooks in order:

1. `01_data_checks.ipynb`: data quality, simulator effects, baselines
2. `02_features_and_split.ipynb`: features and time-based split
3. `03_models.ipynb`: baselines, logistic regression, XGBoost A and B, thresholds
4. `04_final_evaluation.ipynb`: test, subgroups, stress test
5. `05_explanations.ipynb`: SHAP explanations and review priority

## Repository structure

```
notebooks/   analysis notebooks, run in order
src/         feature definitions (features.py)
models/      saved Model B and thresholds
reports/     plots and example review queue
```

## Possible next steps

- Receiving-account history features, built only from earlier transactions.
- Repeat on IEEE-CIS, which uses real (anonymised) e-commerce transactions.
- Serve the model through an API with input validation, tests and Docker.
