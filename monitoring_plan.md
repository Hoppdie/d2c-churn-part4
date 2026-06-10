# Monitoring Plan & Responsible Use
## D2C Churn Scoring Service — Part 4

---

## 1. Monitoring Plan

### 1.1 Data Drift Monitoring

| Signal | What to Track | Method | Frequency | Alert Threshold |
|---|---|---|---|---|
| Feature distributions | Mean/std of `recency_days`, `frequency_180d`, `monetary_180d`, `sessions_30d` | Population Stability Index (PSI) | Weekly | PSI > 0.2 on any key feature |
| Categorical shifts | Distribution of `city_tier`, `acquisition_channel`, `loyalty_tier` | Chi-square test vs training distribution | Weekly | p-value < 0.01 |
| Missing value rate | % nulls in `loyalty_tier`, `avg_rating_180d` | Count comparison | Weekly | >10% increase from training baseline |

**Why it matters:** If the customer base changes (e.g., a new marketing channel brings a different demographic), the model's assumptions break. Feature drift is often the earliest warning sign of model degradation.

### 1.2 Prediction Distribution Monitoring

| Metric | What to Track | Frequency | Alert Threshold |
|---|---|---|---|
| Mean churn probability | Average predicted probability across all scored customers | Daily | >10% shift from training mean |
| High-risk rate | % of customers classified as high-risk (prob > 0.7) | Daily | >15% absolute deviation from baseline |
| Score distribution | Histogram shape of predicted probabilities | Weekly | Bimodal collapse or extreme skew |

**Why it matters:** If the model suddenly predicts everyone as high-risk (or low-risk), something has gone wrong — either the input data is corrupted or the model has drifted.

### 1.3 Business Outcome Monitoring

| Metric | What to Track | Frequency | Action |
|---|---|---|---|
| Actual churn rate | Compare predicted churn rate vs actual (60 days later) | Monthly | If actual deviates by >5pp from predicted, investigate |
| Retention campaign ROI | For customers flagged as high-risk who received outreach: did they purchase? | Monthly | Calculate lift vs control group |
| False negative rate | Among customers who churned, what % were scored as low-risk? | Monthly | If >30%, model is missing too many churners |
| False positive cost | Total campaign spend on customers who would have stayed anyway | Monthly | Track as a cost metric; acceptable if < total budget × 40% |

### 1.4 API Health Monitoring

| Metric | What to Track | Frequency | Alert Threshold |
|---|---|---|---|
| Response time | p50, p95, p99 latency of `/predict` and `/batch_predict` | Continuous | p95 > 500ms |
| Error rate | HTTP 4xx and 5xx responses | Continuous | >2% of requests |
| Uptime | API availability | Continuous | <99.5% over 7 days |
| Throughput | Requests per minute | Continuous | Unexpected spikes (>10x normal) |

### 1.5 Retraining Triggers

The model should be retrained when any of the following occur:

1. **Scheduled:** Every 90 days as a baseline cadence
2. **Drift detected:** PSI > 0.2 on two or more key features
3. **Performance decay:** Monthly actual-vs-predicted churn deviation > 5 percentage points for two consecutive months
4. **Business change:** Major product launch, pricing restructure, new acquisition channel, or significant policy change (e.g., new return policy)
5. **Data pipeline change:** If upstream data sources change schema, frequency, or coverage

---

## 2. Responsible Use Guidelines

### How the API Output SHOULD Be Used

- As **one input** into a retention decision, combined with CRM team judgement
- To **prioritise** which customers to contact first, not to determine the message content
- To **flag** customers for human review when the score is near the decision boundary (probability between 0.35–0.55)
- To **measure** retention campaign effectiveness by comparing outcomes for scored vs unscored cohorts
- To **inform** budget allocation across customer segments

### How the API Output Should NOT Be Used

- **Not for automated exclusion.** A low churn score does not mean the customer should be excluded from all marketing. Loyal customers still benefit from engagement.
- **Not as the sole decision-maker** for high-value customers. Any customer with monetary_180d in the top 10% should receive manual review regardless of model score.
- **Not for customer-facing communication.** Never tell a customer their "churn risk score." The score is an internal operational tool.
- **Not for discriminatory targeting.** Do not use the score to disproportionately target or ignore customers based on demographics (city tier, age group). Monitor prediction rates across demographic groups.
- **Not for punitive action.** The score should never be used to reduce service quality, increase prices, or remove benefits for high-risk customers. The goal is retention, not punishment.
- **Not as a permanent label.** A customer flagged as high-risk today may re-engage tomorrow. Scores should be refreshed regularly and old scores discarded.

### Data Privacy

- The API processes customer behavioural data. Ensure compliance with applicable data protection regulations.
- Input payloads should be transmitted over HTTPS only.
- API logs should not store full customer payloads beyond what is needed for debugging (retain for 30 days maximum).
- Customer IDs in logs should be pseudonymised in any external reporting.

---

## 3. Incident Response

| Scenario | Response |
|---|---|
| Model returns all-zero or all-one predictions | Halt scoring, revert to previous model version, investigate input data |
| API latency exceeds 2 seconds | Scale horizontally or investigate database/model loading issues |
| Actual churn rate diverges >10pp from predicted | Pause campaign decisions based on scores, trigger emergency retraining |
| Data pipeline delivers stale data (>48h old) | Alert data engineering team, halt new scoring until fresh data arrives |
