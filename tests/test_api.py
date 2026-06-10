"""
API tests for the D2C Churn Scoring Service.
Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ── Sample customer payload ──────────────────────────────────────────────────
SAMPLE_CUSTOMER = {
    "recency_days": 45,
    "frequency_180d": 5,
    "monetary_180d": 3200.0,
    "return_rate_180d": 0.1,
    "avg_discount_pct_180d": 0.15,
    "avg_rating_180d": 4.2,
    "category_diversity_180d": 3,
    "ticket_count_90d": 1,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 12.0,
    "days_since_signup": 300,
    "sessions_30d": 8,
    "product_views_30d": 20,
    "cart_adds_30d": 3,
    "wishlist_adds_30d": 1,
    "abandoned_carts_30d": 1,
    "email_opens_30d": 5,
    "campaign_clicks_30d": 2,
    "last_visit_days_ago": 3,
    "city_tier": "Tier 1",
    "age_group": "25-34",
    "acquisition_channel": "Google Search",
    "loyalty_tier": "Silver",
    "preferred_category": "Skin Care",
    "marketing_consent": "Yes",
}

HIGH_RISK_CUSTOMER = {
    "recency_days": 150,
    "frequency_180d": 1,
    "monetary_180d": 400.0,
    "return_rate_180d": 0.5,
    "avg_discount_pct_180d": 0.4,
    "avg_rating_180d": 2.0,
    "category_diversity_180d": 1,
    "ticket_count_90d": 3,
    "negative_ticket_rate_90d": 0.8,
    "avg_resolution_hours_90d": 48.0,
    "days_since_signup": 200,
    "sessions_30d": 0,
    "product_views_30d": 0,
    "cart_adds_30d": 0,
    "wishlist_adds_30d": 0,
    "abandoned_carts_30d": 0,
    "email_opens_30d": 0,
    "campaign_clicks_30d": 0,
    "last_visit_days_ago": 60,
    "city_tier": "Tier 3",
    "age_group": "45+",
    "acquisition_channel": "Marketplace",
    "loyalty_tier": None,
    "preferred_category": "Baby Care",
    "marketing_consent": "No",
}


# ── Test 1: Health check ─────────────────────────────────────────────────────
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "threshold" in data


# ── Test 2: Single prediction ────────────────────────────────────────────────
def test_predict_single():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    # Accept 200 (model loaded) or 503 (model not loaded in test env)
    if response.status_code == 200:
        data = response.json()
        assert "churn_probability" in data
        assert "predicted_class" in data
        assert "risk_level" in data
        assert "risk_explanation" in data
        assert 0 <= data["churn_probability"] <= 1
        assert data["predicted_class"] in [0, 1]
        assert data["risk_level"] in ["low", "medium", "high"]
    else:
        assert response.status_code == 503


# ── Test 3: Batch prediction ─────────────────────────────────────────────────
def test_batch_predict():
    payload = {"customers": [SAMPLE_CUSTOMER, HIGH_RISK_CUSTOMER]}
    response = client.post("/batch_predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert data["total_customers"] == 2
        assert len(data["predictions"]) == 2
        assert "high_risk_count" in data
    else:
        assert response.status_code == 503


# ── Test 4: Input validation — invalid city_tier ─────────────────────────────
def test_invalid_city_tier():
    bad_payload = SAMPLE_CUSTOMER.copy()
    bad_payload["city_tier"] = "Tier 99"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ── Test 5: Input validation — negative recency ─────────────────────────────
def test_negative_recency():
    bad_payload = SAMPLE_CUSTOMER.copy()
    bad_payload["recency_days"] = -5
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ── Test 6: Empty batch ─────────────────────────────────────────────────────
def test_empty_batch():
    response = client.post("/batch_predict", json={"customers": []})
    assert response.status_code == 400
