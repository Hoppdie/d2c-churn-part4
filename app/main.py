"""
D2C Churn Scoring Service — FastAPI Application
Part 4 of D2C Customer Churn Intelligence Capstone
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
import os

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="D2C Churn Scoring Service",
    description="Internal API for predicting customer churn risk. Returns churn probability, predicted class, and risk explanation.",
    version="1.0.0",
)

# ── Load model and artifacts ─────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
FEATURE_COLS_PATH = os.getenv("FEATURE_COLS_PATH", "feature_cols.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", "0.40"))  # default; update after Part 3 run

model = None
feature_cols = None

@app.on_event("startup")
def load_model():
    global model, feature_cols
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: {MODEL_PATH} not found. Run train_model.py first.")
        return
    model = joblib.load(MODEL_PATH)
    if os.path.exists(FEATURE_COLS_PATH):
        feature_cols = joblib.load(FEATURE_COLS_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Threshold: {THRESHOLD}")


# ── Pydantic models ──────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    """Input schema for a single customer prediction."""
    recency_days: int = Field(..., ge=0, description="Days since last order")
    frequency_180d: int = Field(..., ge=0, description="Orders in last 180 days")
    monetary_180d: float = Field(..., ge=0, description="Total spend (INR) in last 180 days")
    return_rate_180d: float = Field(..., ge=0, le=1, description="Return rate (0-1)")
    avg_discount_pct_180d: float = Field(..., ge=0, le=1, description="Avg discount fraction")
    avg_rating_180d: Optional[float] = Field(None, ge=1, le=5, description="Avg order rating (1-5)")
    category_diversity_180d: int = Field(..., ge=0, description="Distinct categories purchased")
    ticket_count_90d: int = Field(..., ge=0, description="Support tickets in last 90 days")
    negative_ticket_rate_90d: float = Field(..., ge=0, le=1, description="Negative ticket proportion")
    avg_resolution_hours_90d: float = Field(..., ge=0, description="Avg ticket resolution hours")
    days_since_signup: int = Field(..., ge=0, description="Days since customer signup")
    sessions_30d: int = Field(..., ge=0, description="Web sessions in last 30 days")
    product_views_30d: int = Field(..., ge=0, description="Product pages viewed in last 30 days")
    cart_adds_30d: int = Field(..., ge=0, description="Cart additions in last 30 days")
    wishlist_adds_30d: int = Field(..., ge=0, description="Wishlist additions in last 30 days")
    abandoned_carts_30d: int = Field(..., ge=0, description="Abandoned carts in last 30 days")
    email_opens_30d: int = Field(..., ge=0, description="Marketing emails opened in last 30 days")
    campaign_clicks_30d: int = Field(..., ge=0, description="Campaign link clicks in last 30 days")
    last_visit_days_ago: int = Field(..., ge=0, description="Days since last website visit")

    # Categorical fields
    city_tier: str = Field(..., description="City tier: Tier 1, Tier 2, Tier 3")
    age_group: str = Field(..., description="Age group: 18-24, 25-34, 35-44, 45+")
    acquisition_channel: str = Field(..., description="Acquisition channel")
    loyalty_tier: Optional[str] = Field(None, description="Loyalty tier or null if not enrolled")
    preferred_category: str = Field(..., description="Preferred product category")
    marketing_consent: str = Field(..., description="Marketing consent: Yes/No")

    @validator('city_tier')
    def validate_city_tier(cls, v):
        valid = ['Tier 1', 'Tier 2', 'Tier 3']
        if v not in valid:
            raise ValueError(f'city_tier must be one of {valid}')
        return v

    @validator('age_group')
    def validate_age_group(cls, v):
        valid = ['18-24', '25-34', '35-44', '45+']
        if v not in valid:
            raise ValueError(f'age_group must be one of {valid}')
        return v

    @validator('marketing_consent')
    def validate_consent(cls, v):
        valid = ['Yes', 'No']
        if v not in valid:
            raise ValueError(f'marketing_consent must be one of {valid}')
        return v


class PredictionResponse(BaseModel):
    churn_probability: float
    predicted_class: int
    risk_level: str
    risk_explanation: str


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold: float


# ── Helper functions ─────────────────────────────────────────────────────────
def build_feature_df(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert Pydantic model to a feature DataFrame matching training schema."""
    data = customer.dict()

    # Build a single-row DataFrame
    row = {}
    for key, val in data.items():
        if key in ['city_tier', 'age_group', 'acquisition_channel',
                    'loyalty_tier', 'preferred_category', 'marketing_consent']:
            continue  # handle categoricals separately
        row[key] = val

    # Fill missing avg_rating with median (3.0)
    if row.get('avg_rating_180d') is None:
        row['avg_rating_180d'] = 3.0

    df = pd.DataFrame([row])

    # One-hot encode categoricals to match training
    cat_data = {
        'city_tier': data['city_tier'],
        'age_group': data['age_group'],
        'acquisition_channel': data['acquisition_channel'],
        'loyalty_tier': data.get('loyalty_tier', np.nan),
        'preferred_category': data['preferred_category'],
        'marketing_consent': data['marketing_consent'],
    }
    cat_df = pd.DataFrame([cat_data])
    cat_encoded = pd.get_dummies(cat_df, drop_first=True, dummy_na=True)

    df = pd.concat([df, cat_encoded], axis=1)

    # Align columns with training feature set
    if feature_cols is not None:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]

    return df


def get_risk_explanation(customer: CustomerFeatures, prob: float) -> str:
    """Generate a human-readable risk explanation."""
    reasons = []

    if customer.recency_days > 90:
        reasons.append(f"inactive for {customer.recency_days} days")
    if customer.sessions_30d == 0:
        reasons.append("zero web sessions in last 30 days")
    if customer.ticket_count_90d >= 2:
        reasons.append(f"{customer.ticket_count_90d} support tickets recently")
    if customer.negative_ticket_rate_90d > 0.5:
        reasons.append("majority negative support interactions")
    if customer.frequency_180d <= 1:
        reasons.append("very low purchase frequency")
    if customer.last_visit_days_ago > 30:
        reasons.append(f"last site visit {customer.last_visit_days_ago} days ago")
    if customer.return_rate_180d > 0.3:
        reasons.append(f"high return rate ({customer.return_rate_180d:.0%})")

    if not reasons:
        if prob >= THRESHOLD:
            return "Multiple minor risk factors combine to elevate churn risk."
        else:
            return "Customer shows healthy engagement across key metrics."

    reason_str = ", ".join(reasons)
    if prob >= THRESHOLD:
        return f"Elevated churn risk due to: {reason_str}."
    else:
        return f"Low churn risk despite some concerns: {reason_str}."


def classify_risk(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    elif prob >= THRESHOLD:
        return "medium"
    else:
        return "low"


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        threshold=THRESHOLD,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_single(customer: CustomerFeatures):
    """Predict churn risk for a single customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    try:
        df = build_feature_df(customer)
        prob = float(model.predict_proba(df)[:, 1][0])
        predicted_class = 1 if prob >= THRESHOLD else 0
        risk_level = classify_risk(prob)
        explanation = get_risk_explanation(customer, prob)

        return PredictionResponse(
            churn_probability=round(prob, 4),
            predicted_class=predicted_class,
            risk_level=risk_level,
            risk_explanation=explanation,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """Predict churn risk for multiple customers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    if len(request.customers) == 0:
        raise HTTPException(status_code=400, detail="Empty customer list.")

    if len(request.customers) > 500:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 500.")

    predictions = []
    for customer in request.customers:
        try:
            df = build_feature_df(customer)
            prob = float(model.predict_proba(df)[:, 1][0])
            predicted_class = 1 if prob >= THRESHOLD else 0
            risk_level = classify_risk(prob)
            explanation = get_risk_explanation(customer, prob)
            predictions.append(PredictionResponse(
                churn_probability=round(prob, 4),
                predicted_class=predicted_class,
                risk_level=risk_level,
                risk_explanation=explanation,
            ))
        except Exception as e:
            predictions.append(PredictionResponse(
                churn_probability=-1.0,
                predicted_class=-1,
                risk_level="error",
                risk_explanation=f"Prediction failed: {str(e)}",
            ))

    high_risk = sum(1 for p in predictions if p.risk_level in ["high", "medium"])

    return BatchResponse(
        predictions=predictions,
        total_customers=len(predictions),
        high_risk_count=high_risk,
    )
