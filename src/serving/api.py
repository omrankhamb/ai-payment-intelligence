"""FastAPI Real-Time Payment Fraud Intelligence Service.

Endpoints:
- POST /score: Synchronous inference, redis velocity lookup, GNN scoring, and explanation.
- GET /health: Liveness and readiness probe with model/cache health checks.
- GET /metrics: Prometheus scraper endpoint.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.features import get_card_key
from src.serving.inference import FraudScorer
from src.serving.metrics import (
    DECISION_TOTAL,
    FRAUD_SCORE_HIST,
    LATENCY_HIST,
    drift_tracker,
)
from src.serving.redis_store import RedisVelocityStore

logger = logging.getLogger("fraudgraph.api")
logging.basicConfig(level=logging.INFO)

scorer: Optional[FraudScorer] = None
redis_store: Optional[RedisVelocityStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer, redis_store
    logger.info("Initializing FraudGraph Scorer & Redis Store...")
    try:
        scorer = FraudScorer(artifacts_dir="artifacts")
        redis_store = RedisVelocityStore()
        logger.info("FraudGraph engine ready.")
    except Exception as e:
        logger.error(f"Failed to initialize FraudScorer: {e}")
        raise e
    yield
    logger.info("Shutting down FraudGraph API service.")


app = FastAPI(
    title="FraudGraph Payment Intelligence API",
    description="Real-time graph-augmented payment fraud prevention system",
    version="1.0.0",
    lifespan=lifespan,
)


class ScoreRequest(BaseModel):
    TransactionID: int = Field(..., description="Unique transaction ID", example=3000001)
    TransactionDT: int = Field(..., description="Transaction timestamp in seconds", example=86400)
    TransactionAmt: float = Field(..., description="Payment transaction amount in USD", example=150.00)
    ProductCD: str = Field(..., description="Product code category", example="W")
    card1: Any = Field(..., description="Primary card ID", example=13926)
    addr1: Optional[float] = Field(None, description="Billing address code", example=315.0)
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None
    features: Optional[List[float]] = Field(None, description="Optional 437-dim feature vector")


class ReasonDetail(BaseModel):
    factor: str
    weight: float
    edge_type: str
    description: str


class ScoreResponse(BaseModel):
    transaction_id: int
    fraud_score: float
    decision: str
    thresholds: Dict[str, float]
    top_reasons: List[ReasonDetail]
    latency_ms: float


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def serve_dashboard():
    """Interactive visual demo UI for real-time payment fraud intelligence."""
    html_file = Path(__file__).parent / "dashboard.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>FraudGraph API Running. Visit <a href='/docs'>/docs</a> for Swagger</h1>")


@app.get("/health", tags=["System"])
def health_check():
    """Health and readiness probe."""
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model scorer not initialized")
    return {
        "status": "healthy",
        "service": "fraudgraph-scorer",
        "version": "1.0.0",
        "redis_connected": redis_store.client is not None if redis_store else False,
    }


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_payment(request: ScoreRequest):
    """Score a single payment in real time."""
    if scorer is None or redis_store is None:
        raise HTTPException(status_code=503, detail="Service initializing")

    t_start = time.perf_counter()
    payload = request.model_dump()

    try:
        # 1. Redis sliding-window lookup
        card_key = get_card_key(payload)
        velocity_info = redis_store.record_and_get_velocity(
            card_key=card_key or "",
            transaction_id=request.TransactionID,
            transaction_dt=request.TransactionDT,
            amount=request.TransactionAmt,
        )

        # 2. Score transaction with GNN & Explainer
        result = scorer.score_transaction(payload, velocity_info=velocity_info)

        # 3. Update Prometheus metrics & Drift tracker
        dur_sec = time.perf_counter() - t_start
        LATENCY_HIST.observe(dur_sec)
        FRAUD_SCORE_HIST.observe(result["fraud_score"])
        DECISION_TOTAL.labels(decision=result["decision"]).inc()
        drift_tracker.record_transaction(request.TransactionAmt)

        return result
    except Exception as e:
        logger.error(f"Inference error on tx {request.TransactionID}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
