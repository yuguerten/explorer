from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import adapters
from .models import InspectionResult, LoadStatus, LoadWeightsRequest, ResolveRequest
from .service import inspect_model

app = FastAPI(title="Local Transformer Model Explorer")

# Local dev only: allow browser origins during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/families")
def families() -> dict:
    return {"families": adapters.supported_families()}


@app.post("/api/inspect", response_model=InspectionResult)
def inspect(req: ResolveRequest) -> InspectionResult:
    try:
        return inspect_model(req.model_id, req.revision)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inspection failed: {e}")


@app.post("/api/weights", response_model=LoadStatus)
def load_weights(req: LoadWeightsRequest) -> LoadStatus:
    # Placeholder: weight loading is a Phase 4 concern. Return clear not-supported.
    return LoadStatus(
        status="unsupported",
        message="Weight loading is not yet implemented in this build.",
        detail="The config-only inspection path is available.",
    )
