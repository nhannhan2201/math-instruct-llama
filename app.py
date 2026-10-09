"""Local API; choose a verified artifact with MATH_ADAPTER_PATH."""
import logging
import os
from contextlib import asynccontextmanager
from threading import Lock

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from src.inference import MathSolver

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    app.state.solver = None
    app.state.generation_lock = Lock()
    try:
        path = os.environ.get("MATH_ADAPTER_PATH")
        if not path:
            raise ValueError("Set MATH_ADAPTER_PATH to a verified adapter artifact")
        app.state.solver = MathSolver(method=os.environ.get("MATH_METHOD", "qlora"), adapter_path=path)
    except Exception:
        logger.exception("Model initialization failed")
    yield
    app.state.solver = None


app = FastAPI(title="Math Instruct Llama API", lifespan=lifespan)


class MathRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8192)
    max_new_tokens: int = Field(default=150, ge=1, le=512)
    temperature: float = Field(default=0.7, gt=0, le=2)

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value


@app.post("/solve")
def solve_math(req: MathRequest):
    solver = getattr(app.state, "solver", None)
    if solver is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    try:
        with app.state.generation_lock:
            answer = solver.solve(question=req.question, max_new_tokens=req.max_new_tokens,
                                  temperature=req.temperature)
    except Exception:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail="Generation failed") from None
    return {"success": True, "question": req.question, "answer": answer}


@app.get("/")
def health_check():
    if getattr(app.state, "solver", None) is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    return {"status": "ready"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
