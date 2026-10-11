"""Route requests to separately running model APIs; never load models here."""
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

MODEL_URLS = {
    "qlora": os.getenv("QLORA_API_URL", "http://127.0.0.1:8000").rstrip("/"),
    "lora": os.getenv("LORA_API_URL", "http://127.0.0.1:8001").rstrip("/"),
}


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        app.state.client = client
        yield


app = FastAPI(title="Math Model Gateway", lifespan=lifespan)
origins = [value.strip() for value in os.getenv("UI_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if value.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins,
    allow_methods=["GET", "POST"], allow_headers=["Content-Type", "ngrok-skip-browser-warning"],
)


class MathRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8192)
    max_new_tokens: int = Field(default=150, ge=1, le=512)
    temperature: float = Field(default=0.7, gt=0, le=2)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value


@app.post("/{method}/solve")
async def solve(method: str, req: MathRequest):
    if method not in MODEL_URLS:
        raise HTTPException(404, "Unknown model method")
    try:
        response = await app.state.client.post(
            MODEL_URLS[method] + "/solve", json=req.model_dump())
    except httpx.TimeoutException:
        raise HTTPException(504, "Model response timed out") from None
    except httpx.RequestError:
        raise HTTPException(502, "Model server unreachable") from None
    if response.status_code != 200:
        status = response.status_code if response.status_code in (422, 500, 503) else 502
        raise HTTPException(status, "Model server request failed")
    try:
        data = response.json()
    except ValueError:
        raise HTTPException(502, "Invalid model server response") from None
    if not isinstance(data, dict) or data.get("success") is not True or not isinstance(data.get("answer"), str):
        raise HTTPException(502, "Invalid model server response")
    return {"success": True, "method": method, "question": req.question, "answer": data["answer"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gateway:app", host="127.0.0.1", port=8080)
