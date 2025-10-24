from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="vectorizer-service")

class TextIn(BaseModel):
    text: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/vectorize")
async def vectorize(payload: TextIn):
    # Placeholder implementation: return a dummy vector
    text = payload.text or ""
    vec = [0.0] * 8
    return {"vector": vec, "length": len(text)}

