from fastapi import FastAPI
from app.routes.detect import router as detect_router

app = FastAPI(title="Object Detection API")

app.include_router(detect_router)

@app.get("/")
def root():
    return {"message": "API is running"}

