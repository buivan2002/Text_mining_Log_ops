"""Điểm khởi chạy FastAPI App."""

from fastapi import FastAPI

from src.api.endpoints import router

app = FastAPI(title="Incident Mining Pipeline API")
app.include_router(router, prefix="/api/v1")
