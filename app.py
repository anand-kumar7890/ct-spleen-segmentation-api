"""
CT Spleen Volumetry API — Application Entry Point.

This module exposes the FastAPI application from `app.main` and provides
a direct execution entry point for local development and Docker containers.
"""

import os
import uvicorn
from app.main import app

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
