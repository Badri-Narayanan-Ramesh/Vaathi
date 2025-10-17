from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.routers import ingest, pages, qa, study_aids, media
import os
import logging
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backend")

app = FastAPI(
    title="AI Tutor Backend API",
    description="FastAPI wrapper for AI Tutor core module",
    version="1.0.0"
)

# CORS
origins = [os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"🔵 REQUEST START: {request.method} {request.url.path}")
    logger.debug(f"   Headers: {dict(request.headers)}")
    logger.debug(f"   Query params: {dict(request.query_params)}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"✅ REQUEST END: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"❌ REQUEST ERROR: {request.method} {request.url.path} - Time: {process_time:.3f}s")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(e), "traceback": traceback.format_exc()}
        )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"🔥 UNHANDLED EXCEPTION: {request.method} {request.url.path}")
    logger.error(f"   Error: {str(exc)}")
    logger.error(f"   Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": str(exc),
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 AI Tutor Backend API starting up...")
    logger.info(f"   Environment: {os.getenv('ENV', 'development')}")
    logger.info(f"   CORS Origins: {origins}")
    logger.info("   Routers registered: ingest, pages, qa, study_aids, media")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 AI Tutor Backend API shutting down...")

# Health check
@app.get("/", tags=["Health"])
async def root():
    logger.debug("Health check endpoint called")
    return {
        "status": "healthy",
        "service": "AI Tutor Backend API",
        "version": "1.0.0",
        "endpoints": ["/ingest", "/pages", "/qa", "/study", "/media"]
    }

@app.get("/health", tags=["Health"])
async def health_check():
    logger.debug("Health check endpoint called")
    return {"status": "ok"}

# Include routers
logger.info("Registering routers...")
app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
app.include_router(pages.router, prefix="/pages", tags=["Pages"])
app.include_router(qa.router, prefix="/qa", tags=["Q&A"])
app.include_router(study_aids.router, prefix="/study", tags=["Study Aids"])
app.include_router(media.router, prefix="/media", tags=["Media"])
logger.info("All routers registered successfully")
