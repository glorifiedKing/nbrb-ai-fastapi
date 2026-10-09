from fastapi import FastAPI, Header, HTTPException, status, APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from app.models.schemas import AnalyzeDrawingRequest
from app.services.document_ingestion import RegulatoryIngestionEngine
from app.services.rag_retriever import RAGRetriever
from app.workers.celery_app import process_drawing_compliance
from app.config import settings
import shutil
import os
import tempfile
import uuid

app = FastAPI(
    title="BIMS Uganda AI Compliance Microservice",
    version="1.0.0",
    docs_url="/api/docs"
)

router = APIRouter(prefix="/api/v1/documents", tags=["Regulatory Documents"])

class QueryStandardsRequest(BaseModel):
    query: str
    discipline: str = "architectural"
    limit: int = 5

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "bims-ai-engine",
        "version": "1.0.0"
    }

@app.post("/api/v1/analyze-drawing", status_code=status.HTTP_202_ACCEPTED)
async def trigger_drawing_analysis(
    payload: AnalyzeDrawingRequest,
    x_internal_secret: str = Header(...)
):
    """
    Called by Laravel when a professional uploads a drawing to a floor.
    Dispatches task to Redis/Celery worker or runs synchronous fallback.
    """
    if x_internal_secret != settings.INTERNAL_API_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized internal call")

    task_id = str(uuid.uuid4())
    try:
        task = process_drawing_compliance.delay(payload.model_dump(mode="json"))
        task_id = task.id
    except Exception as e:
        print(f"Celery queue offline, running compliance process in background thread: {e}")
        process_drawing_compliance(payload.model_dump(mode="json"))

    return {
        "status": "QUEUED",
        "task_id": task_id,
        "drawing_version_id": payload.drawing_version_id
    }

@router.post("/ingest-standard", status_code=status.HTTP_201_CREATED)
async def ingest_regulatory_document(
    document_id: str = Form(...),
    file: UploadFile = File(...),
    x_internal_secret: str = Header(...)
):
    """
    Receives statutory PDF from Laravel admin portal, chunks by clause,
    generates 768d vector embeddings, and indexes into PostgreSQL.
    """
    if x_internal_secret != settings.INTERNAL_API_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized internal call")

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        total_chunks = RegulatoryIngestionEngine.ingest_document(
            document_id=document_id,
            pdf_path=temp_path
        )

        return {
            "status": "SUCCESS",
            "document_id": document_id,
            "total_chunks_vectorized": total_chunks,
            "message": f"Regulatory standard parsed, embedded, and indexed ({total_chunks} clauses)."
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/v1/query-standards")
def query_regulatory_standards(
    payload: QueryStandardsRequest,
    x_internal_secret: str = Header(...)
):
    if x_internal_secret != settings.INTERNAL_API_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized internal call")

    results = RAGRetriever.retrieve_relevant_codes(
        query_text=payload.query,
        discipline=payload.discipline,
        limit=payload.limit
    )

    return {"results": results}

app.include_router(router)
