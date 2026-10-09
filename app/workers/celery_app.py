from celery import Celery
import httpx
from app.config import settings
from app.services.vector_measurer import VectorMeasurementEngine
from app.services.rag_retriever import RAGRetriever
from app.services.gemini_auditor import GeminiAuditor

celery = Celery("bims_tasks", broker=settings.REDIS_URL, backend=settings.REDIS_URL)

@celery.task(name="tasks.process_drawing_compliance")
def process_drawing_compliance(drawing_data: dict):
    drawing_version_id = str(drawing_data["drawing_version_id"])
    file_path = drawing_data["file_path"]
    discipline = drawing_data["discipline"]
    occupancy = drawing_data.get("occupancy_class", "Class A (Residential)")

    # 1. Run Pre-flight verification
    is_valid, scale_str, scale_factor, msg = VectorMeasurementEngine.inspect_and_extract_scale(file_path)

    if not is_valid:
        payload = {
            "drawing_version_id": drawing_version_id,
            "overall_status": "REJECTED_PREFLIGHT",
            "preflight_metrics": {"is_valid": False, "reason": msg},
            "summary_metrics": {"pass_count": 0, "fail_count": 1, "warning_count": 0},
            "discrepancies": [{
                "element_id": "GLOBAL_SHEET",
                "room_or_grid": "Sheet Viewport",
                "clause_reference": "Uganda Building Control Regulations - Submission Standards",
                "required_value": "Explicit scale (1:50, 1:100) or graphic bar required",
                "observed_value": "Missing / Unreadable",
                "severity": "ERROR",
                "recommendation": "Re-export blueprint from CAD/Revit with standard scale notation."
            }],
            "extracted_geometry": None
        }
    else:
        # 2. Extract vector measurements
        geometry = VectorMeasurementEngine.extract_door_and_window_vectors(file_path, scale_factor)

        # 3. Retrieve relevant building standards via RAG
        rag_query = f"{discipline} compliance rules for {occupancy} egress, ventilation, room dimensions"
        regulations = RAGRetriever.retrieve_relevant_codes(rag_query, discipline)

        # 4. Perform AI / geometric audit
        audit_result = GeminiAuditor.audit_drawing(
            pdf_path=file_path,
            discipline=discipline,
            occupancy_class=occupancy,
            regulatory_context=regulations,
            measured_vectors=geometry,
            drawing_version_id=drawing_version_id
        )

        payload = audit_result
        payload["drawing_version_id"] = drawing_version_id
        payload["preflight_metrics"] = {"is_valid": True, "scale": scale_str, "scale_factor": scale_factor}
        payload["extracted_geometry"] = geometry

    # 5. Dispatch results back to Laravel Webhook
    headers = {"X-Internal-Secret": settings.INTERNAL_API_SECRET}
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(settings.LARAVEL_WEBHOOK_URL, json=payload, headers=headers)
            print(f"Webhook response status: {resp.status_code}")
    except Exception as e:
        print(f"Failed to post results to Laravel webhook: {e}")

    return {"status": "dispatched", "version_id": drawing_version_id}
