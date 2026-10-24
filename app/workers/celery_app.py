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
    supplied_discipline = drawing_data.get("discipline")
    occupancy = drawing_data.get("occupancy_class", "Class A (Residential)")

    # 1. Detect & resolve drawing discipline and drawing type
    classification = VectorMeasurementEngine.detect_drawing_discipline_and_type(
        file_path, supplied_discipline=supplied_discipline
    )
    resolved_discipline = classification["resolved_discipline"]
    drawing_type = classification["drawing_type"]

    # 2. Run Pre-flight verification
    is_valid, scale_str, scale_factor, msg = VectorMeasurementEngine.inspect_and_extract_scale(file_path)

    if not is_valid:
        payload = {
            "drawing_version_id": drawing_version_id,
            "discipline": resolved_discipline,
            "drawing_type": drawing_type,
            "overall_status": "REJECTED_PREFLIGHT",
            "preflight_metrics": {
                "is_valid": False, 
                "reason": msg,
                "detected_discipline": classification["detected_discipline"],
                "drawing_type": drawing_type
            },
            "summary_metrics": {"pass_count": 0, "fail_count": 1, "warning_count": 0},
            "discrepancies": [{
                "element_id": f"{resolved_discipline.upper()}_SHEET",
                "room_or_grid": "Sheet Viewport",
                "clause_reference": "Uganda Building Control Regulations - Submission Standards",
                "required_value": "Explicit scale (1:50, 1:100, 1:200) or graphic bar / schematic notation required",
                "observed_value": "Missing / Unreadable",
                "severity": "ERROR",
                "recommendation": "Re-export blueprint from CAD/Revit with standard scale notation."
            }],
            "extracted_geometry": None
        }
    else:
        # 3. Extract vector measurements & floor area
        geometry = VectorMeasurementEngine.extract_door_and_window_vectors(file_path, scale_factor)
        floor_area_data = VectorMeasurementEngine.extract_floor_area_from_sheet(file_path, scale_factor)

        # 4. Formulate discipline-specific statutory RAG query
        if resolved_discipline == "structural":
            rag_query = f"Structural design compliance for {occupancy}: column beam sizing, concrete strength grade, rebar cover, foundation design, live dead loads, structural stability"
        elif resolved_discipline == "electrical":
            rag_query = f"Electrical installations compliance for {occupancy}: earthing grounding, circuit protection MCB RCD 30mA, cable sizing, illumination lux levels, socket height, emergency lighting"
        elif resolved_discipline == "mechanical":
            rag_query = f"Mechanical plumbing compliance for {occupancy}: foul drainage pipe slope gradient, soil vent pipes, water trap seals, grease traps, HVAC ventilation air changes, fire dampers"
        else:
            rag_query = f"Architectural design compliance for {occupancy}: minimum room sizes, ceiling height, window natural light ventilation 10% ratio, doorway corridor egress width, accessibility"

        # 5. Retrieve relevant building standards via RAG (matching discipline first, then complementary)
        regulations = RAGRetriever.retrieve_relevant_codes(rag_query, resolved_discipline)

        # 6. Perform AI / geometric audit
        audit_result = GeminiAuditor.audit_drawing(
            pdf_path=file_path,
            discipline=resolved_discipline,
            drawing_type=drawing_type,
            occupancy_class=occupancy,
            regulatory_context=regulations,
            measured_vectors=geometry,
            drawing_version_id=drawing_version_id
        )

        payload = audit_result
        payload["drawing_version_id"] = drawing_version_id
        payload["discipline"] = resolved_discipline
        payload["drawing_type"] = drawing_type
        payload["classification"] = classification
        payload["preflight_metrics"]["scale"] = scale_str
        payload["preflight_metrics"]["scale_factor"] = scale_factor
        payload["preflight_metrics"]["detected_discipline"] = classification["detected_discipline"]
        payload["preflight_metrics"]["drawing_type"] = drawing_type
        payload["extracted_geometry"] = geometry

        # If Gemini returned floor_area_metrics with valid GFA, preserve it; otherwise use vector/OCR extracted metrics
        current_fam = payload.get("floor_area_metrics")
        if not current_fam or not isinstance(current_fam, dict) or not current_fam.get("gross_floor_area_sqm"):
            if resolved_discipline == "architectural" or floor_area_data.get("is_floor_plan"):
                payload["floor_area_metrics"] = floor_area_data

        active_fam = payload.get("floor_area_metrics")
        if active_fam and isinstance(active_fam, dict):
            payload["gross_floor_area_sqm"] = active_fam.get("gross_floor_area_sqm")
            payload["net_internal_area_sqm"] = active_fam.get("net_internal_area_sqm")

    # 5. Dispatch results back to Laravel Webhook
    headers = {"X-Internal-Secret": settings.INTERNAL_API_SECRET}
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(settings.LARAVEL_WEBHOOK_URL, json=payload, headers=headers)
            print(f"Webhook response status: {resp.status_code}")
    except Exception as e:
        print(f"Failed to post results to Laravel webhook: {e}")

    return {"status": "dispatched", "version_id": drawing_version_id}
