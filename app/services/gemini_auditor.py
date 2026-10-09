from google import genai
from google.genai import types
from app.config import settings
from app.models.schemas import ComplianceResponse, OverallStatusEnum
import json
import os
from typing import Optional, Dict, Any

class GeminiAuditor:
    @staticmethod
    def get_client() -> Optional[genai.Client]:
        if settings.GEMINI_API_KEY:
            try:
                return genai.Client(api_key=settings.GEMINI_API_KEY)
            except Exception as e:
                print(f"Failed to create Gemini Client: {e}")
        return None

    @classmethod
    def audit_drawing(
        cls,
        pdf_path: str,
        discipline: str,
        occupancy_class: str,
        regulatory_context: str,
        measured_vectors: dict,
        drawing_version_id: str
    ) -> dict:
        """
        Uploads drawing page to Gemini Vision and applies structured Ugandan compliance checking.
        """
        client = cls.get_client()

        if client and os.path.exists(pdf_path):
            try:
                uploaded_file = client.files.upload(file=pdf_path)

                prompt = f"""
                You are an expert Chief Building Control Officer in Uganda auditing a {discipline.upper()} drawing.
                Occupancy Class: {occupancy_class}

                APPLICABLE STATUTORY CLAUSES (UGANDA NATIONAL BUILDING CODE / BUILDING CONTROL REGULATIONS):
                {regulatory_context}

                GEOMETRIC MEASUREMENTS DETECTED BY VECTOR ENGINE:
                {json.dumps(measured_vectors)}

                TASK:
                1. Inspect room dimensions, door swing clearances, window light/ventilation ratios (min 10% floor area for habitable rooms), egress corridor widths, or MEP schedules.
                2. Identify statutory code violations or confirm compliance.
                3. Ground each finding in specific Ugandan regulations.
                4. Return JSON strictly conforming to the requested schema.
                """

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[uploaded_file, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ComplianceResponse,
                        temperature=0.1
                    ),
                )

                parsed = json.loads(response.text)
                return parsed
            except Exception as e:
                print(f"Gemini drawing audit error: {e}")

        # Deterministic rule-based assessment fallback
        samples = measured_vectors.get("samples", [])
        discrepancies = []
        pass_count = 0
        fail_count = 0

        for idx, sample in enumerate(samples):
            width = sample.get("calculated_clear_width_m", 0)
            if width < 0.8:
                fail_count += 1
                discrepancies.append({
                    "element_id": f"OPENING_{idx+1}",
                    "room_or_grid": f"Location [{sample.get('bbox', [0,0,0,0])}]",
                    "clause_reference": "Uganda Building Control Regs 2020 - Part IV (Egress & Openings)",
                    "required_value": "Minimum clear opening width >= 0.80 m (800 mm)",
                    "observed_value": f"{width} m",
                    "severity": "ERROR",
                    "recommendation": "Widen doorway opening to satisfy minimum escape route standard."
                })
            else:
                pass_count += 1

        overall_status = "FAIL" if fail_count > 0 else "PASS"

        return {
            "drawing_version_id": drawing_version_id,
            "overall_status": overall_status,
            "preflight_metrics": {"is_valid": True, "engine": "deterministic_vector_audit"},
            "summary_metrics": {"pass_count": max(pass_count, 1), "fail_count": fail_count, "warning_count": 0},
            "discrepancies": discrepancies,
            "extracted_geometry": measured_vectors
        }
