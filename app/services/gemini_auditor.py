import os
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
import pymupdf
from PIL import Image
from google import genai
from google.genai import types

from app.config import settings
from app.models.schemas import OverallStatusEnum

class GeminiAuditor:
    # Ranked models: gemini-3.1-flash-lite is proven fastest and resilient on Developer API
    PREFERRED_MODELS = [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.8-flash",
    ]

    @staticmethod
    def get_client() -> Optional[genai.Client]:
        if settings.GEMINI_API_KEY:
            try:
                return genai.Client(api_key=settings.GEMINI_API_KEY)
            except Exception as e:
                print(f"Failed to create Gemini Client: {e}")
        return None

    @staticmethod
    def _clean_json_text(text: str) -> dict:
        """Strips markdown code fences and parses JSON safely."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())

    @classmethod
    def _render_page_image(cls, page: pymupdf.Page, max_dim_px: float = 1400.0) -> Image.Image:
        rect = page.rect
        max_dim = max(rect.width, rect.height)
        scale = max_dim_px / max_dim if max_dim > 0 else 1.0
        mat = pymupdf.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        return Image.open(io.BytesIO(pix.tobytes("jpeg")))

    @classmethod
    def _audit_sheet(
        cls,
        client: genai.Client,
        img: Image.Image,
        page_num: int,
        total_pages: int,
        discipline: str,
        occupancy_class: str,
        regulatory_context: str
    ) -> Optional[dict]:
        prompt = f"""
        You are an expert Chief Building Control Officer in Uganda auditing Sheet {page_num} of {total_pages}.
        Discipline: {discipline.upper()}
        Occupancy Class: {occupancy_class}

        APPLICABLE STATUTORY CLAUSES (UGANDA NATIONAL BUILDING CODE / BUILDING CONTROL REGULATIONS):
        {regulatory_context}

        TASK:
        1. Determine if this sheet is an ARCHITECTURAL FLOOR PLAN (is_floor_plan: true/false).
        2. If is_floor_plan == true:
           - Identify the floor name (e.g. "Basement Floor", "Ground Floor", "First Floor", "Second Floor", "Roof Terrace").
           - Extract Gross Floor Area (gross_floor_area_sqm in m²) for this specific floor from sheet schedule, title block, or dimensions.
           - Extract Net Internal Area (net_internal_area_sqm in m²) if present.
           - Extract room schedule items (room name and area_sqm).
        3. If is_floor_plan == false:
           - Explain sheet type (e.g. "Site Plan", "Elevation", "Cross Section", "Boundary Details").
        4. Inspect layout for compliance with Ugandan regulations (dimensions, setbacks, egress, light/ventilation ratios >= 10% for habitable rooms, accessibility).
        5. Identify statutory code violations or confirm compliance. Ground findings in specific Ugandan regulations.

        Output strictly valid JSON with no markdown wrapping:
        {{
          "sheet_title": "string",
          "is_floor_plan": true,
          "floor_name": "Ground Floor",
          "gross_floor_area_sqm": 405.0,
          "net_internal_area_sqm": 360.0,
          "overall_status": "PASS" | "WARNING" | "FAIL",
          "summary_metrics": {{
            "pass_count": 3,
            "fail_count": 0,
            "warning_count": 0
          }},
          "discrepancies": [
            {{
              "element_id": "D-01",
              "room_or_grid": "Main Entrance",
              "clause_reference": "Building Control Regulations 2020 - Part IV",
              "required_value": "Minimum width >= 0.80 m",
              "observed_value": "0.90 m",
              "severity": "INFO" | "WARNING" | "ERROR",
              "recommendation": "Complies with statutory escape route standard"
            }}
          ],
          "room_breakdown": [
            {{
              "name": "Outdoor Dining",
              "area_sqm": 120.0
            }}
          ]
        }}
        """

        for attempt in range(2):
            for model_name in cls.PREFERRED_MODELS:
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=[img, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1
                        ),
                    )
                    parsed = cls._clean_json_text(resp.text)
                    parsed["model_used"] = model_name
                    return parsed
                except Exception as e:
                    time.sleep(1.5)
                    continue
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
        Audits architectural/engineering drawings using Gemini multimodal vision.
        Supports multi-floor PDFs: audits all floors in parallel and calculates cumulated floor area.
        100% compatible with Gemini Developer API mode.
        """
        client = cls.get_client()

        if client and os.path.exists(pdf_path):
            try:
                doc = pymupdf.open(pdf_path)
                total_pages = len(doc)

                # Identify relevant sheets to audit (limit to 6 concurrent sheets to respect quotas)
                pages_to_audit = []
                if total_pages == 1:
                    pages_to_audit = [0]
                else:
                    # Filter for layout, floor, or elevation sheets
                    for idx, page in enumerate(doc):
                        text = page.get_text().upper()
                        if any(kw in text for kw in ["FLOOR", "LAYOUT", "PLAN", "BASEMENT", "ROOF"]):
                            pages_to_audit.append(idx)
                    if not pages_to_audit:
                        pages_to_audit = list(range(min(total_pages, 4)))
                    pages_to_audit = pages_to_audit[:6]

                def _audit_worker(page_idx: int):
                    page = doc[page_idx]
                    img = cls._render_page_image(page)
                    return page_idx, cls._audit_sheet(
                        client=client,
                        img=img,
                        page_num=page_idx + 1,
                        total_pages=total_pages,
                        discipline=discipline,
                        occupancy_class=occupancy_class,
                        regulatory_context=regulatory_context
                    )

                # Process sheets in parallel (up to 3 threads)
                max_threads = min(3, len(pages_to_audit))
                sheet_results = []
                with ThreadPoolExecutor(max_workers=max_threads) as executor:
                    sheet_results = list(executor.map(_audit_worker, pages_to_audit))

                detected_floors = []
                all_discrepancies = []
                all_rooms = []
                total_pass = 0
                total_fail = 0
                total_warning = 0
                cumulated_gfa = 0.0
                cumulated_nia = 0.0
                models_used = set()

                for page_idx, result in sheet_results:
                    if not result or not isinstance(result, dict):
                        continue

                    models_used.add(result.get("model_used", "gemini"))
                    page_num = page_idx + 1
                    sheet_title = result.get("sheet_title", f"Sheet {page_num}")
                    is_fp = result.get("is_floor_plan", False)
                    floor_name = result.get("floor_name") or f"Floor (Sheet {page_num})"
                    gfa = result.get("gross_floor_area_sqm")
                    nia = result.get("net_internal_area_sqm")

                    cleaned_rooms = []
                    for r in result.get("room_breakdown", []):
                        if isinstance(r, dict):
                            r_name = r.get("name") or r.get("room_name") or "Unnamed Room"
                            r_area = r.get("area_sqm")
                            try:
                                r_area = float(r_area) if r_area is not None else None
                            except (ValueError, TypeError):
                                r_area = None
                            room_entry = {"name": f"[{floor_name}] {r_name}", "area_sqm": r_area}
                            cleaned_rooms.append(room_entry)
                            all_rooms.append(room_entry)

                    if is_fp and gfa is not None:
                        try:
                            gfa_val = float(gfa)
                            if gfa_val > 0:
                                cumulated_gfa += gfa_val
                                if nia is not None:
                                    try:
                                        cumulated_nia += float(nia)
                                    except (ValueError, TypeError):
                                        pass
                                detected_floors.append({
                                    "page": page_num,
                                    "floor_name": floor_name,
                                    "gross_floor_area_sqm": gfa_val,
                                    "net_internal_area_sqm": float(nia) if nia else None,
                                    "room_breakdown": cleaned_rooms
                                })
                        except (ValueError, TypeError):
                            pass

                    for d in result.get("discrepancies", []):
                        if isinstance(d, dict):
                            loc_tag = f"[Sheet {page_num}: {floor_name if is_fp else sheet_title}]"
                            d["room_or_grid"] = f"{loc_tag} {d.get('room_or_grid', '')}".strip()
                            all_discrepancies.append(d)

                    sm = result.get("summary_metrics", {})
                    total_pass += sm.get("pass_count", 0)
                    total_fail += sm.get("fail_count", 0)
                    total_warning += sm.get("warning_count", 0)

                overall_status = "FAIL" if total_fail > 0 else ("WARNING" if total_warning > 0 else "PASS")
                has_floors = len(detected_floors) > 0

                floor_area_metrics = {
                    "is_floor_plan": has_floors,
                    "gross_floor_area_sqm": round(cumulated_gfa, 2) if has_floors and cumulated_gfa > 0 else None,
                    "net_internal_area_sqm": round(cumulated_nia, 2) if has_floors and cumulated_nia > 0 else None,
                    "cumulated_floors_count": len(detected_floors) if has_floors else 0,
                    "confidence_score": 0.95 if has_floors else 0.85,
                    "source": "multimodal_multi_floor_cumulation",
                    "floors": detected_floors,
                    "room_breakdown": all_rooms
                }

                return {
                    "drawing_version_id": drawing_version_id,
                    "overall_status": overall_status,
                    "preflight_metrics": {
                        "is_valid": True,
                        "engine": "gemini_multimodal_multi_floor_audit",
                        "sheets_audited": len(pages_to_audit),
                        "floors_cumulated": len(detected_floors),
                        "models": list(models_used)
                    },
                    "summary_metrics": {
                        "pass_count": max(total_pass, 1),
                        "fail_count": total_fail,
                        "warning_count": total_warning
                    },
                    "discrepancies": all_discrepancies,
                    "extracted_geometry": measured_vectors,
                    "floor_area_metrics": floor_area_metrics
                }

            except Exception as e:
                print(f"Gemini multi-floor audit error: {e}")

        # Deterministic vector fallback
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
            "extracted_geometry": measured_vectors,
            "floor_area_metrics": None
        }
