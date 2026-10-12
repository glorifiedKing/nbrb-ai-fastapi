import fitz  # PyMuPDF
import re
from typing import Tuple, Optional, Dict, Any

class VectorMeasurementEngine:
    @staticmethod
    def inspect_and_extract_scale(pdf_path: str) -> Tuple[bool, Optional[str], Optional[float], str]:
        """
        Validates page readability and detects numerical scale or graphic ratios.
        Returns: (is_valid, scale_string, scale_factor_m_per_pt, message)
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            return False, None, None, f"Unable to open PDF document: {str(e)}"

        if len(doc) == 0:
            return False, None, None, "Document contains no pages."

        combined_text = ""
        for p_idx in range(min(3, len(doc))):
            combined_text += " " + doc[p_idx].get_text()

        scale_patterns = [
            (r"1\s*:\s*100", 0.03528, "1:100"),
            (r"1\s*:\s*50", 0.01764, "1:50"),
            (r"1\s*:\s*200", 0.07056, "1:200"),
            (r"1\s*:\s*20", 0.007056, "1:20"),
            (r"1\s*:\s*25", 0.00882, "1:25"),
            (r"1\s*:\s*500", 0.1764, "1:500"),
            (r"1/4\"\s*=\s*1'-0\"", 0.03528, "1/4\" = 1'-0\""),
            (r"SCALE\s*[:=\-]?\s*1\s*/\s*100", 0.03528, "1:100"),
            (r"SCALE\s*[:=\-]?\s*1\s*/\s*50", 0.01764, "1:50"),
        ]

        for pattern, factor, standard_name in scale_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True, standard_name, factor, f"Scale {standard_name} verified successfully."

        first_page = doc[0]
        drawings = first_page.get_drawings()
        if len(drawings) > 10:
            return True, "1:100 (Inferred)", 0.03528, "Inferred default scale 1:100 based on vector density."

        return False, None, None, "No standard scale callout (e.g. 1:100, 1:50) detected in drawing."

    @staticmethod
    def extract_door_and_window_vectors(pdf_path: str, scale_factor: float) -> Dict[str, Any]:
        """
        Extracts vector line openings to calculate physical clear opening dimensions.
        """
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return {"openings_detected_count": 0, "samples": []}

            page = doc[0]
            drawings = page.get_drawings()

            potential_openings = []
            for d in drawings:
                rect = d.get("rect")
                if not rect:
                    continue
                width_pts = rect.width
                height_pts = rect.height

                if (15 < width_pts < 300 and height_pts < 25) or (15 < height_pts < 300 and width_pts < 25):
                    dimension_pts = max(width_pts, height_pts)
                    real_width_meters = dimension_pts * scale_factor
                    if 0.6 <= real_width_meters <= 3.5:
                        potential_openings.append({
                            "bbox": [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)],
                            "calculated_clear_width_m": round(real_width_meters, 3)
                        })

            return {
                "openings_detected_count": len(potential_openings),
                "samples": potential_openings[:10]
            }
        except Exception as e:
            return {"openings_detected_count": 0, "samples": [], "error": str(e)}

    @staticmethod
    def extract_floor_area_from_sheet(pdf_path: str, scale_factor: float = 0.03528) -> Dict[str, Any]:
        """
        Parses architectural floor plan sheets for Gross Floor Area (GFA)
        via title block/schedule regex, room tables, or vector boundary estimation.
        """
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return {
                    "is_floor_plan": False,
                    "gross_floor_area_sqm": None,
                    "net_internal_area_sqm": None,
                    "confidence_score": 0.0,
                    "source": "none",
                    "room_breakdown": []
                }

            combined_text = ""
            for page in doc:
                combined_text += "\n" + page.get_text()

            is_floor_plan = bool(re.search(r"\b(FLOOR\s*PLAN|GROUND\s*FLOOR|FIRST\s*FLOOR|TYPICAL\s*FLOOR|BASEMENT\s*PLAN|MEZZANINE|ROOF\s*PLAN)\b", combined_text, re.IGNORECASE))

            # 1. Search for explicit gross floor area in title block or schedules
            area_patterns = [
                r"(?:TOTAL\s*GROSS\s*FLOOR\s*AREA|GROSS\s*FLOOR\s*AREA|TOTAL\s*FLOOR\s*AREA|PLINTH\s*AREA|BUILT[- ]UP\s*AREA)\s*[:=\-]?\s*([0-9]+(?:[\.,][0-9]+)?)\s*(?:SQM|SQ\.?\s*M|M2|M²)?",
                r"(?:FLOOR\s*AREA)\s*[:=\-]?\s*([0-9]+(?:[\.,][0-9]+)?)\s*(?:SQM|SQ\.?\s*M|M2|M²)",
                r"AREA\s*[:=\-]?\s*([0-9]+(?:[\.,][0-9]+)?)\s*(?:SQM|M2|M²)",
            ]

            detected_gfa = None
            for pat in area_patterns:
                match = re.search(pat, combined_text, re.IGNORECASE)
                if match:
                    val_str = match.group(1).replace(",", "")
                    try:
                        detected_gfa = float(val_str)
                        if detected_gfa > 5:
                            break
                    except ValueError:
                        continue

            # 2. Extract room area schedule items if present
            room_breakdown = []
            room_pat = re.compile(
                r"\b(LIVING\s*(?:ROOM)?|MASTER\s*BEDROOM|BEDROOM\s*\d*|KITCHEN|DINING\s*(?:ROOM)?|BATHROOM\s*\d*|TOILET|BALCONY|OFFICE|CONFERENCE|CORRIDOR|STORE|LOBBY)\b\s*[:=\-]?\s*([0-9]+(?:[\.,][0-9]+)?)\s*(?:SQM|M2|M²)?",
                re.IGNORECASE
            )
            for m in room_pat.finditer(combined_text):
                room_name = m.group(1).strip().title()
                try:
                    area_val = float(m.group(2).replace(",", ""))
                    if 2.0 <= area_val <= 500.0:
                        room_breakdown.append({"name": room_name, "area_sqm": round(area_val, 2)})
                except ValueError:
                    pass

            # If room schedule exists and total GFA wasn't stated, sum the rooms
            sum_rooms = sum(r["area_sqm"] for r in room_breakdown)
            if detected_gfa is None and sum_rooms > 10:
                detected_gfa = round(sum_rooms * 1.15, 2)  # add 15% circulation/wall thickness
                source = "room_schedule_summation"
                confidence = 0.85
            elif detected_gfa is not None:
                source = "title_block_schedule"
                confidence = 0.95
            else:
                # 3. Vector geometry bounding estimation fallback if drawing contains dense vectors
                first_page = doc[0]
                drawings = first_page.get_drawings()
                if is_floor_plan and len(drawings) >= 15:
                    # Compute vector bounding box
                    x0s = [d["rect"].x0 for d in drawings if d.get("rect")]
                    y0s = [d["rect"].y0 for d in drawings if d.get("rect")]
                    x1s = [d["rect"].x1 for d in drawings if d.get("rect")]
                    y1s = [d["rect"].y1 for d in drawings if d.get("rect")]

                    if x0s and y0s and x1s and y1s:
                        span_w = (max(x1s) - min(x0s)) * scale_factor
                        span_h = (max(y1s) - min(y0s)) * scale_factor
                        # Plausible building floor span
                        if 6.0 <= span_w <= 100.0 and 6.0 <= span_h <= 100.0:
                            # Inferred usable footprint approx 55% of outer vector bounding envelope
                            detected_gfa = round(span_w * span_h * 0.55, 2)
                            source = "vector_polygon_estimation"
                            confidence = 0.75
                        else:
                            detected_gfa = None
                            source = "none"
                            confidence = 0.0
                    else:
                        source = "none"
                        confidence = 0.0
                else:
                    source = "none"
                    confidence = 0.0

            net_area = round(detected_gfa * 0.88, 2) if detected_gfa else None

            return {
                "is_floor_plan": is_floor_plan,
                "gross_floor_area_sqm": detected_gfa,
                "net_internal_area_sqm": net_area,
                "confidence_score": confidence if detected_gfa else 0.0,
                "source": source,
                "room_breakdown": room_breakdown[:20]
            }
        except Exception as e:
            return {
                "is_floor_plan": False,
                "gross_floor_area_sqm": None,
                "net_internal_area_sqm": None,
                "confidence_score": 0.0,
                "source": "error",
                "error": str(e),
                "room_breakdown": []
            }

