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
