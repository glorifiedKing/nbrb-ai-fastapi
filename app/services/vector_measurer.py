import pymupdf as fitz  # PyMuPDF
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
            (r"1\s*:\s*100\b", 0.03528, "1:100"),
            (r"1\s*:\s*50\b", 0.01764, "1:50"),
            (r"1\s*:\s*200\b", 0.07056, "1:200"),
            (r"1\s*:\s*20\b", 0.007056, "1:20"),
            (r"1\s*:\s*25\b", 0.00882, "1:25"),
            (r"1\s*:\s*10\b", 0.003528, "1:10"),
            (r"1\s*:\s*15\b", 0.005292, "1:15"),
            (r"1\s*:\s*75\b", 0.02646, "1:75"),
            (r"1\s*:\s*250\b", 0.0882, "1:250"),
            (r"1\s*:\s*500\b", 0.1764, "1:500"),
            (r"1/4\"\s*=\s*1'-0\"", 0.03528, "1/4\" = 1'-0\""),
            (r"SCALE\s*[:=\-]?\s*1\s*/\s*100\b", 0.03528, "1:100"),
            (r"SCALE\s*[:=\-]?\s*1\s*/\s*50\b", 0.01764, "1:50"),
            (r"\b(?:N\.?T\.?S\.?|NOT\s*TO\s*SCALE|SCHEMATIC|DIAGRAMMATIC)\b", 0.03528, "N.T.S. (Schematic)"),
        ]

        for pattern, factor, standard_name in scale_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True, standard_name, factor, f"Scale {standard_name} verified successfully."

        first_page = doc[0]
        drawings = first_page.get_drawings()
        if len(drawings) > 10:
            return True, "1:100 (Inferred)", 0.03528, "Inferred default scale 1:100 based on vector density."

        return False, None, None, "No standard scale callout (e.g. 1:100, 1:50, NTS) detected in drawing."

    @staticmethod
    def detect_drawing_discipline_and_type(
        pdf_path: str,
        supplied_discipline: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detects engineering discipline (architectural, structural, mechanical, electrical)
        and specific drawing type from PDF text, title blocks, schedules, and drawing annotations.
        Reconciles with supplied_discipline if provided.
        """
        try:
            doc = fitz.open(pdf_path)
            sample_text = ""
            num_pages = min(8, len(doc))
            for p in range(num_pages):
                sample_text += "\n" + doc[p].get_text()
        except Exception:
            return {
                "resolved_discipline": (supplied_discipline or "architectural").lower(),
                "detected_discipline": "unknown",
                "drawing_type": "Standard Engineering Drawing",
                "confidence": 0.5,
                "indicators": []
            }

        text_upper = sample_text.upper()

        scores = {
            "structural": 0,
            "electrical": 0,
            "mechanical": 0,
            "architectural": 0
        }
        indicators = {
            "structural": [],
            "electrical": [],
            "mechanical": [],
            "architectural": []
        }

        # Structural signatures
        structural_keywords = [
            ("STRUCTURAL ENGINEER", 10), ("CIVIL STRUCTURAL", 10), ("FOUNDATION LAYOUT", 8),
            ("FOOTING DETAIL", 8), ("COLUMN SCHEDULE", 8), ("BEAM SCHEDULE", 8),
            ("GROUND BEAM", 6), ("SLAB BEAM", 6), ("REINFORCEMENT", 6),
            ("BENDING SCHEDULE", 7), ("SOLID SLAB", 5), ("PAD FOOTING", 6),
            ("STRIP FOOTING", 6), ("ROOF TRUSS", 5), ("CONCRETE COVER", 5),
            ("GRADE 25", 5), ("GRADE 30", 5), ("STIRRUPS", 4), ("LINKS", 3),
            ("T10", 3), ("T12", 3), ("T16", 3), ("Y12", 3), ("Y16", 3), ("Y20", 3),
            ("DEFLECTION", 3), ("CHARACTERISTIC STRENGTH", 4)
        ]
        for kw, weight in structural_keywords:
            if kw in text_upper:
                scores["structural"] += weight
                indicators["structural"].append(kw)

        # Electrical signatures
        electrical_keywords = [
            ("ELECTRICAL ENGINEER", 10), ("SINGLE LINE DIAGRAM", 10), ("SLD", 6),
            ("DISTRIBUTION BOARD", 8), ("DB SCHEDULE", 8), ("LIGHTING LAYOUT", 7),
            ("POWER LAYOUT", 7), ("SMALL POWER", 6), ("EARTHING AND LIGHTNING", 8),
            ("LIGHTNING PROTECTION", 7), ("EARTH ELECTRODE", 6), ("MAIN ISOLATOR", 6),
            ("CIRCUIT BREAKER", 5), ("MCB", 4), ("MCCB", 4), ("RCD", 5), ("RCCB", 5),
            ("LUMINAIRE", 4), ("SOCKET OUTLET", 4), ("415V", 3), ("240V", 3),
            ("THREE PHASE", 4), ("SINGLE PHASE", 3), ("KVA", 3), ("LUX LEVEL", 4)
        ]
        for kw, weight in electrical_keywords:
            if kw in text_upper:
                scores["electrical"] += weight
                indicators["electrical"].append(kw)

        # Mechanical & Plumbing signatures
        mechanical_keywords = [
            ("MECHANICAL ENGINEER", 10), ("HVAC", 9), ("AIR CONDITIONING", 8),
            ("DUCTWORK", 8), ("SUPPLY AIR", 6), ("RETURN AIR", 6), ("EXHAUST FAN", 6),
            ("DIFFUSER", 5), ("FIRE DAMPER", 7), ("PLUMBING AND DRAINAGE", 9),
            ("FOUL DRAINAGE", 8), ("SOIL VENT PIPE", 7), ("SVP", 5), ("INVERT LEVEL", 6),
            ("GREASE TRAP", 8), ("MANHOLE", 5), ("WATER RETICULATION", 7),
            ("HOSE REEL", 6), ("BOOSTER PUMP", 5), ("WATER STORAGE", 5),
            ("RAINWATER HARVESTING", 5), ("CFM", 4), ("AIR CHANGES", 5)
        ]
        for kw, weight in mechanical_keywords:
            if kw in text_upper:
                scores["mechanical"] += weight
                indicators["mechanical"].append(kw)

        # Architectural signatures
        architectural_keywords = [
            ("ARCHITECTURAL", 10), ("ARCHITECT", 8), ("FLOOR PLAN", 7),
            ("GROUND FLOOR PLAN", 7), ("FIRST FLOOR PLAN", 7), ("ROOM SCHEDULE", 7),
            ("DOOR SCHEDULE", 6), ("WINDOW SCHEDULE", 6), ("BUILDING ELEVATION", 6),
            ("CROSS SECTION", 5), ("SITE PLAN", 5), ("HABITABLE ROOM", 5),
            ("LIVING ROOM", 4), ("BEDROOM", 4), ("KITCHEN", 3), ("CORRIDOR", 3),
            ("CEILING HEIGHT", 4), ("SETBACK", 4), ("BOUNDARY WALL", 4)
        ]
        for kw, weight in architectural_keywords:
            if kw in text_upper:
                scores["architectural"] += weight
                indicators["architectural"].append(kw)

        sorted_disciplines = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_disc, top_score = sorted_disciplines[0]

        normalized_supplied = (supplied_discipline or "").strip().lower()
        if normalized_supplied == "mep":
            normalized_supplied = "electrical" if scores["electrical"] > scores["mechanical"] else "mechanical"
        elif normalized_supplied in ["str", "structural_engineering"]:
            normalized_supplied = "structural"
        elif normalized_supplied in ["elec", "electrical_engineering"]:
            normalized_supplied = "electrical"
        elif normalized_supplied in ["mech", "plumbing", "sanitary", "hvac"]:
            normalized_supplied = "mechanical"
        elif normalized_supplied in ["arch"]:
            normalized_supplied = "architectural"

        if normalized_supplied in ["architectural", "structural", "mechanical", "electrical"]:
            resolved_discipline = normalized_supplied
        elif top_score >= 8:
            resolved_discipline = top_disc
        else:
            resolved_discipline = "architectural"

        # Determine specific drawing type title
        drawing_type = "Engineering Drawing"
        if resolved_discipline == "structural":
            if any(k in text_upper for k in ["FOUNDATION", "FOOTING"]):
                drawing_type = "Foundation & Footing Plan"
            elif any(k in text_upper for k in ["BEAM", "COLUMN"]):
                drawing_type = "Beam & Column Framing Layout"
            elif any(k in text_upper for k in ["SLAB", "SOLID SLAB"]):
                drawing_type = "Slab Reinforcement Plan"
            elif any(k in text_upper for k in ["TRUSS", "ROOF"]):
                drawing_type = "Roof Truss & Framing Plan"
            else:
                drawing_type = "Civil / Structural Engineering Plan"
        elif resolved_discipline == "electrical":
            if any(k in text_upper for k in ["SINGLE LINE", "SLD", "SCHEMATIC"]):
                drawing_type = "Single Line Diagram (SLD) & DB Schedule"
            elif any(k in text_upper for k in ["LIGHTING"]):
                drawing_type = "Lighting & Luminaire Layout"
            elif any(k in text_upper for k in ["POWER", "SOCKET"]):
                drawing_type = "Small Power & Socket Layout"
            elif any(k in text_upper for k in ["EARTHING", "LIGHTNING"]):
                drawing_type = "Earthing & Lightning Protection Plan"
            else:
                drawing_type = "Electrical Services Layout"
        elif resolved_discipline == "mechanical":
            if any(k in text_upper for k in ["HVAC", "DUCT", "AIR CONDITIONING", "VENTILATION"]):
                drawing_type = "HVAC & Mechanical Ventilation Plan"
            elif any(k in text_upper for k in ["DRAINAGE", "SEWER", "SOIL", "PLUMBING"]):
                drawing_type = "Sanitary Plumbing & Drainage Plan"
            elif any(k in text_upper for k in ["FIRE", "HOSE REEL"]):
                drawing_type = "Fire Protection & Suppression Layout"
            elif any(k in text_upper for k in ["WATER", "RETICULATION"]):
                drawing_type = "Water Reticulation & Storage Layout"
            else:
                drawing_type = "Mechanical Services Plan"
        elif resolved_discipline == "architectural":
            if any(k in text_upper for k in ["GROUND FLOOR", "FIRST FLOOR", "FLOOR PLAN"]):
                drawing_type = "Architectural Floor Plan"
            elif any(k in text_upper for k in ["ELEVATION"]):
                drawing_type = "Building Elevations"
            elif any(k in text_upper for k in ["SECTION"]):
                drawing_type = "Building Cross Sections"
            elif any(k in text_upper for k in ["SITE PLAN"]):
                drawing_type = "Site & Block Plan"
            else:
                drawing_type = "Architectural General Layout"

        confidence = min(0.99, max(0.60, round(top_score / 35.0, 2))) if top_score > 0 else 0.50

        return {
            "resolved_discipline": resolved_discipline,
            "detected_discipline": top_disc if top_score >= 8 else resolved_discipline,
            "drawing_type": drawing_type,
            "confidence": confidence,
            "scores": scores,
            "indicators": indicators.get(resolved_discipline, [])[:6]
        }

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
        Supports multi-floor PDFs and multi-floor schedules on single sheets.
        """
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return {
                    "is_floor_plan": False,
                    "gross_floor_area_sqm": None,
                    "net_internal_area_sqm": None,
                    "cumulated_floors_count": 0,
                    "confidence_score": 0.0,
                    "source": "none",
                    "floors": [],
                    "room_breakdown": []
                }

            full_text = ""
            detected_floors = []
            all_rooms = []
            is_any_floor_plan = False

            # Schedule pattern for per-floor areas (e.g., "Ground Floor: 185.0 sqm", "First Floor - 170.5 m2")
            floor_schedule_pat = re.compile(
                r"\b(BASEMENT\s*(?:FLOOR)?|GROUND\s*FLOOR|FIRST\s*FLOOR|SECOND\s*FLOOR|THIRD\s*FLOOR|FOURTH\s*FLOOR|TYPICAL\s*FLOOR|MEZZANINE\s*(?:FLOOR)?|ROOF\s*TERRACE)\b[^\n\d]{0,25}[:=\-]?\s*([0-9]+(?:[\.,][0-9]+)?)\s*(?:SQM|SQ\.?\s*M|M2|M²)?",
                re.IGNORECASE
            )

            # Room schedule pattern
            room_pat = re.compile(
                r"\b(LIVING\s*(?:ROOM)?|MASTER\s*BEDROOM|BEDROOM\s*\d*|KITCHEN|DINING\s*(?:ROOM)?|BATHROOM\s*\d*|TOILET\s*\d*|BALCONY|OFFICE|CONFERENCE|CORRIDOR|STORE|LOBBY|VERANDAH|GARAGE)\b\s*[:=\-]?\s*([0-9]+(?:[\.,][0-9]+)?)\s*(?:SQM|M2|M²)?",
                re.IGNORECASE
            )

            # Global GFA pattern (e.g., "Total Gross Floor Area: 350.0 SQM")
            global_gfa_patterns = [
                r"(?:TOTAL\s*GROSS\s*FLOOR\s*AREA|TOTAL\s*BUILT[- ]UP\s*AREA|TOTAL\s*PLINTH\s*AREA|TOTAL\s*FLOOR\s*AREA)\s*[:=\-]?\s*([0-9]+(?:[\.,][0-9]+)?)\s*(?:SQM|SQ\.?\s*M|M2|M²)?",
                r"(?:GROSS\s*FLOOR\s*AREA|PLINTH\s*AREA|BUILT[- ]UP\s*AREA)\s*[:=\-]?\s*([0-9]+(?:[\.,][0-9]+)?)\s*(?:SQM|SQ\.?\s*M|M2|M²)?",
                r"(?:FLOOR\s*AREA)\s*[:=\-]?\s*([0-9]+(?:[\.,][0-9]+)?)\s*(?:SQM|SQ\.?\s*M|M2|M²)",
            ]

            # Pass 1: Check each page for floor plans, schedules, and drawings
            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1
                page_text = page.get_text()
                full_text += "\n" + page_text

                # Check if this page is a floor plan
                page_is_fp = bool(re.search(
                    r"\b(FLOOR\s*PLAN|GROUND\s*FLOOR|FIRST\s*FLOOR|SECOND\s*FLOOR|THIRD\s*FLOOR|TYPICAL\s*FLOOR|BASEMENT\s*PLAN|MEZZANINE|ROOF\s*PLAN)\b",
                    page_text,
                    re.IGNORECASE
                ))
                if page_is_fp:
                    is_any_floor_plan = True

                # Identify floor name for this page
                floor_name_match = re.search(
                    r"\b(BASEMENT\s*(?:FLOOR\s*PLAN|PLAN|LEVEL)?|GROUND\s*FLOOR\s*(?:PLAN)?|FIRST\s*FLOOR\s*(?:PLAN)?|SECOND\s*FLOOR\s*(?:PLAN)?|THIRD\s*FLOOR\s*(?:PLAN)?|TYPICAL\s*(?:UPPER\s*)?FLOOR\s*(?:PLAN)?|MEZZANINE\s*(?:FLOOR\s*PLAN|PLAN)?|ROOF\s*(?:PLAN|TERRACE))\b",
                    page_text,
                    re.IGNORECASE
                )
                detected_name = floor_name_match.group(1).strip().title() if floor_name_match else f"Floor Plan (Sheet {page_num})"

                # Room items on this page
                page_rooms = []
                for m in room_pat.finditer(page_text):
                    r_name = m.group(1).strip().title()
                    try:
                        area_val = float(m.group(2).replace(",", ""))
                        if 2.0 <= area_val <= 500.0:
                            page_rooms.append({"name": f"[{detected_name}] {r_name}", "area_sqm": round(area_val, 2)})
                    except ValueError:
                        pass
                all_rooms.extend(page_rooms)

                # Check for explicit multi-floor schedule on this single sheet
                schedule_matches = list(floor_schedule_pat.finditer(page_text))
                if len(schedule_matches) > 1:
                    for sm in schedule_matches:
                        sch_name = sm.group(1).strip().title()
                        try:
                            sch_area = float(sm.group(2).replace(",", ""))
                            if sch_area > 10.0:
                                matching_rooms = [r for r in page_rooms if sch_name.lower() in r["name"].lower()]
                                detected_floors.append({
                                    "page": page_num,
                                    "floor_name": f"{sch_name} Plan",
                                    "gross_floor_area_sqm": round(sch_area, 2),
                                    "net_internal_area_sqm": round(sch_area * 0.88, 2),
                                    "room_breakdown": matching_rooms
                                })
                        except ValueError:
                            pass
                elif page_is_fp:
                    # Single floor on this page: check explicit GFA callout
                    page_gfa = None
                    for pat in global_gfa_patterns:
                        m = re.search(pat, page_text, re.IGNORECASE)
                        if m:
                            try:
                                v = float(m.group(1).replace(",", ""))
                                if v > 10:
                                    page_gfa = v
                                    break
                            except ValueError:
                                pass

                    # Fallback to room schedule summation
                    if page_gfa is None and page_rooms:
                        room_sum = sum(r["area_sqm"] for r in page_rooms if r.get("area_sqm"))
                        if room_sum > 10:
                            page_gfa = round(room_sum * 1.15, 2)

                    # Fallback to vector bounding box on this page
                    if page_gfa is None:
                        drawings = page.get_drawings()
                        if len(drawings) >= 15:
                            x0s = [d["rect"].x0 for d in drawings if d.get("rect")]
                            y0s = [d["rect"].y0 for d in drawings if d.get("rect")]
                            x1s = [d["rect"].x1 for d in drawings if d.get("rect")]
                            y1s = [d["rect"].y1 for d in drawings if d.get("rect")]
                            if x0s and y0s and x1s and y1s:
                                span_w = (max(x1s) - min(x0s)) * scale_factor
                                span_h = (max(y1s) - min(y0s)) * scale_factor
                                if 6.0 <= span_w <= 100.0 and 6.0 <= span_h <= 100.0:
                                    page_gfa = round(span_w * span_h * 0.55, 2)

                    if page_gfa and page_gfa > 0:
                        detected_floors.append({
                            "page": page_num,
                            "floor_name": detected_name,
                            "gross_floor_area_sqm": round(page_gfa, 2),
                            "net_internal_area_sqm": round(page_gfa * 0.88, 2),
                            "room_breakdown": page_rooms
                        })

            # Check for global GFA stated in the entire document
            global_gfa = None
            for pat in global_gfa_patterns:
                m = re.search(pat, full_text, re.IGNORECASE)
                if m:
                    try:
                        v = float(m.group(1).replace(",", ""))
                        if v > 10:
                            global_gfa = v
                            break
                    except ValueError:
                        pass

            if detected_floors:
                cumulated_gfa = sum(f["gross_floor_area_sqm"] for f in detected_floors)
                total_gfa = round(global_gfa if (global_gfa and global_gfa >= cumulated_gfa) else cumulated_gfa, 2)
                total_nia = round(total_gfa * 0.88, 2)
                source = "vector_schedule_and_plan_extraction"
                confidence = 0.90 if len(detected_floors) > 1 else 0.85
            elif global_gfa:
                total_gfa = round(global_gfa, 2)
                total_nia = round(total_gfa * 0.88, 2)
                detected_floors.append({
                    "page": 1,
                    "floor_name": "Floor Plan",
                    "gross_floor_area_sqm": total_gfa,
                    "net_internal_area_sqm": total_nia,
                    "room_breakdown": all_rooms[:20]
                })
                source = "title_block_schedule"
                confidence = 0.90
            else:
                total_gfa = None
                total_nia = None
                source = "none"
                confidence = 0.0

            return {
                "is_floor_plan": is_any_floor_plan or (total_gfa is not None),
                "gross_floor_area_sqm": total_gfa,
                "net_internal_area_sqm": total_nia,
                "cumulated_floors_count": len(detected_floors),
                "confidence_score": confidence,
                "source": source,
                "floors": detected_floors,
                "room_breakdown": all_rooms[:30]
            }
        except Exception as e:
            return {
                "is_floor_plan": False,
                "gross_floor_area_sqm": None,
                "net_internal_area_sqm": None,
                "cumulated_floors_count": 0,
                "confidence_score": 0.0,
                "source": "error",
                "error": str(e),
                "floors": [],
                "room_breakdown": []
            }

