import pymupdf as fitz  # PyMuPDF
import re
import uuid
import psycopg
import math
from pgvector.psycopg import register_vector
from google import genai
from google.genai import types
from app.config import settings
from typing import Optional, List, Dict, Any

class RegulatoryIngestionEngine:

    @staticmethod
    def get_gemini_client() -> Optional[genai.Client]:
        if settings.GEMINI_API_KEY:
            try:
                return genai.Client(api_key=settings.GEMINI_API_KEY)
            except Exception as e:
                print(f"Failed to create Gemini client: {e}")
        return None

    @staticmethod
    def _sanitize_text(val: Optional[str]) -> str:
        """
        Removes null bytes (0x00) which are forbidden in PostgreSQL text and JSON fields.
        """
        if not val:
            return ""
        return val.replace("\x00", "")

    @classmethod
    def _sanitize_metadata(cls, meta: Any) -> Any:
        """
        Recursively sanitizes metadata dictionaries, lists, and strings of any null bytes.
        """
        if isinstance(meta, str):
            return cls._sanitize_text(meta)
        elif isinstance(meta, dict):
            return {cls._sanitize_text(str(k)): cls._sanitize_metadata(v) for k, v in meta.items()}
        elif isinstance(meta, list):
            return [cls._sanitize_metadata(item) for item in meta]
        return meta

    @staticmethod
    def decode_char_shifted(c: str) -> str:
        """
        Decodes characters from PDFs exported with custom TrueType/Identity-H font subsets
        lacking ToUnicode CMaps, where glyph IDs are shifted by -29 relative to ASCII.
        """
        code = ord(c)
        if code == 0:
            return ''
        if 0x0D7A <= code <= 0x0D93:
            # Small-caps uppercase A-Z mapped sequentially from 0x0d7a (3450)
            return chr(code - 0x0D7A + ord('A'))
        elif code in (0x0D75, 0x0D76):
            return 'ff'
        elif code == 0x0D77:
            return 'ffl'
        elif code == 0xBF:
            return 'fi'
        elif code == 0xC0:
            return 'fl'
        elif code == 0xB2:
            return '-'
        elif code in (0xB3, 0xB4):
            return '"'
        elif code in (3, 32):
            return ' '
        elif code == 10:
            return '\n'
        elif 1 <= code <= 127:
            target = code + 29
            if 32 <= target <= 126:
                return chr(target)
        return c

    @classmethod
    def extract_clean_text_from_doc(cls, doc: fitz.Document) -> str:
        """
        Detects if PDF fonts suffer from the +29 subset shift and extracts
        clean, human-readable text page by page.
        """
        sample_text = ""
        for p in range(min(5, len(doc))):
            sample_text += doc[p].get_text()

        is_shifted = ('\x03' in sample_text and sample_text.count('\x03') > 10) or any(
            sig in sample_text for sig in ['6XEFODVV', '7$%/(', '5HJXODWLRQ', '6HFWLRQ', '%8,/', '3RSXODW', '3ODVWLF']
        )

        full_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            if not is_shifted:
                page_text = page.get_text()
            else:
                blocks = page.get_text('dict').get('blocks', [])
                lines_out = []
                for b in blocks:
                    if 'lines' not in b:
                        continue
                    for l in b['lines']:
                        line_str = ''
                        for s in l['spans']:
                            txt = s['text']
                            fn = s['font'].lower()
                            if ('italic' in fn or 'boldital' in fn or 'minion' in fn or 'calibri' in fn) and '\x03' not in txt:
                                line_str += txt
                            else:
                                line_str += ''.join(cls.decode_char_shifted(ch) for ch in txt)
                        if line_str.strip():
                            lines_out.append(line_str)
                page_text = '\n'.join(lines_out)
            page_text = cls._sanitize_text(page_text)
            full_text += f"\n--- PAGE {page_num + 1} ---\n" + page_text

        return cls._sanitize_text(full_text)

    @classmethod
    def extract_and_chunk_pdf(cls, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts clean text from statutory PDF and chunks by clause/section headers.
        Detects patterns like 'Section 12.4', 'Regulation 14', 'Paragraph 79', 'Table 4', or 'Part II'.
        """
        doc = fitz.open(pdf_path)
        full_text = cls.extract_clean_text_from_doc(doc)

        clause_pattern = re.compile(
            r"(?P<clause>(?:Section|Regulation|Clause|Part|Paragraph|Table|Schedule)?\s*\d+(?:\.\d+)*)\s*[\.\-:]\s+(?P<title>[A-Z][^\n]{2,80})\n(?P<body>[\s\S]*?)(?=(?:(?:Section|Regulation|Clause|Part|Paragraph|Table|Schedule)?\s*\d+(?:\.\d+)*)\s*[\.\-:]\s+[A-Z]|\Z)",
            re.MULTILINE | re.IGNORECASE
        )

        chunks = []
        for match in clause_pattern.finditer(full_text):
            clause_num = cls._sanitize_text(match.group("clause")).strip()
            title = cls._sanitize_text(match.group("title")).strip()
            body = cls._sanitize_text(match.group("body")).strip()

            cleaned_body = re.sub(r"--- PAGE \d+ ---", "", body).strip()

            if len(cleaned_body) > 30:
                chunks.append({
                    "clause_number": clause_num,
                    "clause_title": title,
                    "content": cls._sanitize_text(f"{clause_num} {title}\n{cleaned_body}"),
                    "metadata": {
                        "source": "Uganda Building Regulations & Code",
                        "title": title
                    }
                })

        if len(chunks) < 5:
            # Fallback word-slice chunking
            words = full_text.split()
            chunk_size = 350
            overlap = 50
            chunks = []
            for i in range(0, len(words), chunk_size - overlap):
                chunk_slice = " ".join(words[i:i + chunk_size])
                cleaned_slice = cls._sanitize_text(re.sub(r"--- PAGE \d+ ---", "", chunk_slice)).strip()
                if len(cleaned_slice) > 40:
                    chunks.append({
                        "clause_number": f"Sec-{(i // (chunk_size - overlap)) + 1}",
                        "clause_title": "General Provision",
                        "content": cleaned_slice,
                        "metadata": {"type": "fallback_chunk"}
                    })

        return chunks

    @classmethod
    def generate_embeddings_batch(
        cls,
        texts: List[str],
    ) -> List[List[float]]:
        """Generate and validate Gemini embeddings. Never return fake vectors."""

        if not texts:
            return []

        client = cls.get_gemini_client()

        if client is None:
            raise RuntimeError(
                "Gemini client unavailable. Check GEMINI_API_KEY."
            )

        embeddings: List[List[float]] = []
        batch_size = 16
        expected_dimensions = 768

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=expected_dimensions,
                    ),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Gemini embedding failed for batch "
                    f"{i // batch_size + 1}: {exc}"
                ) from exc

            batch_embeddings = response.embeddings or []

            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} embeddings but received "
                    f"{len(batch_embeddings)}."
                )

            for index, item in enumerate(batch_embeddings):
                values = item.values

                if values is None or len(values) != expected_dimensions:
                    raise RuntimeError(
                        f"Invalid embedding dimensions for chunk {i + index}."
                    )

                vector = [float(value) for value in values]

                if not all(math.isfinite(value) for value in vector):
                    raise RuntimeError(
                        f"Embedding contains NaN or infinity at chunk "
                        f"{i + index}."
                    )

                if not any(value != 0.0 for value in vector):
                    raise RuntimeError(
                        f"Embedding is entirely zero at chunk {i + index}. "
                        "Aborting ingestion."
                    )

                embeddings.append(vector)

            print(
                f"Generated embeddings: "
                f"{len(embeddings)}/{len(texts)}"
            )

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: {len(embeddings)} generated "
                f"for {len(texts)} chunks."
            )

        return embeddings

    @classmethod
    def ingest_document(
        cls, 
        document_id: str, 
        pdf_path: str
    ) -> int:
        """
        Parses PDF, computes embeddings, and performs batch insertion into PostgreSQL.
        """
        try:
            chunks = cls.extract_and_chunk_pdf(pdf_path)
            if not chunks:
                raise ValueError("No extractable text or clauses found in statutory PDF.")

            chunk_texts = [cls._sanitize_text(c["content"]) for c in chunks]
            embeddings = cls.generate_embeddings_batch(chunk_texts)

            with psycopg.connect(settings.DATABASE_URL) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM regulatory_chunks WHERE regulatory_document_id = %s;", (document_id,))

                    insert_query = """
                    INSERT INTO regulatory_chunks (
                        id, regulatory_document_id, clause_number, clause_title, content, metadata, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::vector);
                    """

                    rows_to_insert = [
                        (
                            str(uuid.uuid4()),
                            document_id,
                            cls._sanitize_text(c["clause_number"]),
                            cls._sanitize_text(c["clause_title"]),
                            cls._sanitize_text(c["content"]),
                            psycopg.types.json.Jsonb(cls._sanitize_metadata(c.get("metadata", {}))),
                            "[" + ",".join(str(v) for v in embeddings[idx]) + "]"
                        )
                        for idx, c in enumerate(chunks)
                    ]

                    cur.executemany(insert_query, rows_to_insert)

                    cur.execute(
                        "UPDATE regulatory_documents SET total_chunks = %s, status = 'indexed', error_message = NULL, updated_at = NOW() WHERE id = %s;",
                        (len(chunks), document_id)
                    )

                    conn.commit()

            return len(chunks)
        except Exception as e:
            try:
                with psycopg.connect(settings.DATABASE_URL) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE regulatory_documents SET status = 'failed', error_message = %s, updated_at = NOW() WHERE id = %s;",
                            (str(e)[:500], document_id)
                        )
                        conn.commit()
            except Exception as update_err:
                print(f"Failed to record ingestion error in regulatory_documents: {update_err}")
            raise
