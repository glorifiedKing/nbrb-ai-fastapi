import pymupdf as fitz  # PyMuPDF
import re
import uuid
import psycopg
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
    def decode_char_shifted(c: str) -> str:
        """
        Decodes characters from PDFs exported with custom TrueType/Identity-H font subsets
        lacking ToUnicode CMaps, where glyph IDs are shifted by -29 relative to ASCII.
        """
        code = ord(c)
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
            full_text += f"\n--- PAGE {page_num + 1} ---\n" + page_text

        return full_text

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
            clause_num = match.group("clause").strip()
            title = match.group("title").strip()
            body = match.group("body").strip()

            cleaned_body = re.sub(r"--- PAGE \d+ ---", "", body).strip()

            if len(cleaned_body) > 30:
                chunks.append({
                    "clause_number": clause_num,
                    "clause_title": title,
                    "content": f"{clause_num} {title}\n{cleaned_body}",
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
                cleaned_slice = re.sub(r"--- PAGE \d+ ---", "", chunk_slice).strip()
                if len(cleaned_slice) > 40:
                    chunks.append({
                        "clause_number": f"Sec-{(i // (chunk_size - overlap)) + 1}",
                        "clause_title": "General Provision",
                        "content": cleaned_slice,
                        "metadata": {"type": "fallback_chunk"}
                    })

        return chunks

    @classmethod
    def generate_embeddings_batch(cls, texts: List[str]) -> List[List[float]]:
        """
        Generates 768-dimensional embeddings using Gemini gemini-embedding-001 with MRL.
        """
        client = cls.get_gemini_client()
        embeddings = []
        batch_size = 16

        if client:
            try:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=batch,
                        config=types.EmbedContentConfig(
                            output_dimensionality=768
                        )
                    )
                    for item in response.embeddings:
                        embeddings.append(item.values)
                return embeddings
            except Exception as e:
                print(f"Gemini batch embedding error: {e}")

        return [[0.0] * 768 for _ in texts]

    @classmethod
    def ingest_document(
        cls, 
        document_id: str, 
        pdf_path: str
    ) -> int:
        """
        Parses PDF, computes embeddings, and performs batch insertion into PostgreSQL.
        """
        chunks = cls.extract_and_chunk_pdf(pdf_path)
        if not chunks:
            raise ValueError("No extractable text or clauses found in statutory PDF.")

        chunk_texts = [c["content"] for c in chunks]
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
                        c["clause_number"],
                        c["clause_title"],
                        c["content"],
                        psycopg.types.json.Jsonb(c["metadata"]),
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
