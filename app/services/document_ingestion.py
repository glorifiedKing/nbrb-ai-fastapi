import fitz  # PyMuPDF
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
    def extract_and_chunk_pdf(pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from statutory PDF and chunks by clause/section headers.
        Detects patterns like 'Section 12.4', 'Regulation 14', 'Clause 3.1.2', or '14.2 Ventilation'.
        """
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            full_text += f"\n--- PAGE {page_num + 1} ---\n" + page.get_text()

        clause_pattern = re.compile(
            r"(?P<clause>(?:Section|Regulation|Clause|Part)?\s*\d+(?:\.\d+)*)\s*[\.\-:]?\s+(?P<title>[A-Z][^\n]{3,80})\n(?P<body>[\s\S]*?)(?=(?:(?:Section|Regulation|Clause|Part)?\s*\d+(?:\.\d+)*)\s*[\.\-:]?\s+[A-Z]|\Z)",
            re.MULTILINE
        )

        chunks = []
        for match in clause_pattern.finditer(full_text):
            clause_num = match.group("clause").strip()
            title = match.group("title").strip()
            body = match.group("body").strip()

            cleaned_body = re.sub(r"--- PAGE \d+ ---", "", body).strip()

            if len(cleaned_body) > 25:
                chunks.append({
                    "clause_number": clause_num,
                    "clause_title": title,
                    "content": f"{clause_num} {title}\n{cleaned_body}",
                    "metadata": {
                        "source": "Uganda Building Regulations",
                        "title": title
                    }
                })

        if len(chunks) < 3:
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
                ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                """

                rows_to_insert = [
                    (
                        str(uuid.uuid4()),
                        document_id,
                        c["clause_number"],
                        c["clause_title"],
                        c["content"],
                        psycopg.types.json.Jsonb(c["metadata"]),
                        embeddings[idx]
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
