import psycopg
from pgvector.psycopg import register_vector
from google import genai
from google.genai import types
from app.config import settings
from typing import Optional

class RAGRetriever:
    @classmethod
    def get_gemini_client(cls) -> Optional[genai.Client]:
        if settings.GEMINI_API_KEY:
            try:
                return genai.Client(api_key=settings.GEMINI_API_KEY)
            except Exception as e:
                print(f"Failed to initialize Gemini Client: {e}")
        return None

    @classmethod
    def retrieve_relevant_codes(cls, query_text: str, discipline: str, limit: int = 5) -> str:
        """
        Retrieves relevant statutory chunks from PostgreSQL using vector cosine similarity
        or full-text keyword search as fallback.
        """
        client = cls.get_gemini_client()
        query_vector = None

        if client:
            try:
                res = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=query_text,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )
                query_vector = res.embeddings[0].values
            except Exception as e:
                print(f"Embedding query failed, falling back to keyword search: {e}")

        try:
            with psycopg.connect(settings.DATABASE_URL) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    if query_vector:
                        sql = """
                        SELECT c.clause_number, c.clause_title, c.content, d.title
                        FROM regulatory_chunks c
                        JOIN regulatory_documents d ON d.id = c.regulatory_document_id
                        WHERE d.is_active = TRUE
                          AND (d.category = %s OR d.category = 'general_building_control' OR %s = 'general')
                        ORDER BY c.embedding <=> %s
                        LIMIT %s;
                        """
                        cur.execute(sql, (discipline, discipline, query_vector, limit))
                    else:
                        keywords = [f"%{w.lower()}%" for w in query_text.split() if len(w) > 3]
                        if not keywords:
                            keywords = ["%building%"]
                        sql = """
                        SELECT c.clause_number, c.clause_title, c.content, d.title
                        FROM regulatory_chunks c
                        JOIN regulatory_documents d ON d.id = c.regulatory_document_id
                        WHERE d.is_active = TRUE
                          AND (c.content ILIKE %s OR c.clause_title ILIKE %s)
                        LIMIT %s;
                        """
                        cur.execute(sql, (keywords[0], keywords[0], limit))

                    rows = cur.fetchall()
                    if not rows:
                        cur.execute("""
                        SELECT c.clause_number, c.clause_title, c.content, d.title
                        FROM regulatory_chunks c
                        JOIN regulatory_documents d ON d.id = c.regulatory_document_id
                        WHERE d.is_active = TRUE
                        LIMIT %s;
                        """, (limit,))
                        rows = cur.fetchall()

                    results = []
                    for row in rows:
                        clause_num = row[0] or "General"
                        clause_title = row[1] or "Provision"
                        results.append(f"[{row[3]} - {clause_num} ({clause_title})]:\n{row[2]}")

                    return "\n\n".join(results)
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            return "Uganda National Building Code (2019) and Building Control Regulations (2020) require compliance with minimum room dimensions, light/ventilation ratios, and egress standards."
