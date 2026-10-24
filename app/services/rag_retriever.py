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
    def retrieve_relevant_codes(cls, query_text: str, discipline: str, limit: int = 6) -> str:
        """
        Retrieves relevant statutory chunks from PostgreSQL using vector cosine similarity
        or full-text keyword search. Prioritizes documents matching the target discipline first,
        then includes complementary general building control regulations.
        """
        client = cls.get_gemini_client()
        query_vector = None
        target_disc = (discipline or "architectural").lower()

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

        primary_limit = max(3, limit - 2)
        complementary_limit = max(2, limit - primary_limit)

        try:
            with psycopg.connect(settings.DATABASE_URL) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    primary_rows = []
                    complementary_rows = []

                    if query_vector:
                        vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

                        # 1. First retrieve from primary matching discipline
                        sql_primary = """
                        SELECT c.clause_number, c.clause_title, c.content, d.title, d.category
                        FROM regulatory_chunks c
                        JOIN regulatory_documents d ON d.id = c.regulatory_document_id
                        WHERE d.is_active = TRUE
                          AND d.category = %s
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s;
                        """
                        cur.execute(sql_primary, (target_disc, vector_str, primary_limit))
                        primary_rows = cur.fetchall()

                        # 2. Retrieve complementary statutory clauses (general building control / other codes)
                        sql_comp = """
                        SELECT c.clause_number, c.clause_title, c.content, d.title, d.category
                        FROM regulatory_chunks c
                        JOIN regulatory_documents d ON d.id = c.regulatory_document_id
                        WHERE d.is_active = TRUE
                          AND d.category != %s
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s;
                        """
                        cur.execute(sql_comp, (target_disc, vector_str, complementary_limit))
                        complementary_rows = cur.fetchall()

                    else:
                        keywords = [f"%{w.lower()}%" for w in query_text.split() if len(w) > 3]
                        kw_pattern = keywords[0] if keywords else "%building%"

                        # 1. Primary discipline keyword search
                        sql_primary = """
                        SELECT c.clause_number, c.clause_title, c.content, d.title, d.category
                        FROM regulatory_chunks c
                        JOIN regulatory_documents d ON d.id = c.regulatory_document_id
                        WHERE d.is_active = TRUE
                          AND d.category = %s
                          AND (c.content ILIKE %s OR c.clause_title ILIKE %s)
                        LIMIT %s;
                        """
                        cur.execute(sql_primary, (target_disc, kw_pattern, kw_pattern, primary_limit))
                        primary_rows = cur.fetchall()

                        # 2. Complementary keyword search
                        sql_comp = """
                        SELECT c.clause_number, c.clause_title, c.content, d.title, d.category
                        FROM regulatory_chunks c
                        JOIN regulatory_documents d ON d.id = c.regulatory_document_id
                        WHERE d.is_active = TRUE
                          AND d.category != %s
                          AND (c.content ILIKE %s OR c.clause_title ILIKE %s)
                        LIMIT %s;
                        """
                        cur.execute(sql_comp, (target_disc, kw_pattern, kw_pattern, complementary_limit))
                        complementary_rows = cur.fetchall()

                    # Fallback if no primary rows matched the target discipline
                    if not primary_rows and not complementary_rows:
                        cur.execute("""
                        SELECT c.clause_number, c.clause_title, c.content, d.title, d.category
                        FROM regulatory_chunks c
                        JOIN regulatory_documents d ON d.id = c.regulatory_document_id
                        WHERE d.is_active = TRUE
                        LIMIT %s;
                        """, (limit,))
                        primary_rows = cur.fetchall()

                    results = []
                    if primary_rows:
                        results.append(f"=== PRIMARY STATUTORY STANDARDS: {target_disc.upper()} ===")
                        for row in primary_rows:
                            clause_num = row[0] or "General"
                            clause_title = row[1] or "Provision"
                            doc_title = row[3]
                            results.append(f"[{doc_title} - Clause {clause_num} ({clause_title})]:\n{row[2]}")

                    if complementary_rows:
                        results.append(f"\n=== COMPLEMENTARY GENERAL STATUTORY CODES ===")
                        for row in complementary_rows:
                            clause_num = row[0] or "General"
                            clause_title = row[1] or "Provision"
                            doc_title = row[3]
                            results.append(f"[{doc_title} - Clause {clause_num} ({clause_title})]:\n{row[2]}")

                    if results:
                        return "\n\n".join(results)

                    return "Uganda National Building Code (2019) and Building Control Regulations (2020) require compliance with statutory structural stability, safety, and serviceability standards."
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            return "Uganda National Building Code (2019) and Building Control Regulations (2020) require compliance with statutory structural stability, safety, and serviceability standards."
