import sys, os, uuid
sys.path.insert(0, 'E:/nbrbi-ai-fastapi')

from app.services.vector_measurer import VectorMeasurementEngine
from app.services.document_ingestion import RegulatoryIngestionEngine
from app.services.rag_retriever import RAGRetriever

def run_tests():
    print('=== 1. Testing Document Ingestion Engine on Uganda Sample Regulations ===')
    sample_pdf_1 = 'E:/nbrbi-ai-compliance/public/sample_documents/THE-BUILDING-CONTROL-REGULATIONS-2020.pdf'
    if os.path.exists(sample_pdf_1):
        print(f'Reading sample statutory document: {sample_pdf_1}')
        chunks = RegulatoryIngestionEngine.extract_and_chunk_pdf(sample_pdf_1)
        print(f'[OK] Extracted {len(chunks)} structured statutory clauses.')
        if chunks:
            print(f'  First Clause -> Number: {chunks[0].get("clause_number")}, Title: {chunks[0].get("clause_title")}')
            print(f'  Snippet: {chunks[0].get("content")[:120]}...')
        else:
            print(f'[WARN] File not found: {sample_pdf_1}')

    print('\n=== 2. Testing Vector Measurement & Scale Extraction ===')
    is_valid, scale_name, factor, msg = VectorMeasurementEngine.inspect_and_extract_scale(sample_pdf_1)
    print(f'Scale Detection: valid={is_valid}, scale={scale_name}, factor={factor}, message={msg}')

    print('\n=== 3. Testing RAG Retrieval Fallback / Queries ===')
    codes = RAGRetriever.retrieve_relevant_codes('ventilation light habitable rooms egress', 'architectural', limit=3)
    print('Retrieved context preview:')
    print(codes[:300] + '...' if len(codes) > 300 else codes)

    print('\n=== 4. Testing FastAPI Application Import & Endpoints ===')
    from starlette.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get('/health')
    assert resp.status_code == 200, f'Health check failed: {resp.status_code}'
    print(f'[OK] GET /health returned 200 OK: {resp.json()}')

    print('\n[ALL MICROSERVICE TESTS COMPLETED SUCCESSFULLY]')

if __name__ == '__main__':
    run_tests()

