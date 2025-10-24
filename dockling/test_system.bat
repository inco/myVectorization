@echo off
echo Testing Library Vectorization RAG System...
echo.

REM Set Qdrant environment variables
set QDRANT_HOST=192.168.1.13
set QDRANT_PORT=6333
set QDRANT_API_KEY=4c77aa13-30ff-41e0-ac87-ab1e0c791d2e
set QDRANT_COLLECTION=market_documents

echo Environment variables set:
echo QDRANT_HOST=%QDRANT_HOST%
echo QDRANT_PORT=%QDRANT_PORT%
echo QDRANT_COLLECTION=%QDRANT_COLLECTION%
echo.

echo Running system test...
python test_system.py

echo.
echo If tests passed, you can now use:
echo.
echo 1. Vectorize documents:
echo    python -m src.cli vectorize ./documents --vector-store qdrant --collection-name market_documents
echo.
echo 2. Test a query:
echo    python -m src.cli query "What is machine learning?" --vector-store qdrant --collection-name market_documents
echo.
echo 3. Interactive Q&A:
echo    python -m src.cli ask --vector-store qdrant --collection-name market_documents
echo.

pause

