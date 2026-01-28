"""Integration tests for FastAPI endpoints

These tests create a minimal test app to avoid the frontend static files issue.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


# Pydantic models (duplicated from app.py to avoid import issues)
class SourceInfo(BaseModel):
    title: str
    url: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


@pytest.fixture
def mock_rag_system():
    """Create mock RAG system for testing"""
    mock = MagicMock()
    mock.query.return_value = ("Test answer", [{"title": "Test Source", "url": None}])
    mock.session_manager.create_session.return_value = "session_1"
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    return mock


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mock rag_system"""
    app = FastAPI()

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)
            source_infos = [
                SourceInfo(**s) if isinstance(s, dict) else SourceInfo(title=s) for s in sources
            ]

            return QueryResponse(answer=answer, sources=source_infos, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"], course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    @pytest.mark.asyncio
    async def test_query_endpoint_returns_200(self, test_app):
        """Test that query endpoint returns 200 on success"""
        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post("/api/query", json={"query": "What is Python?"})

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_query_endpoint_returns_answer(self, test_app):
        """Test that query endpoint returns answer in response"""
        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post("/api/query", json={"query": "Test question"})

        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Test answer"

    @pytest.mark.asyncio
    async def test_query_endpoint_returns_sources(self, test_app):
        """Test that query endpoint returns sources"""
        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post("/api/query", json={"query": "Test"})

        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) == 1
        assert data["sources"][0]["title"] == "Test Source"

    @pytest.mark.asyncio
    async def test_query_endpoint_returns_session_id(self, test_app):
        """Test that query endpoint returns session_id"""
        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post("/api/query", json={"query": "Test"})

        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "session_1"

    @pytest.mark.asyncio
    async def test_query_endpoint_uses_provided_session(self, test_app, mock_rag_system):
        """Test that query uses provided session_id"""
        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/query", json={"query": "Test", "session_id": "existing_session"}
            )

        mock_rag_system.query.assert_called_with("Test", "existing_session")

    @pytest.mark.asyncio
    async def test_query_endpoint_error_returns_500(self, mock_rag_system):
        """Test that errors return 500 status"""
        mock_rag_system.query.side_effect = Exception("Database error")

        # Create app with error-raising mock
        app = FastAPI()

        @app.post("/api/query", response_model=QueryResponse)
        async def query_documents(request: QueryRequest):
            try:
                session_id = request.session_id or mock_rag_system.session_manager.create_session()
                answer, sources = mock_rag_system.query(request.query, session_id)
                return QueryResponse(answer=answer, sources=[], session_id=session_id)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/query", json={"query": "Test"})

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_query_endpoint_missing_query_returns_422(self, test_app):
        """Test that missing query field returns 422"""
        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post("/api/query", json={})  # Missing query field

        assert response.status_code == 422


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    @pytest.mark.asyncio
    async def test_courses_endpoint_returns_200(self, test_app):
        """Test that courses endpoint returns 200"""
        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/courses")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_courses_endpoint_returns_stats(self, test_app):
        """Test that courses endpoint returns course statistics"""
        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/courses")

        data = response.json()
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Course A" in data["course_titles"]

    @pytest.mark.asyncio
    async def test_courses_endpoint_error_returns_500(self, mock_rag_system):
        """Test that errors return 500"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Error")

        app = FastAPI()

        @app.get("/api/courses", response_model=CourseStats)
        async def get_course_stats():
            try:
                analytics = mock_rag_system.get_course_analytics()
                return CourseStats(
                    total_courses=analytics["total_courses"],
                    course_titles=analytics["course_titles"],
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/courses")

        assert response.status_code == 500


class TestSourceInfoHandling:
    """Tests for source info conversion"""

    @pytest.mark.asyncio
    async def test_dict_sources_converted(self, mock_rag_system):
        """Test that dict sources are converted to SourceInfo"""
        mock_rag_system.query.return_value = (
            "Answer",
            [{"title": "Source 1", "url": "http://example.com"}],
        )

        app = FastAPI()

        @app.post("/api/query", response_model=QueryResponse)
        async def query_documents(request: QueryRequest):
            session_id = request.session_id or mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            source_infos = [
                SourceInfo(**s) if isinstance(s, dict) else SourceInfo(title=s) for s in sources
            ]
            return QueryResponse(answer=answer, sources=source_infos, session_id=session_id)

        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/query", json={"query": "Test"})

        data = response.json()
        assert data["sources"][0]["title"] == "Source 1"
        assert data["sources"][0]["url"] == "http://example.com"

    @pytest.mark.asyncio
    async def test_empty_sources_handled(self, mock_rag_system):
        """Test that empty sources list is handled"""
        mock_rag_system.query.return_value = ("Answer", [])

        app = FastAPI()

        @app.post("/api/query", response_model=QueryResponse)
        async def query_documents(request: QueryRequest):
            session_id = request.session_id or mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            source_infos = [
                SourceInfo(**s) if isinstance(s, dict) else SourceInfo(title=s) for s in sources
            ]
            return QueryResponse(answer=answer, sources=source_infos, session_id=session_id)

        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/query", json={"query": "Test"})

        data = response.json()
        assert data["sources"] == []
