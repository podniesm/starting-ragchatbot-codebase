"""Tests for FastAPI API endpoints

This module tests the /api/query and /api/courses endpoints.
Uses a test app defined inline to avoid import issues with static file mounts.
"""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional
from unittest.mock import MagicMock


# =============================================================================
# Pydantic models (mirrored from app.py to avoid import issues)
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None


class SourceInfo(BaseModel):
    """Source information with optional URL"""
    title: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[SourceInfo]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]


# =============================================================================
# Test App Factory
# =============================================================================

def create_test_app(mock_rag_system: MagicMock) -> FastAPI:
    """
    Create a test FastAPI app with the same endpoints as app.py.
    Uses dependency injection for the RAG system to enable mocking.
    """
    app = FastAPI(title="Course Materials RAG System - Test")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            source_infos = [
                SourceInfo(**s) if isinstance(s, dict) else SourceInfo(title=s)
                for s in sources
            ]

            return QueryResponse(
                answer=answer,
                sources=source_infos,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Root endpoint for health check"""
        return {"status": "ok", "message": "Course Materials RAG System"}

    return app


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def test_client(mock_rag_system):
    """Create test client with mocked RAG system"""
    app = create_test_app(mock_rag_system)
    return TestClient(app)


@pytest.fixture
def test_client_error(mock_rag_system_error):
    """Create test client with RAG system that raises errors"""
    app = create_test_app(mock_rag_system_error)
    return TestClient(app)


# =============================================================================
# POST /api/query Tests
# =============================================================================

class TestQueryEndpoint:
    """Tests for the /api/query endpoint"""

    def test_query_without_session_creates_new_session(
        self, test_client, mock_rag_system
    ):
        """Query without session_id should create a new session"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_existing_session(self, test_client, mock_rag_system):
        """Query with session_id should use existing session"""
        response = test_client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": "my-session"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "my-session"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("Tell me more", "my-session")

    def test_query_returns_answer_and_sources(self, test_client):
        """Query should return answer and sources in correct format"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test answer about Python."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["title"] == "Test Course - Lesson 1"
        assert data["sources"][0]["url"] == "https://example.com/lesson1"

    def test_query_handles_string_sources(self, test_client, mock_rag_system):
        """Query should handle sources as plain strings"""
        mock_rag_system.query.return_value = (
            "Answer",
            ["Source 1", "Source 2"]  # Plain strings instead of dicts
        )

        response = test_client.post(
            "/api/query",
            json={"query": "test"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["title"] == "Source 1"
        assert data["sources"][0]["url"] is None

    def test_query_validation_missing_query(self, test_client):
        """Query endpoint should reject request without query field"""
        response = test_client.post(
            "/api/query",
            json={"session_id": "test"}
        )

        assert response.status_code == 422  # Validation error

    def test_query_validation_empty_query(self, test_client):
        """Query endpoint should accept empty query string"""
        # Note: Empty string is technically valid per the schema
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )

        assert response.status_code == 200

    def test_query_error_returns_500(self, test_client_error):
        """Query endpoint should return 500 on RAG system error"""
        response = test_client_error.post(
            "/api/query",
            json={"query": "test"}
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "RAG system error" in data["detail"]


# =============================================================================
# GET /api/courses Tests
# =============================================================================

class TestCoursesEndpoint:
    """Tests for the /api/courses endpoint"""

    def test_get_courses_returns_stats(self, test_client):
        """Courses endpoint should return course statistics"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Python Basics", "Machine Learning"]

    def test_get_courses_calls_analytics(self, test_client, mock_rag_system):
        """Courses endpoint should call get_course_analytics"""
        test_client.get("/api/courses")
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_courses_error_returns_500(self, test_client_error):
        """Courses endpoint should return 500 on error"""
        response = test_client_error.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Analytics error" in data["detail"]


# =============================================================================
# Root Endpoint Tests
# =============================================================================

class TestRootEndpoint:
    """Tests for the root endpoint"""

    def test_root_returns_status(self, test_client):
        """Root endpoint should return status ok"""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# =============================================================================
# Request/Response Format Tests
# =============================================================================

class TestRequestResponseFormats:
    """Tests for request/response format handling"""

    def test_query_accepts_json_content_type(self, test_client):
        """Query endpoint should accept JSON content type"""
        response = test_client.post(
            "/api/query",
            json={"query": "test"},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200

    def test_query_response_is_json(self, test_client):
        """Query endpoint should return JSON response"""
        response = test_client.post(
            "/api/query",
            json={"query": "test"}
        )
        assert response.headers["content-type"] == "application/json"

    def test_courses_response_is_json(self, test_client):
        """Courses endpoint should return JSON response"""
        response = test_client.get("/api/courses")
        assert response.headers["content-type"] == "application/json"

    def test_invalid_json_returns_error(self, test_client):
        """Invalid JSON should return validation error"""
        response = test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
