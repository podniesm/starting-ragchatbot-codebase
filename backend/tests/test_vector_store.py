"""Tests for VectorStore and SearchResults - demonstrates MAX_RESULTS=0 bug"""
import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import VectorStore, SearchResults


class TestSearchResultsDataclass:
    """Tests for SearchResults dataclass"""

    def test_from_chroma_extracts_documents(self):
        """Test creating SearchResults from ChromaDB response"""
        chroma_response = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"key": "val1"}, {"key": "val2"}]],
            "distances": [[0.1, 0.2]]
        }

        results = SearchResults.from_chroma(chroma_response)

        assert results.documents == ["doc1", "doc2"]
        assert len(results.metadata) == 2
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_handles_empty_response(self):
        """Test handling empty ChromaDB response"""
        chroma_response = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        results = SearchResults.from_chroma(chroma_response)

        assert results.documents == []
        assert results.metadata == []
        assert results.is_empty()

    def test_empty_with_error_message(self):
        """Test creating empty results with error message"""
        results = SearchResults.empty("No course found")

        assert results.is_empty()
        assert results.error == "No course found"
        assert results.documents == []

    def test_is_empty_true_for_no_documents(self):
        """Test is_empty() returns True when no documents"""
        results = SearchResults(documents=[], metadata=[], distances=[])

        assert results.is_empty()

    def test_is_empty_false_when_has_documents(self):
        """Test is_empty() returns False when has documents"""
        results = SearchResults(
            documents=["doc"],
            metadata=[{"key": "value"}],
            distances=[0.1]
        )

        assert not results.is_empty()


class TestVectorStoreMaxResultsBug:
    """Tests demonstrating the MAX_RESULTS=0 bug"""

    def test_search_with_max_results_zero_returns_empty(self, tmp_path):
        """CRITICAL: Demonstrates the MAX_RESULTS=0 bug

        This test shows that when max_results=0 (as in the buggy config),
        ChromaDB returns zero results regardless of matching content.
        """
        # Create store with buggy max_results=0
        store = VectorStore(
            chroma_path=str(tmp_path / "chroma_zero"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=0  # Bug condition
        )

        # Add content
        store.course_content.add(
            documents=["Test content about Python programming"],
            metadatas=[{"course_title": "Test", "lesson_number": 0, "chunk_index": 0}],
            ids=["test_0"]
        )

        # Search - should find content but won't due to n_results=0
        results = store.search(query="Python")

        # BUG: With max_results=0, this returns empty!
        assert results.is_empty(), (
            "BUG CONFIRMED: max_results=0 causes empty results. "
            "Fix: Change MAX_RESULTS from 0 to 5 in config.py"
        )

    def test_search_with_positive_max_results_finds_content(self, tmp_path):
        """Test that search works correctly with max_results > 0"""
        # Create store with correct max_results
        store = VectorStore(
            chroma_path=str(tmp_path / "chroma_positive"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5  # Correct value
        )

        # Add content
        store.course_content.add(
            documents=["Test content about Python programming"],
            metadatas=[{"course_title": "Test", "lesson_number": 0, "chunk_index": 0}],
            ids=["test_0"]
        )

        # Search should find content
        results = store.search(query="Python")

        assert not results.is_empty(), "Search should find matching content"
        assert len(results.documents) >= 1


class TestVectorStoreSearch:
    """Tests for VectorStore.search() method"""

    def test_search_returns_search_results(self, test_vector_store, sample_course, sample_chunks):
        """Test that search returns SearchResults object"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        results = test_vector_store.search(query="Python")

        assert isinstance(results, SearchResults)

    def test_search_finds_relevant_content(self, test_vector_store, sample_course, sample_chunks):
        """Test that search finds semantically relevant content"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        results = test_vector_store.search(query="Python programming")

        assert not results.is_empty()
        # Should contain Python-related content
        found_python = any("Python" in doc for doc in results.documents)
        assert found_python

    def test_search_returns_metadata(self, test_vector_store, sample_course, sample_chunks):
        """Test that search returns metadata with results"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        results = test_vector_store.search(query="content")

        assert len(results.metadata) == len(results.documents)
        for meta in results.metadata:
            assert "course_title" in meta
            assert "lesson_number" in meta

    def test_search_with_course_filter(self, test_vector_store, sample_course, sample_chunks):
        """Test search with course name filter"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        results = test_vector_store.search(query="content", course_name="Test")

        if not results.is_empty():
            for meta in results.metadata:
                assert meta["course_title"] == "Test Course"

    def test_search_with_invalid_course_returns_error(self, test_vector_store):
        """Test that invalid course name returns error when no courses exist"""
        # Test with empty store - no courses to match
        results = test_vector_store.search(query="content", course_name="Any Course Name")

        assert results.is_empty()
        assert "No course found" in results.error

    def test_search_with_lesson_filter(self, test_vector_store, sample_course, sample_chunks):
        """Test search with lesson number filter"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        results = test_vector_store.search(query="content", lesson_number=0)

        if not results.is_empty():
            for meta in results.metadata:
                assert meta.get("lesson_number") == 0

    def test_search_with_limit_override(self, test_vector_store, sample_course, sample_chunks):
        """Test that limit parameter overrides max_results"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        results = test_vector_store.search(query="content", limit=1)

        assert len(results.documents) <= 1


class TestVectorStoreCourseResolution:
    """Tests for course name resolution"""

    def test_resolve_exact_course_name(self, test_vector_store, sample_course):
        """Test resolving exact course name"""
        test_vector_store.add_course_metadata(sample_course)

        resolved = test_vector_store._resolve_course_name("Test Course")

        assert resolved == "Test Course"

    def test_resolve_partial_course_name(self, test_vector_store, sample_course):
        """Test fuzzy matching with partial course name"""
        test_vector_store.add_course_metadata(sample_course)

        resolved = test_vector_store._resolve_course_name("Test")

        assert resolved == "Test Course"

    def test_resolve_nonexistent_course(self, test_vector_store):
        """Test that nonexistent course returns None"""
        resolved = test_vector_store._resolve_course_name("Nonexistent Course XYZ123")

        assert resolved is None


class TestVectorStoreCourseMetadata:
    """Tests for course metadata operations"""

    def test_add_course_metadata(self, test_vector_store, sample_course):
        """Test adding course metadata"""
        test_vector_store.add_course_metadata(sample_course)

        titles = test_vector_store.get_existing_course_titles()

        assert "Test Course" in titles

    def test_get_course_count(self, test_vector_store, sample_course):
        """Test getting course count"""
        assert test_vector_store.get_course_count() == 0

        test_vector_store.add_course_metadata(sample_course)

        assert test_vector_store.get_course_count() == 1

    def test_get_course_outline(self, test_vector_store, sample_course):
        """Test getting course outline"""
        test_vector_store.add_course_metadata(sample_course)

        outline = test_vector_store.get_course_outline("Test")

        assert outline is not None
        assert outline["title"] == "Test Course"
        assert len(outline["lessons"]) == 2

    def test_get_course_outline_nonexistent(self, test_vector_store):
        """Test getting outline for nonexistent course"""
        outline = test_vector_store.get_course_outline("Nonexistent")

        assert outline is None

    def test_get_lesson_link(self, test_vector_store, sample_course):
        """Test getting lesson link"""
        test_vector_store.add_course_metadata(sample_course)

        link = test_vector_store.get_lesson_link("Test Course", 0)

        assert link == "https://example.com/lesson0"

    def test_get_course_link(self, test_vector_store, sample_course):
        """Test getting course link"""
        test_vector_store.add_course_metadata(sample_course)

        link = test_vector_store.get_course_link("Test Course")

        assert link == "https://example.com/course"


class TestVectorStoreContentOperations:
    """Tests for course content operations"""

    def test_add_course_content(self, test_vector_store, sample_chunks):
        """Test adding course content chunks"""
        test_vector_store.add_course_content(sample_chunks)

        # Search should find the content
        results = test_vector_store.search(query="Python")

        assert not results.is_empty()

    def test_add_empty_chunks_list(self, test_vector_store):
        """Test that adding empty chunks list doesn't error"""
        test_vector_store.add_course_content([])

        # Should still work, just no content
        results = test_vector_store.search(query="anything")

        assert results.is_empty()

    def test_clear_all_data(self, test_vector_store, sample_course, sample_chunks):
        """Test clearing all data"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        assert test_vector_store.get_course_count() == 1

        test_vector_store.clear_all_data()

        assert test_vector_store.get_course_count() == 0


class TestVectorStoreFilterBuilding:
    """Tests for filter building logic"""

    def test_build_filter_with_course_only(self, test_vector_store):
        """Test building filter with course title only"""
        filter_dict = test_vector_store._build_filter("Test Course", None)

        assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_with_lesson_only(self, test_vector_store):
        """Test building filter with lesson number only"""
        filter_dict = test_vector_store._build_filter(None, 1)

        assert filter_dict == {"lesson_number": 1}

    def test_build_filter_with_both(self, test_vector_store):
        """Test building filter with both course and lesson"""
        filter_dict = test_vector_store._build_filter("Test Course", 1)

        assert "$and" in filter_dict
        conditions = filter_dict["$and"]
        assert {"course_title": "Test Course"} in conditions
        assert {"lesson_number": 1} in conditions

    def test_build_filter_with_neither(self, test_vector_store):
        """Test building filter with no parameters"""
        filter_dict = test_vector_store._build_filter(None, None)

        assert filter_dict is None
