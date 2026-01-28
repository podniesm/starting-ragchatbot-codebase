"""Tests for configuration validation - catches MAX_RESULTS=0 bug"""

import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


class TestConfigValidation:
    """Tests to validate configuration and detect bugs"""

    def test_max_results_not_zero(self):
        """CRITICAL: MAX_RESULTS must be > 0 for searches to return results

        This test will FAIL if config.MAX_RESULTS is 0, which causes all
        ChromaDB searches to return zero results.
        """
        from config import config

        assert config.MAX_RESULTS > 0, (
            f"BUG DETECTED: MAX_RESULTS is {config.MAX_RESULTS}. "
            "This causes all ChromaDB searches to return zero results! "
            "Fix: Change MAX_RESULTS to 5 in config.py"
        )

    def test_max_results_reasonable_range(self):
        """MAX_RESULTS should be between 1 and 20 for reasonable performance"""
        from config import config

        assert (
            1 <= config.MAX_RESULTS <= 20
        ), f"MAX_RESULTS={config.MAX_RESULTS} is outside reasonable range [1, 20]"

    def test_chunk_size_positive(self):
        """CHUNK_SIZE must be positive for document processing"""
        from config import config

        assert config.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"

    def test_chunk_overlap_less_than_chunk_size(self):
        """CHUNK_OVERLAP must be less than CHUNK_SIZE"""
        from config import config

        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE, (
            f"CHUNK_OVERLAP ({config.CHUNK_OVERLAP}) must be less than "
            f"CHUNK_SIZE ({config.CHUNK_SIZE})"
        )

    def test_max_history_positive(self):
        """MAX_HISTORY must be positive for conversation context"""
        from config import config

        assert config.MAX_HISTORY > 0, "MAX_HISTORY must be positive"

    def test_anthropic_model_specified(self):
        """ANTHROPIC_MODEL must be specified"""
        from config import config

        assert config.ANTHROPIC_MODEL, "ANTHROPIC_MODEL must be specified"

    def test_embedding_model_specified(self):
        """EMBEDDING_MODEL must be specified"""
        from config import config

        assert config.EMBEDDING_MODEL, "EMBEDDING_MODEL must be specified"
