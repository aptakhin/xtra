"""Shared pytest configuration and fixtures."""

import pytest


@pytest.fixture(scope="module")
def vcr_config():
    """VCR configuration to filter sensitive data from recordings."""
    return {
        "filter_headers": [
            ("authorization", "Bearer REDACTED"),
            ("x-api-key", "REDACTED"),
            ("api-key", "REDACTED"),
            ("x-goog-api-key", "REDACTED"),
        ],
        "filter_post_data_parameters": [
            ("api_key", "REDACTED"),
        ],
        "record_mode": "once",
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }
