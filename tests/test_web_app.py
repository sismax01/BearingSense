"""Unit tests for the FastAPI interactive web workbench routes."""

from __future__ import annotations

import unittest
from bearingsense.web_app import create_app


class TestWebAppConfiguration(unittest.TestCase):
    """Verify that the FastAPI application initializes and exposes expected endpoints."""

    def test_create_app_routes_and_health(self) -> None:
        app = create_app()
        route_paths = {getattr(r, "path", "") for r in app.routes}
        self.assertIn("/", route_paths)
        self.assertIn("/api/health", route_paths)
        self.assertIn("/analyze-synthetic", route_paths)
        self.assertIn("/analyze-upload", route_paths)
        self.assertTrue(hasattr(app.state, "model"))
        self.assertEqual(len(app.state.model.classes_), 4)


if __name__ == "__main__":
    unittest.main()
