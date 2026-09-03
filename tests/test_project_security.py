import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules import state as state_module


class ProjectPathSecurityTests(unittest.TestCase):
    def test_accepts_normal_project_names(self):
        self.assertEqual(state_module.normalize_project_name("Film 01_PL"), "Film 01_PL")

    def test_rejects_path_traversal(self):
        for name in ("../secret", "folder/name", "folder\\name", ".", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                state_module.normalize_project_name(name)

    def test_project_path_stays_below_project_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(state_module, "PROJECTS_DIR", temp_dir):
                result = state_module.project_file_path("Safe Project")
                self.assertEqual(result.parent, Path(temp_dir).resolve())


if __name__ == "__main__":
    unittest.main()
