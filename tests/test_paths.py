from __future__ import annotations

import unittest
from pathlib import Path

from pressuretrace.paths import data_dir, manifests_dir, repo_root, results_dir, splits_dir


class PathsTestCase(unittest.TestCase):
    def test_repo_root_resolves_project_root(self) -> None:
        root_dir = Path(__file__).resolve().parents[1]
        root = repo_root()
        self.assertEqual(root, root_dir)
        self.assertTrue((root / "pyproject.toml").exists())

    def test_common_directories_are_derived_from_root(self) -> None:
        root_dir = Path(__file__).resolve().parents[1]
        self.assertEqual(data_dir(), root_dir / "data")
        self.assertEqual(results_dir(), root_dir / "results")
        self.assertEqual(manifests_dir(), root_dir / "data" / "manifests")
        self.assertEqual(splits_dir(), root_dir / "data" / "splits")


if __name__ == "__main__":
    unittest.main()
