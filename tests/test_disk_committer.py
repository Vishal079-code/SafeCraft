"""Tests for Atomic Disk Committer module."""

import unittest
import tempfile
import shutil
from pathlib import Path
from agent.disk_committer import commit_staged_workspace_to_disk, DiskCommitError


class TestDiskCommitter(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_normal_write(self):
        staged = {
            "main.py": "print('hello')",
            "auth/login.py": "def login(): pass"
        }
        success, count = commit_staged_workspace_to_disk(staged, self.test_dir)
        self.assertTrue(success)
        self.assertEqual(count, 2)

        main_file = Path(self.test_dir) / "main.py"
        auth_file = Path(self.test_dir) / "auth" / "login.py"
        self.assertTrue(main_file.exists())
        self.assertEqual(main_file.read_text(encoding="utf-8"), "print('hello')")
        self.assertTrue(auth_file.exists())
        self.assertEqual(auth_file.read_text(encoding="utf-8"), "def login(): pass")

    def test_absolute_path_attack_blocked(self):
        staged = {
            "main.py": "print(1)",
            "/etc/passwd": "malicious content"
        }
        with self.assertRaises(DiskCommitError):
            commit_staged_workspace_to_disk(staged, self.test_dir)

    def test_traversal_attack_blocked(self):
        staged = {
            "main.py": "print(1)",
            "../../../../tmp/evil.txt": "malicious content"
        }
        with self.assertRaises(DiskCommitError):
            commit_staged_workspace_to_disk(staged, self.test_dir)

    def test_atomic_failure_no_partial_write(self):
        staged = {
            "safe_new_file.py": "print(2)",
            "/etc/shadow": "attack"
        }
        with self.assertRaises(DiskCommitError):
            commit_staged_workspace_to_disk(staged, self.test_dir)

        safe_file = Path(self.test_dir) / "safe_new_file.py"
        self.assertFalse(safe_file.exists())


if __name__ == "__main__":
    unittest.main()