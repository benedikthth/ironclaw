"""Tests for the general shell tool (the PhD's hands for any system)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.phd import default_tools  # noqa: E402
from ironclaw.tools.base import ToolContext  # noqa: E402
from ironclaw.tools.shell import ShellExec  # noqa: E402


class TestShell(unittest.TestCase):
    def test_runs_a_command_and_returns_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = ShellExec().run({"command": "echo hello"}, ToolContext(workspace=tmp))
        self.assertIn("hello", out)
        self.assertIn("[exit 0]", out)

    def test_runs_in_the_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "marker.txt"), "w") as fh:
                fh.write("x")
            out = ShellExec().run({"command": "ls"}, ToolContext(workspace=tmp))
        self.assertIn("marker.txt", out)

    def test_nonzero_exit_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = ShellExec().run({"command": "exit 3"}, ToolContext(workspace=tmp))
        self.assertIn("[exit 3]", out)

    def test_shell_is_in_the_phd_toolset(self):
        self.assertIn("shell", {t.name for t in default_tools()})


if __name__ == "__main__":
    unittest.main()
