import unittest


class TestHarnessSmoke(unittest.TestCase):
    """Proves `python3 -m unittest discover -s _substrate/tests -p 'test_*.py'`
    runs green. This is a non-package tests dir (no __init__.py): discovery uses
    -s _substrate/tests and each real test prepends _substrate/ to sys.path."""

    def test_discovery_runs_green(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
