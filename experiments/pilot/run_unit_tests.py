"""Run the repository's unit tests and print one JSON object: test id -> ok, skip, fail or error.

The tests need two environments that cannot be combined: most need mini-swe-agent (the
Mini-SWE environment), two need torch and transformers (the training environment), and
transformers refuses the tokenizers version that mini-swe-agent's dependencies install.
preflight.py therefore runs everything in the Mini-SWE environment and then the tests that
could not import torch there in the training environment.

    python experiments/pilot/run_unit_tests.py                 # every test under tests/
    python experiments/pilot/run_unit_tests.py --only <id>...  # the named tests
"""
import argparse
import io
import json
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]


class Outcomes(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.outcomes, self.details = {}, {}

    def addSuccess(self, test):
        self.outcomes[test.id()] = 'ok'

    def addSkip(self, test, reason):
        self.outcomes[test.id()] = 'skip'

    def addFailure(self, test, err):
        self.outcomes[test.id()] = 'fail'
        self.details[test.id()] = self._exc_info_to_string(err, test)[-1500:]

    def addError(self, test, err):
        self.outcomes[test.id()] = 'error'
        self.details[test.id()] = self._exc_info_to_string(err, test)[-1500:]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    sys.path[:0] = [str(REPO / 'tests'), str(REPO)]
    loader = unittest.defaultTestLoader
    suite = loader.loadTestsFromNames(args.only) if args.only else loader.discover(str(REPO / 'tests'))
    result = Outcomes()
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()  # the tests print; keep the JSON clean
    try:
        suite.run(result)
    finally:
        sys.stdout, sys.stderr = stdout, stderr
    print(json.dumps({'outcomes': result.outcomes, 'details': result.details}))


if __name__ == '__main__':
    main()
