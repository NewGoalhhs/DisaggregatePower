from abc import abstractmethod

from core.TestResult import TestResult


class Test:
    def __init__(self):
        self.results = []

    def assert_true(self, value, message):
        self.results.append(TestResult(value, message))

    def assert_false(self, value, message):
        self.results.append(TestResult(value, message))

    def get_results(self):
        results = self.results
        self.results = []
        return results