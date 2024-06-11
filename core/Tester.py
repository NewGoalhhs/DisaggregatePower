import os
import time
from datetime import datetime

from app import __ROOT__
from core.Test import Test
from core.TestResult import TestResult
from helper.PrintHelper import PrintHelper


class Tester:
    def __init__(self):
        self.test_results = []
        self.p = PrintHelper()

    def get_tests(self, path='tests', depth=5):
        if depth < 0:
            return []

        with os.scandir(__ROOT__+'/'+path) as entries:
            tests = []
            for entry in entries:
                new_path = path+'/'+entry.name
                if entry.name == '__pycache__':
                    continue
                elif entry.is_dir():
                    tests += self.get_tests(new_path, depth-1)
                else:
                    tests.append(self.get_class_from_path(new_path))
            return tests

    def get_class_from_path(self, path):
        module = path.replace('/', '.').replace('.py', '')
        class_name = path.split('/')[-1].split('.')[0]
        _module_ = __import__(module, fromlist=[class_name])
        _class_ = getattr(_module_, class_name)
        return _class_

    def handle_result(self, result):
        if isinstance(result, Exception):
            test_result = TestResult(False, str(result), result)
            self.test_results.append(test_result)
            return test_result
        elif isinstance(result, bool):
            message = "Test passed" if result else "Test failed"
            test_result = TestResult(result, message)
            self.test_results.append(test_result)
            return test_result
        elif isinstance(result, TestResult):
            self.test_results.append(result)
            return result
        elif isinstance(result, list):
            for res in result:
                res = self.handle_result(res)
                if not res.success:
                    return res
            return TestResult(True, "All tests passed")

    def run_test(self, test, test_instance):
        # get all methods from tests
        methods = [method for method in dir(test) if callable(getattr(test, method)) and method.startswith("test")]
        # run all methods
        for method in methods:
            try:
                test_method = getattr(test_instance, method)()
                result = self.handle_result(test_instance.get_results())

                result.set_method(method)
                result.set_cls(test.__name__)
                self.print_result(result)
            except Exception as e:
                raise e
                result = self.handle_result(e)
                result.set_method(method)
                result.set_cls(test.__name__)
                self.print_result(result)

    def get_test_instance(self, test):
        try:
            test_instance = test()
            return test_instance
        except Exception as e:
            raise e
            result = self.handle_result(e)
            result.set_cls(test.__name__)
            return e

    def print_result(self, result: TestResult):
        class_method_str = f"{result.cls}.{result.method}"
        if result.success:
            self.p.print_line(f"{class_method_str} - {self.p.get_color_code('green')}Test passed: {result.message}{self.p.get_color_code('reset')}")
        else:
            self.p.print_line(f"{class_method_str} - {self.p.get_color_code('red')}Test failed: {result.message}{self.p.get_color_code('reset')}")


    def run(self):
        self.p.print_line()
        self.p.print_line()
        self.p.print_line()
        tests = self.get_tests()
        start_time = time.time()
        for test in tests:
            test_instance = self.get_test_instance(test)
            if isinstance(test_instance, Test):
                self.p.print_line()
                self.p.print_line(f"Running test: {test.__name__} - {self.p.get_color_code('green')}Initialization successful {self.p.get_color_code('reset')}")
                self.p.print_cut()
                self.run_test(test, test_instance)
                self.p.print_cut()
            else:
                self.p.print_line(f"Could not run test: {test.__name__} - {self.p.get_color_code('red')}Initialization failed {self.p.get_color_code('reset')}")
                self.p.print_line(f"Error: {test_instance}")
                self.p.print_cut()
        self.p.print_line()
        self.print_summary(time_taken=time.time()-start_time)

    def print_summary(self, time_taken: float = 0.0):
        total_results = len(self.test_results)
        total_passed = len([result for result in self.test_results if result.success])
        time_taken_str = f"{time_taken:.2f} seconds"
        results_to_passed = f"{total_passed}/{total_results}"
        if total_passed == total_results:
            self.p.print_line(f"{self.p.get_color_code('green')}All tests passed!! {self.p.get_color_code('reset')}  {results_to_passed}   {time_taken_str}")
        else:
            self.p.print_line(f"{self.p.get_color_code('red')}Some tests failed!! {self.p.get_color_code('reset')}  {results_to_passed}   {time_taken_str}")