

class TestResult:
    def __init__(self, success: bool, message: str = '', exception: Exception = None):
        self.cls = 'Unknown'
        self.method = 'Unknown'
        self.success = success
        self.message = message
        self.exception = exception

    def set_cls(self, cls):
        self.cls = cls

    def set_method(self, method):
        self.method = method

    def success(self):
        return self.success

    def get_exception(self):
        return self.exception

