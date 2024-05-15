

class BasePrintHelper:
    @staticmethod
    def get_color_code(color: str) -> str:
        color_codes = {
            'black': '\033[30m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'reset': '\033[0m'
        }
        return color_codes.get(color, '\033[0m')  # default to 'reset' if color not found