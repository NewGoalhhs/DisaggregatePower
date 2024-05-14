from core.Generate import Generate


class SimpleGenerate(Generate):
    def __init__(self):
        super().__init__()

    def run(self, p):
        print('Simple generate')
        pass
