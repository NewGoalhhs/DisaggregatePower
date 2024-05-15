from core.Generate import Generate
from model.Appliance import Appliance


class SimpleGenerate(Generate):
    def __init__(self):
        super().__init__()

    def run(self, p):
        appliances = Appliance.fetch_all()

        p.print_heading('Appliances')

        for appliance in appliances:
            p.print_line(appliance.name)
        pass
