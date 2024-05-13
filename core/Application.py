from helper.PrintHelper import PrintHelper


class Application:
    def __init__(self):
        app = self
        pass

    def run(self):
        p = PrintHelper()

        p.print_heading('Welcome to the application')

        p.open_options()
        p.add_option('First option', lambda: p.print_line('First option selected'))
        p.add_option('Second option', lambda: p.print_line('Second option selected'))
        p.choose_option()

        p.reset_lines()
        p.print_heading('Second page')
