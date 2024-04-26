from core.Migration import Migration


class M003Appliance(Migration):
    def up(self):
        self.add_sql('''
        CREATE TABLE Appliance (
            `id`   BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            `name` VARCHAR(255) NOT NULL
            )
        ''')

    def down(self):
        self.add_sql('DROP TABLE IF EXISTS Appliance')
        