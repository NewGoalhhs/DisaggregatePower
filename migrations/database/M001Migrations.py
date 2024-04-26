from core.Migration import Migration


class M001Migrations(Migration):
    def up(self):
        self.add_sql('''
        CREATE TABLE Migrations (
            `id` INT AUTO_INCREMENT PRIMARY KEY, 
            `migration_name` VARCHAR(255),
            `created` TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
            )
        ''')

    def down(self):
        self.add_sql('DROP TABLE IF EXISTS Migrations')
