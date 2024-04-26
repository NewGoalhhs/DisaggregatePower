from core.Migration import Migration


class M004PowerUsage(Migration):
    def up(self):
        self.add_sql('''
        CREATE TABLE PowerUsage (
            `id`        BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            `building_id` BIGINT NOT NULL,
            `datetime`    DATETIME NOT NULL,
            `power_usage` FLOAT NOT NULL,
            FOREIGN KEY (`building_id`) REFERENCES `Building` (`id`)
            )
        ''')
        self.add_sql('''
            CREATE UNIQUE INDEX  `powerusage_building_datetime_unique` ON
            `PowerUsage`(`building_id`, `datetime`)
        ''')

    def down(self):
        self.add_sql('DROP TABLE IF EXISTS PowerUsage')
        