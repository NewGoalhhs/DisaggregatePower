from core.Migration import Migration


class M005PowerUsageAppliance(Migration):
    def up(self):
        self.add_sql('''
        CREATE TABLE `PowerUsage_Appliance` (
            `id`             BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            `PowerUsage_id`   BIGINT NOT NULL,
            `Appliance_id`    BIGINT NOT NULL,
            `appliance_power` FLOAT  NOT NULL,
            FOREIGN KEY (`PowerUsage_id`) REFERENCES `PowerUsage` (`id`),
            FOREIGN KEY (`Appliance_id`) REFERENCES `Appliance` (`id`)
            )
        ''')
        self.add_sql('''
            CREATE UNIQUE INDEX `powerusage_appliance_combination_unique` ON
            `PowerUsage_Appliance`(`PowerUsage_id`, `Appliance_id`)
        ''')

    def down(self):
        self.add_sql('DROP TABLE IF EXISTS PowerUsage_Appliance')
        