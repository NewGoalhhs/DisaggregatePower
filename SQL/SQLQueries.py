class DatabaseOperations:
    CREATE_DB = "CREATE DATABASE IF NOT EXISTS {}"
    SELECT_ALL = "SELECT * FROM {}"
    SELECT_WHERE = "SELECT * FROM {} WHERE {}"
    SELECT_MAX = "SELECT MAX({}) FROM {}"


class MigrationOperations:
    CREATE_MIGRATIONS_TABLE = '''
        CREATE TABLE Migrations (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `migration_name` VARCHAR(255),
            `created` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
    DROP_MIGRATIONS_TABLE = 'DROP TABLE IF EXISTS Migrations'


class BuildingOperations:
    CREATE_BUILDING_TABLE = '''
        CREATE TABLE Building (
            `id`   BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            `name` VARCHAR(255) NOT NULL
            )
        '''
    DROP_BUILDING_TABLE = 'DROP TABLE IF EXISTS Building'


class ApplianceOperations:
    CREATE_APPLIANCE_TABLE = '''
        CREATE TABLE Appliance (
            `id`   BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            `name` VARCHAR(255) NOT NULL
            )
        '''
    DROP_APPLIANCE_TABLE = 'DROP TABLE IF EXISTS Appliance'


class PowerUsageOperations:
    CREATE_POWER_USAGE_TABLE = '''
        CREATE TABLE PowerUsage (
            `id`        BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            `building_id` BIGINT NOT NULL,
            `datetime`    DATETIME NOT NULL,
            `power_usage` FLOAT NOT NULL,
            FOREIGN KEY (`building_id`) REFERENCES `Building` (`id`)
            )
        '''
    CREATE_POWER_USAGE_INDEX = '''
        CREATE UNIQUE INDEX  `powerusage_building_datetime_unique` ON
        `PowerUsage`(`building_id`, `datetime`)
    '''
    DROP_POWER_USAGE_TABLE = 'DROP TABLE IF EXISTS PowerUsage'


class PowerUsageApplianceOperations:
    CREATE_POWER_USAGE_APPLIANCE_TABLE = '''
        CREATE TABLE PowerUsage_Appliance (
            `id`             BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            `PowerUsage_id`   BIGINT NOT NULL,
            `Appliance_id`    BIGINT NOT NULL,
            `appliance_power` FLOAT  NOT NULL,
            FOREIGN KEY (`PowerUsage_id`) REFERENCES `PowerUsage` (`id`),
            FOREIGN KEY (`Appliance_id`) REFERENCES `Appliance` (`id`)
            )
        '''
    CREATE_POWER_USAGE_APPLIANCE_INDEX = '''
        CREATE UNIQUE INDEX `powerusage_appliance_combination_unique` ON
        `PowerUsage_Appliance`(`PowerUsage_id`, `Appliance_id`)
    '''
    DROP_POWER_USAGE_APPLIANCE_TABLE = 'DROP TABLE IF EXISTS PowerUsage_Appliance'
