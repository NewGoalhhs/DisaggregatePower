class DatabaseOperations:
    SELECT_ALL = "SELECT * FROM {}"
    SELECT_WHERE = "SELECT * FROM {} WHERE {}"
    SELECT_MAX = "SELECT MAX({}) FROM {}"


class MigrationOperations:
    CREATE_MIGRATIONS_TABLE = '''
        CREATE TABLE Migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            migration_name TEXT,
            created DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        '''
    DROP_MIGRATIONS_TABLE = 'DROP TABLE Migrations'


class BuildingOperations:
    CREATE_BUILDING_TABLE = '''
        CREATE TABLE Building (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
            )
        '''
    DROP_BUILDING_TABLE = 'DROP TABLE Building'
    INSERT_BUILDING = 'INSERT INTO Building (name) VALUES ("{}")'
    GET_BUILDING_ID = 'SELECT id FROM Building WHERE name = "{}"'

class ApplianceOperations:
    CREATE_APPLIANCE_TABLE = '''
        CREATE TABLE Appliance (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
            )
        '''
    DROP_APPLIANCE_TABLE = 'DROP TABLE Appliance'
    INSERT_APPLIANCE = 'INSERT INTO Appliance (name) VALUES ("{}")'


class PowerUsageOperations:
    CREATE_POWER_USAGE_TABLE = '''
        CREATE TABLE PowerUsage (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            building_id INTEGER NOT NULL,
            datetime    DATETIME NOT NULL,
            power_usage REAL NOT NULL,
            FOREIGN KEY (building_id) REFERENCES Building (id)
            )
        '''
    CREATE_POWER_USAGE_INDEX = '''
        CREATE UNIQUE INDEX  powerusage_building_datetime_unique ON
        PowerUsage(building_id, datetime)
    '''
    DROP_POWER_USAGE_TABLE = 'DROP TABLE PowerUsage'
    INSERT_POWER_USAGE = 'INSERT INTO PowerUsage (building_id, datetime, power_usage) VALUES ("{}", "{}", "{}")'

class PowerUsageApplianceOperations:
    CREATE_POWER_USAGE_APPLIANCE_TABLE = '''
        CREATE TABLE PowerUsage_Appliance (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            PowerUsage_id   INTEGER NOT NULL,
            Appliance_id    INTEGER NOT NULL,
            appliance_power REAL  NOT NULL,
            FOREIGN KEY (PowerUsage_id) REFERENCES PowerUsage (id),
            FOREIGN KEY (Appliance_id) REFERENCES Appliance (id)
            )
        '''
    CREATE_POWER_USAGE_APPLIANCE_INDEX = '''
        CREATE UNIQUE INDEX powerusage_appliance_combination_unique ON
        PowerUsage_Appliance(PowerUsage_id, Appliance_id)
    '''
    DROP_POWER_USAGE_APPLIANCE_TABLE = 'DROP TABLE PowerUsage_Appliance'
    INSERT_POWER_USAGE_APPLIANCE = 'INSERT INTO PowerUsage_Appliance (PowerUsage_id, Appliance_id, appliance_power) VALUES ("{}", "{}", "{}")'