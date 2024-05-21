class DatabaseOperations:
    SELECT_ALL = "SELECT * FROM {}"
    SELECT_WHERE = "SELECT * FROM {} WHERE {} = {}"
    SELECT_MAX = "SELECT MAX({}) FROM {}"
    SELECT_APPLIANCE_THRESHOLD = """
        SELECT appliance_power
        FROM (
            SELECT appliance_power, 
                   COUNT(*) OVER() as total_rows, 
                   ROW_NUMBER() OVER(ORDER BY appliance_power) as row_number
            FROM PowerUsage_Appliance
            WHERE appliance_power > 0 AND Appliance_id = {}
        ) 
        WHERE row_number >= total_rows * 0.25
        ORDER BY appliance_power
        LIMIT 1
        """

    SELECT_POWER_USAGE_APPLIANCE_FOR_BUILDING = """
        SELECT PowerUsage_id, Appliance_id, appliance_power
        FROM PowerUsage_Appliance
        WHERE PowerUsage_id IN (
            SELECT id
            FROM PowerUsage
            WHERE building_id = {}
        )
        """

    SELECT_BUILDING_IDS_WITH_APPLIANCE = """
        SELECT DISTINCT building_id
        FROM PowerUsage
        WHERE id IN (
            SELECT PowerUsage_id
            FROM PowerUsage_Appliance
            WHERE Appliance_id = {}
        )
        """

    SELECT_POWER_USAGE_APPLIANCE_FOR_APPLIANCE = """
        SELECT PowerUsage_Appliance.appliance_power
        FROM PowerUsage_Appliance
        JOIN PowerUsage ON PowerUsage_Appliance.PowerUsage_id = PowerUsage.id
        WHERE Appliance_id = {}
        AND PowerUsage.timestamp BETWEEN '{}' AND '{}'
    """

    SELECT_ALL_JOIN = """
        SELECT * FROM {} JOIN {} ON {} = {}
    """


class MigrationOperations:
    CREATE_MIGRATIONS_TABLE = '''
        CREATE TABLE Migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            migration_name TEXT,
            created DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        '''
    DROP_MIGRATIONS_TABLE = 'DROP TABLE Migrations'
    INSERT_MIGRATION = 'INSERT INTO Migrations (migration_name) VALUES ("{}")'
    SELECT_MIGRATION = "SELECT * FROM Migrations WHERE migration_name = '{}'"


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


class IsUsingApplianceOperations:
    CREATE_IS_USING_APPLIANCE_TABLE = '''
        CREATE TABLE IsUsingAppliance (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            PowerUsage_id   INTEGER NOT NULL,
            Appliance_id    INTEGER NOT NULL,
            FOREIGN KEY (PowerUsage_id) REFERENCES PowerUsage (id),
            FOREIGN KEY (Appliance_id) REFERENCES Appliance (id)
            )
        '''
    CREATE_IS_USING_APPLIANCE_INDEX = '''
        CREATE UNIQUE INDEX isusingappliance_combination_unique ON
        IsUsingAppliance(PowerUsage_id, Appliance_id)
    '''
    DROP_IS_USING_APPLIANCE_TABLE = 'DROP TABLE IsUsingAppliance'
    INSERT_IS_USING_APPLIANCE = 'INSERT INTO IsUsingAppliance (PowerUsage_id, Appliance_id) VALUES ("{}", "{}")'
