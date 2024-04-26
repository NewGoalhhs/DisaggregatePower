import os

from core.Database import Database


class Migrations:
    def __init__(self):
        # self.db = Database()
        self.db = None
        pass

    def get_migration_files(self, path='migrations', max_depth=5):
        migration_files = []

        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.py'):
                    migration_files.append(entry.name)
                if entry.is_dir():
                    migration_files += self.get_migration_files(path + '/' + entry.name, max_depth - 1)

        return migration_files

    def migrate(self):
        print('Migrating...')

        print('Scanning migrations directory...')
        # Scan migrations directory for migration files
        migration_files = self.get_migration_files()

        print('Found ' + str(len(migration_files)) + ' migration files')

        for migration_file in migration_files:
            print('Running migration file: ' + migration_file)

            # import the class from the file
            migration_import = __import__('migrations.' + migration_file[:-3], fromlist=['Migration'])
            migration_class = getattr(migration_import, migration_file[:-3])

            migration = migration_class(self.db)
            # run the migration
            migration.down()
            migration.migrate()
            migration.up()
            migration.migrate()

            print('Migration file: ' + migration_file + ' ran successfully')
