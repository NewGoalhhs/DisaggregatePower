import os

from core.Database import Database


class Migrations:
    def __init__(self):
        self.db = Database(True)

    def get_migration_files(self, path='migrations', max_depth=5):
        migration_files = []

        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.py'):
                    migration_files.append(path + '/' + entry.name)
                if entry.is_dir():
                    migration_files += self.get_migration_files(path + '/' + entry.name, max_depth - 1)

        return migration_files

    def migrate(self):
        print('Migrating...')

        print('Scanning migrations directory...')
        # Scan migrations directory for migration files
        migration_paths = self.get_migration_files()

        print('Found ' + str(len(migration_paths)) + ' migration files')

        for migration_path in migration_paths:
            print('Running migration file: ' + migration_path)

            migration_path = migration_path.replace('/', '.').replace('\\', '.')[0:-3]
            migration_file = migration_path.split('.')[-1]

            # import the class from the file
            migration_import = __import__(migration_path, fromlist=['Migration'])
            migration_class = getattr(migration_import, migration_file)

            migration = migration_class(self.db)
            # run the migration
            migration.down()
            migration.migrate()
            migration.up()
            migration.migrate()

            print('Migration file: ' + migration_path + ' ran successfully')
