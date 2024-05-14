import os

from core.Database import Database
from os import listdir
from os.path import isfile, join
class Migrations:
    def __init__(self):
        self.db = Database()

    def get_migration_files(self, path='migrations', max_depth=5):
        migration_files = []

        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.py'):
                    migration_files.append(path + '/' + entry.name)
                if entry.is_dir():
                    migration_files += self.get_migration_files(path + '/' + entry.name, max_depth - 1)

        migration_files.sort(
            key=lambda x: int(x.split('/')[-1][1:4]) if x.split('/')[-1][1:4].isdigit() else 0
        )

        return migration_files

    def migrate(self):
        print('Migrating...')

        print('Scanning migrations directory...')
        # Scan migrations directory for migration files
        migration_paths = self.get_migration_files()

        print('Found ' + str(len(migration_paths)) + ' migration files')

        migrations = []

        for migration_path in migration_paths:
            print()
            print('Running migration file: ' + migration_path)

            migration_path = migration_path.replace('/', '.').replace('\\', '.')[0:-3]
            migration_file = migration_path.split('.')[-1]

            # import the class from the file
            migration_import = __import__(migration_path, fromlist=['Migration'])
            migration_class = getattr(migration_import, migration_file)

            migration = migration_class(self.db)

            # Not used for migrations maybe for later
            # try:
            #     migration.down()
            #     migration.migrate()
            # except Exception as e:
            #     print('Migration file: ' + migration_path + ' failed with error: ' + str(e))

            try:
                migration.up()
                migration.migrate()
            except Exception as e:
                print('Migration file: ' + migration_path + ' failed with error: ' + str(e))

            migrations.append(migration)

        for file in self.get_files():
            print('Inserting data from file: ' + file + ' into database')
            for migration in migrations:
                print('Inserting data into database using migration: ' + str(migration))
                migration.reset_queries()
                migration.insert(file)
                migration.migrate()

    def get_files(self):
        dir = 'Data/'

        # load all the files in the directory
        return [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
