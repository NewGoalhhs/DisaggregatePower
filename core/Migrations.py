import os

from features.Classify.IsUsingAppliance import IsUsingAppliance

from core.Database import Database
from os import listdir
from os.path import isfile, join

from helper.LoadingBarHelper import LoadingBarHelper


class Migrations:
    def __init__(self):
        self.db = Database()

    def scan_migration_directory(self, path='migrations', max_depth=5, lb=None):
        migration_files = []
        if lb:
            lb.set_goal(lb.goal + len(list(os.scandir(path))))
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if lb:
                    lb.update(index)
                if entry.is_file() and entry.name.endswith('.py'):
                    migration_files.append(path + '/' + entry.name)
                if entry.is_dir():
                    migration_files += self.scan_migration_directory(path + '/' + entry.name, max_depth - 1, lb=lb)

        return migration_files

    def get_migration_files(self):

        lb = LoadingBarHelper('Scanning migrations directory', 100, 0)

        migration_files = self.scan_migration_directory(lb=lb)

        lb.finish()

        migration_files.sort(
            key=lambda x: int(x.split('/')[-1][1:4]) if x.split('/')[-1][1:4].isdigit() else 0
        )

        return migration_files

    def migrate(self):
        print('Migrating...')

        # Scan migrations directory for migration files
        migration_paths = self.get_migration_files()

        print('Found ' + str(len(migration_paths)) + ' migration files')

        migrations = []

        for migration_path in migration_paths:
            lb = LoadingBarHelper(migration_path, 1, 0)

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
                lb.update(1)
                migration.up()
                migration.migrate()
            except Exception as e:
                print('Migration file: ' + migration_path + ' failed with error: ' + str(e))
            lb.finish()

            migrations.append(migration)

        for file in self.get_files():
            print()
            for migration in migrations:
                migration.add_loading_bar(LoadingBarHelper(file + ': ' + migration.__class__.__name__, 1, 0))
                migration.reset_queries()
                migration.insert(file)
                migration.migrate()

        IsUsingAppliance(self.db).up()
        IsUsingAppliance(self.db).insert()

    def get_files(self):
        directory = 'Data/'

        # load all the files in the directory
        return [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
