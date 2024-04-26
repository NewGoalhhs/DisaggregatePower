from core.Migrations import Migrations
from dotenv import load_dotenv

migration = Migrations()
load_dotenv()

if __name__ == '__main__':
    migration.migrate()
