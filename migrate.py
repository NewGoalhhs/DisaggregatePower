from core.Migrations import Migrations
from dotenv import load_dotenv

load_dotenv()
migration = Migrations()

if __name__ == '__main__':
    migration.migrate()
