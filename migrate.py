from core.Migrations import Migrations
from dotenv import load_dotenv

load_dotenv()
migrations = Migrations()

if __name__ == '__main__':
    migrations.migrate()
