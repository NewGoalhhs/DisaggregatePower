from core.Migration import Migration
from SQL.SQLQueries import PowerUsageOperations as Query
from SQL.SQLQueries import BuildingOperations as BuildingQuery
from core.Database import Database
import pandas as pd


class M004PowerUsage(Migration):
    def up(self):
        self.add_sql(Query.CREATE_POWER_USAGE_TABLE)
        self.add_sql(Query.CREATE_POWER_USAGE_INDEX)

    def down(self):
        self.add_sql(Query.DROP_POWER_USAGE_TABLE)

    def insert(self, csv_path):
        data_frames = []
        building = ("REDDUS_" + csv_path.split('_')[1].split('.')[0])
        building_id = Database.query(BuildingQuery.GET_BUILDING_ID.format(building))[0][0]

        df = pd.read_csv(csv_path).dropna()

        df['time'] = pd.to_datetime(df['time'])

        df['building_id'] = building_id

        data_frames.append(df[['building_id', 'time', 'main']])

        final_df = pd.concat(data_frames, ignore_index=True)

        self.set_loading_bar_status('Retrieving data')
        self.set_loading_bar_goal(len(final_df)*2)

        for x in final_df.to_numpy():
            self.lb.update(1)
            record = tuple(x)
            self.add_sql(Query.INSERT_POWER_USAGE.format(*record))