import sqlite3

import pandas as pd


class SQLiteStorage:
    def __init__(self, filename):
        self.conn = sqlite3.connect(filename)
        self.path = filename

    def replace_records(self, table, table_name, **kwargs):
        table.to_sql(table_name, con=self.conn, if_exists='replace', index=False, method="multi", chunksize=1000, **kwargs)
        self.create_index_for_table(table, table_name)

    def add_records(self, table, table_name, create_index=None, **kwargs):
        table.to_sql(table_name, con=self.conn, if_exists='append', index=False, method="multi", chunksize=1000, **kwargs)
        if create_index is not None and create_index is not False:
            self.create_index_for_table(table, table_name, create_index=create_index)

    def create_index_for_table(self, table, table_name, create_index=True):
        if create_index is True:
            index_columns = (repr(col) for col in table.columns)
        else:
            index_columns = create_index

        self.create_index_for_columns(index_columns, table_name)

    def create_index_for_columns(self, columns, table_name):
        self.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name} 
            ON {table_name}({','.join(repr(col) for col in columns)})
            """
        )

    def query(self, query_string, **kwargs):
        return pd.read_sql(query_string, self.conn, **kwargs)

    def execute(self, query_string):
        self.conn.execute(query_string)
        self.conn.commit()

    def drop_table(self, table_name):
        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.conn.execute(f"DROP INDEX IF EXISTS idx_{table_name}")
        self.conn.commit()

    def __del__(self):
        self.conn.close()
        # if os.path.isfile(self.path):
        #     os.remove(self.path)