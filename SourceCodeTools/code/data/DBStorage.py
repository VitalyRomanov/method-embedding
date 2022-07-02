import os
import sqlite3
from enum import Enum

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Boolean, Date, String, Integer


class AbstractDBStorage:
    class DataTypes(Enum):
        INT_PRIMARY = 0
        INT_NOT_NULL = 1
        INT = 2
        TEXT = 3
        TEXT_NOT_NULL = 4

    DataTypesDecoder = dict()
    # class DataTypesDecoder(Enum):
    #     pass
    conn = None

    @classmethod
    def get_storage_file_name(cls, path):
        raise NotImplementedError()

    @classmethod
    def verify_imported(cls, path):
        raise NotImplementedError()

    @classmethod
    def add_import_completed_flag(cls, path):
        raise NotImplementedError()

    def create_index_for_table(self, table, table_name, create_index=True):
        if create_index is True:
            index_columns = (repr(col) for col in table.columns)
        else:
            index_columns = create_index

        self.create_index_for_columns(index_columns, table_name)

    def create_index_for_columns(self, columns, table_name):
        raise NotImplementedError()

    def replace_records(self, table, table_name, **kwargs):
        if "dtype" in kwargs:
            kwargs["dtype"] = self.decode_dtypes(kwargs["dtype"])
        table.to_sql(table_name, con=self.conn, if_exists='replace', index=False, method="multi", chunksize=1000, **kwargs)
        self.create_index_for_table(table, table_name)

    def add_records(self, table, table_name, create_index=None, **kwargs):
        if "dtype" in kwargs:
            kwargs["dtype"] = self.decode_dtypes(kwargs["dtype"])
        table.to_sql(table_name, con=self.conn, if_exists='append', index=False, method="multi", chunksize=1000, **kwargs)
        if create_index is not None and create_index is not False:
            self.create_index_for_table(table, table_name, create_index=create_index)

    def query(self, query_string, **kwargs):
        return pd.read_sql(query_string, self.conn, **kwargs)

    def commit(self):
        self.conn.commit()

    def execute(self, query_string, commit=True):
        self.conn.execute(query_string)
        if commit:
            self.commit()

    def drop_table(self, table_name):
        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.conn.execute(f"DROP INDEX IF EXISTS idx_{table_name}")
        self.commit()

    @classmethod
    def decode_dtypes(cls, dtypes):
        return {
            key: cls.DataTypesDecoder[val] for key, val in dtypes.items()
        }


class SQLiteStorage(AbstractDBStorage):
    DataTypesDecoder = {
        AbstractDBStorage.DataTypes.INT_PRIMARY: "INT PRIMARY",
        AbstractDBStorage.DataTypes.INT_NOT_NULL: "INT NOT NULL",
        AbstractDBStorage.DataTypes.INT: "INT",
        AbstractDBStorage.DataTypes.TEXT: "TEXT",
        AbstractDBStorage.DataTypes.TEXT_NOT_NULL: "TEXT NOT NULL"
    }

    def __init__(self, filename):
        self.conn = sqlite3.connect(filename)
        self.path = filename

    @classmethod
    def get_storage_file_name(cls, path):
        return os.path.join(path, "dataset.db")

    @classmethod
    def verify_imported(cls, path):
        return os.path.isfile(path)

    @classmethod
    def add_import_completed_flag(cls, path):
        pass

    def create_index_for_columns(self, columns, table_name):
        self.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name} 
            ON {table_name}({",".join(repr(col) for col in columns)})
            """
        )

    def __del__(self):
        self.conn.close()


class PostgresStorage(AbstractDBStorage):
    DataTypesDecoder = {
        AbstractDBStorage.DataTypes.INT_PRIMARY: Integer,
        AbstractDBStorage.DataTypes.INT_NOT_NULL: Integer,
        AbstractDBStorage.DataTypes.INT: Integer,
        AbstractDBStorage.DataTypes.TEXT: String,
        AbstractDBStorage.DataTypes.TEXT_NOT_NULL: String
    }

    def __init__(self, hostname):
        self.conn = create_engine(f"postgresql://sct:sct@{hostname}:5432/postgres")
        self.path = hostname

    @classmethod
    def get_storage_file_name(cls, path):
        return os.path.join(path, "dataset_imported")

    @classmethod
    def verify_imported(cls, path):
        if os.path.isfile(path):
            with open(path, "r") as db_file:
                assert db_file.read().startswith("Import complete"), "Import was interrupted. Delete existing data"
            return True
        return False

    @classmethod
    def add_import_completed_flag(cls, path):
        with open(path, "w") as db_file:
            db_file.write("Import complete")

    def create_index_for_columns(self, columns, table_name):
        self.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name} 
            ON {table_name}({",".join(col for col in columns)})
            """
        )

    def commit(self):
        pass
