
import sqlite3
import pandas as pd 
from sqlalchemy import create_engine, Table, Column, Float, Integer, String, MetaData
from sqlalchemy.orm import sessionmaker, mapper
from sqlalchemy.ext.declarative import declarative_base
from time import time
import sys
import traceback
import os 
import re
t = time()

# function to load data
def load_data(file_name):
    file_name = str(file_name)
    # pd.read_csv(filepath_or_buffer)
    data = pd.read_csv('src/'+file_name, sep=",", header=[0], index_col=[0])

    return data

# function to create the table
def create_db_tables(sqlite_table_name, file_name):
    file_name = str(file_name)
    sqlite_table_name = str(sqlite_table_name)
    engine = create_engine("sqlite:///project_db.db", echo=True)
    sqlite_conn = engine.connect()
    data_load = load_data(file_name)
    return data_load.to_sql(sqlite_table_name, sqlite_conn, if_exists='fail')

if __name__ == 'main':
    t = time()

try:
    ''' create db and instert train and ideal data table into db'''
    while True:
        engine = create_engine("sqlite:///project_db.db", echo=True)
        sqlite_conn = engine.connect()
        try:
            load_data('train.csv')
            create_db_tables("training_data", "./train.csv") 
            load_data('ideal.csv')
            create_db_tables("ideal_function_data", "./ideal.csv") 
            break
        except ValueError:
            print("Deleting and recreating existing tables")
            sqlite_conn.close()
            os.remove("project_db.db")
except sqlite3.Error as er:
    print("SQLITE error: %s" %(" ".join(er.args)))
    print("SQLITE error class: " + er.__class__)

    exc_type, exc_value, exc_tb = sys.exc_info()
    print(traceback.format_exception(exc_type, exc_value, exc_tb))

finally:
    sqlite_conn.close()
    print("Time used to process db table "+ str(time() - t) +" secs")