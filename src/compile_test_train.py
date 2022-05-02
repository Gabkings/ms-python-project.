import sqlite3
from sqlalchemy import create_engine
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pt 

# contrusting the main figure with 2 plots
fig, axs = pt.plot(2, figsize=(12,8))

# db connection

conn = sqlite3.connect("project_db.db")
cursor = conn.cursor()

# building plots for training dataset 
sql = "SELECT * FROM training_data"
cursor.execute(sql)
training_data = np.array(cursor.fetchall())
x = training_data[:,0]
axs[0].set_title("Training data")
for i in range(1, len(training_data[0,:]), 1):
    y = training_data[:, i]
    axs[0].plot(x,y, label='{}'.format(i))
axs[0].legend()

# building plots for training dataset 
sql = "SELECT * FROM ideal_function_data"
cursor.execute(sql)
ideal_data = np.array(cursor.fetchall())
x = ideal_data[:,0]
axs[0].set_title("Ideal functions")
for i in range(1, len(ideal_data[0,:]), 1):
    y = ideal_data[:, i]
    axs[1].plot(x,y, label='{}'.format(i))
pt.savefig("figure-1.png")
pt.show()

