import sqlite3
from sqlalchemy import create_engine
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pt 

# contrusting the main figure with 2 plots
fig, axs = pt.subplots(2, figsize=(12,8))

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

# finding the ideal function (i_f) to fit the test function

''' this list will keep the data on the best i.f and their max delta'''
chosen_i_f_number_abs_delta = []  

for i in range(1, len(training_data[0, :]), 1):
    ''' Holds data on sum (delta**2) from the training set devition from all ideal functions'''
    sum_delta_sqr_list = []
    
    ''' Holds data max(abs(delta)) for all i.f '''
    sum_delta_sqr_list = []

    ''' we take an ideal function'''
    for j in range(1, len(ideal_data[0,:]), 1):
        ''' find delta for all points'''
        delta_list = np.subtract(training_data[:, i], ideal_data[:j])
        max_abs_delta = np.amax(np.abs(delta_list))
        delta_sqr_list = np.sqaure(delta_list)
        sum_delta_sqr = np.sum(delta_sqr_list)
        sum_delta_sqr_list.append(sum_delta_sqr)
        abs_delta_list.append(max_abs_delta)
    ''' find index of best i.f for this training set by minimal sum(delta**2)'''
    best_i_function_index = sum_delta_sqr_list.index(min(sum_delta_sqr_list))

    '''store the corresponding max(abs(delta))'''
    max_abs_delta_best_i_f = abs_delta_list[best_i_function_index]
    '''add data to final list '''
    chosen_i_f_number_abs_delta.append((best_i_function_index + 1, max_abs_delta_best_i_f))

'''plotting ideal data together with corresponding ideal function'''
fig, axs = pt.subplots(2,2, figsize(12,8))
k = 0
for i in range(2):
    for j in range(2):
        k += 1
        y_training = training_data[:, k]
        ''' i.f loaded from table by index from previously constructed list'''
        y_ideal = ideal_data[:, chosen_i_f_number_abs_delta[k-1][0]]
        axs[i, j].set_title("Training data set N{}".format(k))
        axs[i,j].plot(x, y_training, label='training')
        axs[i,j].plot(x,y_training, label="ideal")
        axs[i,j].legend()
plt.savefig("figure-2.png")
plt.show()

# reading the test data set
with open("test.csv",'r') as f:
    test_data = list(csv.reader(f, delimiter=","))
    ''' transform to np.array'''
    test_data = np.array(test_data[1:], dtype=np.float)

    '''sort for convinience'''
    test_data = test_data[np.argsort(test_data[:,0])]

    ''' filtering points that can be attributed to chosen ideal functions'''
    filtered_test_data = []
    
    for j in range(len(test_data[:, 0])):
        x = test_data[j, 0]
        y_test = test_data[j, 1]
        if x in ideal_data[:, 0]:
            for i in range(4):
                '''for every chosen ideal f'''
                y_ideal_index = np.where(ideal_data[:,0] == x)

                y_ideal = ideal_data[y_ideal_index, chosen_i_f_number_abs_delta[i][0]]
                if abs(y_test - y_ideal) <= np.sqrt(2) * chosen_i_f_number_abs_delta[i][1]:
                    filtered_test_data.append([x, y_test, chosen_i_f_number_abs_delta[i][0], float(abs(y_test - y_ideal)) ])

# reformar lst to np array
filtered_test_data = np.array(filtered_test_data)

# create last plot
fig, axs = pt.subplots(1, figsize=(12, 8))
axs.set_title("Test dataset")
x_test = test_data[:, 0]
y_test = test_data[:, 1]
axs.plot(x_test, y_test, "r+", label='test data')
pt.savefig("figure-3.png")
