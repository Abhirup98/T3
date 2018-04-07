# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 14:51:16 2018

@author: LUCIA
"""

import numpy as np

movie_title_headers = ['MovieID', 'YearOfRelease','Title']
data_headers = ['MovieID', 'UserID', 'Ratings']



def copyData(filename):
    """Reads the data and retund a list with the data in the cvs"""
    data=[]
    #filename = "training_set.csv"
    try:
        fh = open(filename,'r')
    except IOError:
        print('cannot open', filename)
    else:
        for new in fh:
            if new !='\n':
                addIt =  new[:-1].split(',')
                data.append(addIt)
    finally:
        fh.close()
        #print(data)
    return data


movie_filename = 'movie_titles.txt'
training_filename = 'TrainingRatings.txt'
testing_filename = 'TestingRatings.txt'


movie_data = np.array(copyData(movie_filename))
training_data = np.array(copyData(training_filename))
testing_data = np.array(copyData(testing_filename))