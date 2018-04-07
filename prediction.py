# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:57:12 2018

@author: LUCIA
"""
import pandas as pd
import numpy as np
from  math import sqrt
filename = 'TrainingRatings.txt'
df = pd.read_csv(filename,header=None)
df.columns = ['MovieID','UserID','Rating']
print('Data loaded')
data_by_UserID = df.groupby('UserID')
data_by_MovieID = df.groupby('MovieID')
print('Data organized')

mean_vote = df[['UserID','Rating']].groupby('UserID').mean()

list_of_UserID = mean_vote.index.values

print('means computed')
global_w = pd.DataFrame(index = list_of_UserID, columns = list_of_UserID)

print ('Matrix w created')

def update_w_for(a,selected_movie):
    global global_w
    votes_of_a = data_by_UserID.get_group(a).drop('UserID', axis = 1).set_index('MovieID')
    #movies_of_a = votes_of_a.index.values #let see if we need this


    #The following two lines can be removed if we only pass the usersID_for_movie to the function
    votes_for_movie = data_by_MovieID.get_group(selected_movie).drop('MovieID', axis = 1).set_index('UserID')
    usersID_for_movie = votes_for_movie.index.values
    mean_a = mean_vote.loc[a,'Rating']

    for user in usersID_for_movie: #user is like the i in the formula
        if(np.isnan(global_w.loc[a,user])):
            #here we make the computation of the weights
            votes_of_i = data_by_UserID.get_group(user).drop('UserID', axis = 1).set_index('MovieID')
            #movies_of_i = votes_of_i.index.values #let see if we need this
            mean_user = mean_vote.loc[user,'Rating']
            
            common_movies = votes_of_a.index.intersection(votes_of_i.index)
            #Here we start the computation:
            if len(common_movies)> 0:            
                sum_lin = 0
                sum_sqr_a = 0
                sum_sqr_i = 0
                
                for movie in common_movies:
                    d_a =(votes_of_a.loc[movie,'Rating'] - mean_a)
                    d_i =(votes_of_i.loc[movie,'Rating'] - mean_user)
                    #print(votes_of_i.loc[movie,'Rating'], mean_vote.loc[user,'Rating'], d_i)
                    sum_lin += d_a * d_i
                    sum_sqr_a += d_a**2
                    sum_sqr_i += d_i**2
                #Potential divisionby 0 problem
                if (sum_lin != 0):
                    #print('Calculating w for a and i :',a,' ',user)
                    #print(sum_sqr_a)
                    #print(sum_sqr_i)
                    w_value_ai = sum_lin / sqrt(sum_sqr_a * sum_sqr_i)
                else:
                    w_value_ai = 0
            else:
                w_value_ai = 0
            global_w.loc[a,user] = w_value_ai 
            global_w.loc[user,a] = w_value_ai
            print('w for ',a,'and',user,': ',global_w.loc[a,user])
    #This is the end of the function,
    #It only updates w values

def predicted_vote(a,selected_movie):
    global global_w
    global mean_vote
    update_w_for(a,selected_movie)
    votes_for_movie = data_by_MovieID.get_group(selected_movie).drop('MovieID', axis = 1).set_index('UserID')
    usersID_for_movie = votes_for_movie.index.values
    sum_weights = 0
    sum_w_times_v = 0
    for user in usersID_for_movie:
        sum_weights += abs(global_w.loc[user,a])
        sum_w_times_v += global_w.loc[user,a] * (votes_for_movie.loc[user,'Rating'] - mean_vote.loc[user,'Rating'] )
    kappa = 1/sum_weights
    return mean_vote.loc[user,'Rating'] + kappa * sum_w_times_v

predicted_vote(553787,8)

    

