#!/usr/bin/python

import numpy as np
import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below


# metodo para reconhecer o outlier e a key a qual ele pertence
"""
maxbonus = np.amax(data)
for i in data_dict:
    if data_dict[i]['bonus'] == maxbonus:
        print ('\nThe biggest bonnus value is %.2f and the related name is %s\n' % (maxbonus, i))
"""

data_dict.pop('TOTAL', 0 )  # removendo a Key TOTAL (outlier)
data = featureFormat(data_dict, features) # redefinindo o array data

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

bonus = []
bonus_bandit = []
for i in data_dict:
    if data_dict[i]['bonus'] != 'NaN':
        bonus.append(data_dict[i]['bonus'])

bonus = sorted(bonus,reverse=True)
bonus_bandit = bonus[1:3]

for i in data_dict:
    if data_dict[i]['bonus'] == bonus_bandit[0]:
        bandit_1 = i
for j in data_dict:
    if data_dict[j]['bonus'] == bonus_bandit[1]:
        bandit_2 = j

print ('\nThe two bandits are %s and %s and your respectively are %.2f and %.2f\n' % (bandit_1, bandit_2,bonus_bandit[0], bonus_bandit[1]))
