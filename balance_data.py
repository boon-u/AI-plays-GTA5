## Change numbers according to your specs 

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

lefts = []
rights = []
straights = []
st_lefts = []
st_rights = []
br_lefts = []
br_rights = []
breaks = []



##shuffle(train_data)
no_train_data_i_have = 52

for i in range(0, no_train_data_i_have+1):
    train_data = np.load('training_data-{}.npy'.format(i))
    for data in train_data:
        img = data[0]
        choice = data[1]
##        print(choice)

        if choice == [0,0,1,0,0,0,0,0,0]:
            lefts.append([img, choice])
        elif choice == [0,0,0,1,0,0,0,0,0]:
            rights.append([img, choice])
        elif choice == [1,0,0,0,0,0,0,0,0]:
            straights.append([img, choice])
        elif choice == [0,0,0,0,1,0,0,0,0]:
            st_lefts.append([img, choice])
        elif choice == [0,0,0,0,0,1,0,0,0]:
            st_rights.append([img, choice])
        elif choice == [0,0,0,0,0,0,1,0,0]:
            br_lefts.append([img, choice])
        elif choice == [0,0,0,0,0,0,0,1,0]:
            br_rights.append([img, choice])
        elif choice == [0,1,0,0,0,0,0,0,0]:
            breaks.append([img, choice])
        
straights = straights[:3501]
                
print('a',len(lefts))
print('d',len(rights))
print('w',len(straights))
print('wa',len(st_lefts))
print('wd',len(st_rights))
print('sa',len(br_lefts))
print('sd',len(br_rights))
print('s',len(breaks))


final_data = straights + lefts + rights + st_lefts + st_rights + br_lefts + br_rights + breaks
shuffle(final_data)
print(len(final_data))
##
np.save('bike-training_data-final-final.npy', final_data)
##                                
