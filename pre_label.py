'''
Preprocesses the label files generated from tfpose thingy
'We fix NaN values by doing linear interpolation ' \
'between the previous and next values that make sense'
'''
import os
import numpy as np
from matplotlib import pyplot as plt


def interpolate(idx, lst):
    """
    :param idx: last index to make sense before NaN stuff. -1 if start with NaN
    :param lst: list to meme
    :return: same list with NaN values handled
    """
    i = 1
    try:
        while lst[idx + i] == 'NaN':
            i += 1
    except:
        pass

    if idx + i == len(lst):
        j = 1
        while j < i:
            lst[idx + j] = lst[idx + j - 1]
            j += 1
        return lst
    # Now i is the index of the next legit element
    if idx == -1:
        i -= 1
        j = 0
        while j < i:
            lst[j] = lst[i]
            j += 1
        return lst

    diffx = (lst[idx][0] - lst[idx + i][0])/i
    diffy = (lst[idx][1] - lst[idx + i][1])/i

    k = 1
    while k < i:
        lst[idx + k] = (lst[idx + k - 1][0] - diffx, lst[idx + k - 1][1] - diffy)
        k += 1
    return lst



f = open('labels/label1.dat', 'r')
first = next(f)
lst = [eval(j) if j != 'NaN' else 'NaN' for j in [i.rstrip() for i in f.readlines()]]
nan_indices = []
for i in range(len(lst)):
    if lst[i] == 'NaN':
        nan_indices.append(i)
indices_to_process = []
for nan_index in nan_indices:
    if(nan_index) == 0:
        indices_to_process.append(-1)
        continue
    if lst[nan_index-1] != 'NaN':
        indices_to_process.append(nan_index-1)

# round
def fc(x):
    if x == 'NaN':
        return 'NaN'
    else:
        return (round(x[0],3), round(x[1],3))


for idx in indices_to_process:
    lst = interpolate(idx, lst)
lst = list(map(fc, lst))
with open('labels/label1FIXED.dat', 'w+') as f:
    f.write(first)
    for element in lst:
        f.write('({},{})\n'.format(element[0], element[1]))

