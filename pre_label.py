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

    if i == len(lst):
        j = 1
        while j < i:
            lst[idx + j] = lst[idx + j - 1]
            j += 1
        return lst
    # Now i is the index of the next legit element
    if idx == -1:
        j = 0
        while j < i:
            lst[j] = lst[i]
            j += 1
        return lst

    diffx, diffy = (lst[idx][0] - lst[i][0])/i, (lst[idx][1] - lst[i][1])/i
    print(diffx, diffy)
    k = 1
    while k < i:
        lst[idx + k] = (lst[idx + k - 1][0] - diffx, lst[idx + k - 1][1] - diffy)
        k += 1
    return lst



f = open('labels/label2.dat', 'r')
next(f)
lst = [eval(j) if j != 'NaN' else 'NaN' for j in [i.rstrip() for i in f.readlines()]]
nan_indices = []
for i in range(len(lst)):
    if lst[i] == 'NaN':
        nan_indices.append(i)



