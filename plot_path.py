import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
vid = cv2.VideoCapture('videos/vid1.mov')
fps = vid.get(cv2.CAP_PROP_FPS)
print(fps)
fig = plt.figure()
plt.xlim(0, 1)
plt.ylim(0, 1)
graph, = plt.plot([], [], 'o')

f = open('labels/label1FIXED.dat', 'r')
first = next(f)
lst = [eval(j) if j != 'NaN' else 'NaN' for j in [i.rstrip() for i in f.readlines()]]

def animate(i):
    if i == 0:
        print('NEW')
    graph.set_data(lst[i][0], 1-lst[i][1])
    return graph

ani = FuncAnimation(fig, animate, frames=len(lst), interval=220)
plt.show()