import numpy as np
import os
import pandas as pd
from ggplot import *


# Start by generating data for us to test our SVM on.
x1 = np.random.uniform(low=1, high=10, size=(100,))
y1 = np.random.uniform(low=1, high=10, size=(100,))

x2 = np.random.uniform(low=1, high=10, size=(100,))
y2 = np.random.uniform(low=20, high=50, size=(100,))


labels = ['Cat1']*100 + ['Cat2']*100

data = {'x': np.concatenate([x1, x2]), 'y': np.concatenate([y1, y2]), 'label': labels}
frame = pd.DataFrame(data)

plt = ggplot(aes(x='x', y='y', color='label'), data=frame) +\
    geom_point() +\
    theme_bw() +\
    xlab("x") +\
    ylab("y") +\
    ggtitle("Test")
plt.show()
'''
 Now we know that we will want to create a function that will test a hyperplane and then perform gradient
 descent until we arrive at our conclusion.
'''

class mySVM:
    def __init__(self, name):
        self.name = name
    def model(data)











