import numpy as np
import os
import pandas as pd
from ggplot import *
from sklearn.metrics import accuracy_score


# Start by generating data for us to test our SVM on.
x1 = np.random.uniform(low=1, high=10, size=(100,))
y1 = np.random.uniform(low=1, high=10, size=(100,))

x2 = np.random.uniform(low=1, high=10, size=(100,))
y2 = np.random.uniform(low=20, high=50, size=(100,))


labels = [1]*100 + [-1]*100

data = {'x': np.concatenate([x1, x2]), 'y': np.concatenate([y1, y2]), 'label': labels}
frame = pd.DataFrame(data)

plt = ggplot(aes(x='x', y='y', color='label'), data=frame) +\
    geom_point() +\
    theme_bw() +\
    xlab("x") +\
    ylab("y") +\
    ggtitle("Test")
#plt.show()
'''
 Now we know that we will want to create a function that will test a hyperplane and then perform gradient
 descent until we arrive at our conclusion.
'''

class mySVM:
    def __init__(self, df):
        self.df = df
    def train(self, df, epochs=1000, t=10, alpha = 0.05):
        x = df.iloc[:,0].to_numpy().reshape(df.shape[0], 1)
        y = df.iloc[:,1].to_numpy().reshape(df.shape[0], 1)
        w = np.zeros((df.shape[0], 1))
        epoch = 1
        while (epoch < epochs):
            yhat = w * x
            prod = yhat * y
            print('The current epoch is: ' + str(epoch) + ' out of a total of ' + str(epochs) + ' epochs')
            count = 0
            for val in prod:
                if (val >= 1):
                    cost = 0
                    w = w - alpha * (1/epoch * w)
                else:
                    cost = 1 - val
                    w = w + alpha * (x[count] * y[count] - 1/epoch * w)
                count+=1
            epoch+=1
        df['weight'] = w
    def predict(self, df):
        x = df.iloc[:,0].to_numpy().reshape(df.shape[0], 1)
        w = df.weight.to_numpy().reshape(df.shape[0],1)
        label = df.label.to_numpy()
        print(label)
        y_pred = w * x
        predictions = []
        for val in y_pred:
            if(val > 1):
                predictions.append(1)
            else:
                predictions.append(-1)

        print(predictions)
        print(accuracy_score(label,predictions))
        

test = mySVM(df = frame)
test.train(test.df)
test.predict(test.df)











