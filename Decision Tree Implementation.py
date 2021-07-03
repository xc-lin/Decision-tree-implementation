import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns

ds = pd.read_csv('titanic.csv')
# ds.info()
# ds.head()
cols_to_drop = [
    'PassengerId',
    'Name',
    'Ticket',
    'Cabin',
    'Embarked',
]

df = ds.drop(cols_to_drop, axis=1)


# print(df.head())


def convert_sex_to_num(s):
    if s == 'male':
        return 0
    elif s == 'female':
        return 1
    else:
        return s


df.Sex = df.Sex.map(convert_sex_to_num)
# print(df.head())

data = df.dropna()
data.describe()
plt.figure()
sns.heatmap(data.corr())
plt.show()

input_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
out_cols = ['Survived']

X = data[input_cols]
y = data[out_cols]

# X.head()
print(X.shape, y.shape)
data = data.reset_index(drop=True)


def divide_data(x_data, fkey, fval):
    x_right = pd.DataFrame([], columns=x_data.columns)
    x_left = pd.DataFrame([], columns=x_data.columns)

    for ix in range(x_data.shape[0]):
        # Retrieve the current value for the fkey column lets call it val
        try:
            val = x_data[fkey].loc[ix]

        except:
            print(x_data[fkey])
            val = x_data[fkey].loc[ix]
        # Check where the row needs to go
        if val > fval:
            # pass the row to right
            x_right = x_right.append(x_data.loc[ix])
        else:
            # pass the row to left
            x_left = x_left.append(x_data.loc[ix])

    # return the divided datasets
    return x_left, x_right


def entropy(col):
    p=[]
    p.append(col.mean())
    p.append(1 - p[0])

    ent = 0.0
    for px in p:
        ent += (-1 * px * np.log2(px))
    return ent


def information_gain(xdata, fkey, fval):
    left, right = divide_data(xdata, fkey, fval)

    if len(left) == 0 or len(right) == 0:
        return -10000
    return entropy(xdata.Survived) - (
            entropy(left.Survived) * float(left.shape[0] / left.shape[0] + right.shape[0]) + entropy(
        right.Survived) * float(right.shape[0] / left.shape[0] + right.shape[0]))


for fx in X.columns:
    print(fx)
    print(information_gain(data, fx, data[fx].mean()))


class DecisionTree:
    def __init__(self, depth=0, max_depth=5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None

    def train(self, X_train):

        print(self.depth, '-' * 10)
        # Get the best possible feature and division value (gains)
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        gains = []
        for fx in features:
            gains.append(information_gain(X_train, fx, X_train[fx].mean()))

        # store the best feature (using min information gain)
        self.fkey = features[np.argmax(gains)]
        self.fval = X_train[self.fkey].mean()

        # divide the dataset and reset index
        data_left, data_right = divide_data(X_train, self.fkey, self.fval)
        data_left = data_left.reset_index(drop=True)
        data_right = data_right.reset_index(drop=True)

        # Check the shapes and depth if it has exceeded max_depth or not in case it has make predictions
        if data_left.shape[0] == 0 or data_right.shape[0] == 0:
            if X_train.Survived.mean() >= 0.5:
                self.target = 'Survived'
            else:
                self.target = 'Dead'
            return

        if self.depth >= self.max_depth:
            if X_train.Survived.mean() >= 0.5:
                self.target = 'Survived'
            else:
                self.target = 'Dead'
            return

        # branch to right
        self.right = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth)
        self.right.train(data_right)
        # branch to left
        self.left = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth)
        self.left.train(data_left)

        # Make your prediction
        if X_train.Survived.mean() >= 0.5:
            self.target = 'Survived'
        else:
            self.target = 'Dead'

        return

    def predict(self, test):
        if test[self.fkey] > self.fval:
            # go right
            if self.right is None:
                return self.target
            return self.right.predict(test)
        else:
            # go left
            if self.left is None:
                return self.target
            return self.left.predict(test)


split = int(0.8 * data.shape[0])

training_data = data[:split]
testing_data = data[split:]

dt = DecisionTree()
dt.train(training_data)

print(dt.fkey, dt.fval)
print(dt.right.fkey, dt.right.fval)
print(dt.left.fkey, dt.left.fval)

print(dt.right.right.fkey, dt.right.right.fval)
print(dt.right.left.fkey, dt.right.left.fval)

print(dt.left.right.fkey, dt.left.right.fval)
print(dt.left.left.fkey, dt.left.left.fval)

for ix in testing_data.index[:10]:
    print (dt.predict(testing_data.loc[ix]))

correct = 0
for ix in testing_data.index:
    a = dt.predict(testing_data.loc[ix])
    if testing_data.loc[ix].Survived == 0 :
        if a == 'Dead' :
            correct += 1
    if testing_data.loc[ix].Survived == 1 :
        if a == 'Survived' :
            correct += 1
print (correct)
print (testing_data.shape[0])
print (float(correct/testing_data.shape[0]))