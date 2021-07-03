import pandas as pd
import sklearn
import sns as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

split = int(0.8 * data.shape[0])
DT = DecisionTreeClassifier()
DT.fit(X[:split], y[:split])

print(DT.score(X[split:], y[split:]))

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X[:split], y[:split])

print(rf.score(X[split:], y[split:]))
