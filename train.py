import pandas 
train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')
print(train.describe())
print(test.describe())
print(train.dtypes)
print(train.isnull().sum())
print(test.dtypes)
print(test.isnull().sum())
