import pandas 
train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')
print(train.describe())
print(test.describe())
print(train.dtypes)
print(train.isnull().sum())
print(test.dtypes)
print(test.isnull().sum())
#creating a model using only bath,balcony as features and
#price as target
#Removing unnecessary columns for the current version 
train = train.drop(['area_type','availability','location','size','society','total_sqft'],axis=1)
test = test.drop(['area_type','availability','location','size','society','total_sqft'],axis=1)
print(train.dtypes)
print(train.isnull().sum())
print(test.isnull().sum())

#Handling nan values, for now it is mean of the column
print(train['bath'].shape)
train['bath'] = train['bath'].fillna(train['bath'].mean())
train['balcony'] = train['balcony'].fillna(train['balcony'].mean())
test['bath'] = test['bath'].fillna(test['bath'].mean())
test['balcony'] = test['balcony'].fillna(test['balcony'].mean())
print(train.isnull().sum())
print(test.isnull().sum())


