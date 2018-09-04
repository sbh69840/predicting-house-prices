import pandas 
train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')
#creating a model using only bath,balcony as features and
#price as target
#Removing unnecessary columns for the current version 
train = train.drop(['area_type','availability','location','size','society','total_sqft'],axis=1)
test = test.drop(['area_type','availability','location','size','society','total_sqft'],axis=1)

#Handling nan values, for now it is mean of the column
print(train['bath'].shape)
train['bath'] = train['bath'].fillna(train['bath'].mean())
train['balcony'] = train['balcony'].fillna(train['balcony'].mean())
test['bath'] = test['bath'].fillna(test['bath'].mean())
test['balcony'] = test['balcony'].fillna(test['balcony'].mean())
print(train.isnull().sum())
print(test.isnull().sum())
print("maximum of bath in train is: = {0}".format(train['bath'].max()))
print("maximum of balcony in train is: = {0}".format(train['balcony'].max()))

print("maximum of bath in test is: = {0}".format(train['bath'].max()))
print("maximum of balcony in test is: = {0}".format(train['balcony'].max()))

train['bath'] = train['bath'].apply(lambda x: x/40)
train['balcony'] = train['balcony'].apply(lambda x: x/3)
test['bath'] = test['bath'].apply(lambda x: x/40)
test['balcony'] = test['balcony'].apply(lambda x: x/3)

print(train.head())
print(test.head())

#Cleaning done

#now let's create a simple model that accepts integer and return



