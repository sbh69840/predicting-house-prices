import pandas 
import numpy as np 
from sklearn.model_selection import train_test_split
train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')
#creating a model using only bath,balcony as features and
#price as target
#Removing unnecessary columns for the current version 
train = train.drop(['area_type','availability','location','size','society','total_sqft'],axis=1)
test = test.drop(['area_type','availability','location','size','society','total_sqft'],axis=1)

#Handling nan values, for now it is mean of the column
train['bath'] = train['bath'].fillna(train['bath'].mean())
train['balcony'] = train['balcony'].fillna(train['balcony'].mean())
test['bath'] = test['bath'].fillna(test['bath'].mean())
test['balcony'] = test['balcony'].fillna(test['balcony'].mean())

train['bath'] = train['bath'].apply(lambda x: x/40)
train['balcony'] = train['balcony'].apply(lambda x: x/3)
test['bath'] = test['bath'].apply(lambda x: x/40)
test['balcony'] = test['balcony'].apply(lambda x: x/3)

train['price'] = np.log1p(train['price'])


X = train[['bath','balcony']].values
y = train['price'].values
print(X.shape)
#Cleaning done
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,stratify=y,random_state=7)

#now let's create a simple model that accepts integer and return
from keras.models import Sequential 
from keras.layers import Conv1D,MaxPooling1D,Dropout,Dense,Flatten

model = Sequential()
#model.add(Conv1D(32,3,input_shape=(2,1),activation='relu'))


