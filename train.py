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
X_testing = test[['bath','balcony']].values.reshape(-1,2,1)

X = train[['bath','balcony']].values
y = train['price'].values
print(X.shape)
#Cleaning done
print(y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=7)
print(X_train.shape)
#now let's create a simple model that accepts integer and return
from keras.models import Sequential 
from keras.layers import Conv1D,MaxPooling1D,Dropout,Dense,Flatten
from keras.callbacks import EarlyStopping,ModelCheckpoint 
#Reshape train and validation data to 3 dimenstional, since 
#The input shape is =(batch_size,2,1)

X_train = np.expand_dims(X_train,axis=2)
print(X_train.shape)
X_test = np.expand_dims(X_test,axis=2)
print(X_test.shape)

#Simple model wih 2 Layer convolution1D
model = Sequential()
model.add(Conv1D(32,2,input_shape=(2,1),activation='relu'))
model.add(Conv1D(32,1,activation='relu'))
model.add(Flatten())
model.add(Dense(1))

# save = ModelCheckpoint('model.h5',monitor='val_loss',save_best_only=True,mode='min')
# early = EarlyStopping(monitor='val_loss',mode='min',patience=5)
# model.compile(optimizer='adam',loss='mse')

# model.fit(X_train,y_train,validation_data=(X_test,y_test),verbose=1,batch_size=64,epochs=100,callbacks=[early,save])


#The underlying code is for creaating the sbmission,
#since the hackathon holders expect me to submit in xl form I am using
#ExcelWriter of pandas 
from keras.models import load_model 
model = load_model('model.h5')
pred = model.predict(X_testing)
pred = pred.reshape(-1,)
prediction = []
for a in pred:
    prediction.append(np.expm1(a))
print(prediction)
sub = pandas.DataFrame()
sub['price'] = prediction
print(sub.head()) 
writer = pandas.ExcelWriter('submit.xlsx')
sub.to_excel(writer,index=False)
writer.save()
#The score with this model out of 1 was 0.78247318 and I was placed at 63rd position.
