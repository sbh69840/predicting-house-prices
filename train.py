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
X_testing = test[['bath','balcony']].values

X = train[['bath','balcony']].values
y = train['price'].values
#Cleaning done
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=7)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error 
#Method to accept a regressor, fit it and predict, and give the best value possible.
def regression(X_train,y_train,X_test,y_test,regressor,parameters):
    reg = GridSearchCV(regressor,parameters,n_jobs=-1,cv=10,scoring="neg_mean_squared_error")
    reg.fit(X_train,y_train)
    mse = mean_squared_error(y_test,reg.predict(X_test))
    print(mse)
    return reg.best_estimator_,mse

from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import AdaBoostRegressor 


ens = [GradientBoostingRegressor(),BaggingRegressor(),KNeighborsRegressor(),\
       RandomForestRegressor(),ExtraTreesRegressor(),DecisionTreeRegressor(),\
       AdaBoostRegressor()]
scores = {}
for a in ens:
    reg,mse = regression(X_train,y_train,X_test,y_test,a,{})
    scores[a] = mse 
maxi = 1
for a in scores:
    if (scores[a]<maxi):
        print(scores[a],a)
        maxi = scores[a]
#According to the above code, GradientBoostingRegressor is the best with default parameters,
reg,mse = regression(X_train,y_train,X_test,y_test,GradientBoostingRegressor(),{})

    
    


#The underlying code is for creating the sbmission,
#since the hackathon holders expect me to submit in xl form I am using
#ExcelWriter of pandas 
# from keras.models import load_model 
# model = load_model('model.h5')
pred = reg.predict(X_testing)
pred = pred.reshape(-1,)
prediction = []
for a in pred:
    prediction.append(np.expm1(a))
print(prediction)
sub = pandas.DataFrame()
sub['price'] = prediction
print(sub.head()) 
writer = pandas.ExcelWriter('submit1-sklearn.xlsx')
sub.to_excel(writer,index=False)
writer.save()
#The score with this sklearnregressor,default parameters out of 1 was 0.78336541, which is
# 0.01 score increase from previos COnv1D single layer nueral network and I was placed at 63rd position
#even with this score.
