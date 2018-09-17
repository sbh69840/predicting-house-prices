import pandas 
import numpy as np 
from sklearn.model_selection import train_test_split
pandas.set_option('display.max_columns',20)
train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')
print(train.shape)
#Here I will be dropping all the nan values if they are in any columns,
#because if you look at unique values in train set(after removing nan) and test set(without)
#removing nan, then the values in test set is sub set of values in train set(after rmoving nan)
#and that is true 95% of the times, and hence I would prefer to just drop, all nan's
#instead of filling it with values which aren't right(which is my intution, for example, mean
#of the columns, etc)
#>>>>> You can chck the above thing by replacing train with test in the variable
#a and then compare it with train, and you can verify.
train = train.dropna()
print(train.shape)
print(train.head())
print("Names of columns in the dataset are : ==>:",train.columns)
print("Datatypes of each column in the dataset are: ==>:",train.dtypes)
print("Shape of the dataset is : ==>:",train.shape)
#Listing all the unique values in each column
a = [list(train[a].unique()) for a in train.columns]
print("Unique values in AREA_TYPE column (not converted to int): ==>:",a[0])
new_area = {}
for b in a[0]:
    if b =="Super built-up  Area":
        new_area[b] = 4
    elif b=="Plot  Area":
        new_area[b] = 1
    elif b=="Built-up  Area":
        new_area[b] = 3
    else:
        new_area[b] = 2
#In the above for loop, I am converting the four area types to integers, I have just used
#my intution to decide which has to be weighed more than other. I know it is not good, as
#I am not letting the algorithm to decide which is better, but since I am will be trying
#both machine learning approaches involving neural networks and non-neural networks because 
#many people still don't get it why we use neural network
#I will slowly increase dimensions for both non-neural netwroks and neural networks and see 
#what happens.
print("Unique values in area_type converted to int: ==>:",new_area)
print("Unique values in availability column (not converted to int): ==>:",a[1])
new_avail = {}
for b in a[1]:
    if b=="Ready To Move":
        new_avail[b] = 1
    else:
        new_avail[b] = 0
#In the above for loop I am setting all the values equal to ready to move to 1
#And all the values not equal to "READY TO MOVE" to 0, because if a value is not
#ready to move, i.e it can be any date, like 19-AUG, but we don't know at what point of
#time is the ounting of date starting? i.e we don't know what is the refernce date to which 
#we are comparing it to. If the year was mentioned we could have taken a day before the least
#date as the reference date but it is not mentioned, so it is obvious to make all the dates
#to zero. As most of them would prefer READY TO MOVE.
print("Unique values in availability converted to int: ==>:",new_avail)

#print(a[2]) is not required, because the location can't be numerically understood as of now, 
#therefore we will preserve it for later.
#For seperating number of bedrooms to only integer instead of BHK and Bedroom
print(a[3]) 
room = [str(a[3][i]).split()[0] for i in range(len(a[3]))]
new_room = {}
for ind,b in enumerate(a[3]):
    new_room[b] = room[ind]
print("Unique values in rooms converted to int: ==>:",new_room)

#Getting the index of all the unique size of the property with values other than sqft,
#i.e Acres,Yards...etc
strs = []
for i in range(len(a[5])):
    try:
        b = float(a[5][i])
    except:
        s = a[5][i].split(' ')
        try:
            if s.index('-')!=-1:
                #Finding the mean of two values beside '-'
                new_s = (float(s[0])+float(s[2]))/2
        except:
            strs.append(i)
print("Length of unique values with dimensions other than sq.ft",len(strs))        
print("Values of unique values in total_sqft with dimensions other than sq.ft",[a[5][i] for i in strs])
new_sqft = []
not_sqft = [a[5][i] for i in strs]
for ind,a in enumerate(not_sqft):
    index = a.find('Sq. Meter')
    if index!=-1:
        num = float(a[:index])
        num = num*10.7639
        new_sqft.append(num)
        continue 
    index = a.find('Sq. Yards')
    if index!=-1:
        num = float(a[:index])
        num = num*9
        new_sqft.append(num) 
        continue
    index = a.find('Acres')
    if index!=-1:
        num = float(a[:index])
        num = num*43560 
        new_sqft.append(num)
        continue 
print("Now all the unique values in total_sqft are converted to sq. ft",new_sqft)




#Today I was only able to do till converting yards and sq meter to total_sqft.
#Finally back to solving this problem after my internal examination and a hackathon (WNS-analytics wizard)



"""train = train.drop(['area_type','availability','location','size','society','total_sqft'],axis=1)
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
#even with this score."""
