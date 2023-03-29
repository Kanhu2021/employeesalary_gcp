import numpy as np
import pandas as pd
data = pd.read_csv('salary.csv')
data['experience'] = data['experience'].map({'five':5,'two':2,'seven':7,'three':3,'ten':10,'eleven':11,0:0})
data['experience'].fillna(0,inplace=True)
data['test_score'].fillna(data['test_score'].mean(),inplace=True)
data.isnull().sum()
data.info()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
x = data.iloc[:,:-1]
y = data['salary']
model.fit(x,y)
model.predict([[2,3,4]])
#final_model = 
import pickle
pickle.dump(model,open('regmodel.pkl','wb'))

test = pickle.load(open('regmodel.pkl','rb'))
test.predict([[4,5,6]])


