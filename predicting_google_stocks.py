#Hello there... myself Rakesh pandey
#Student at NIT jalndhar(First year -CSE branch)
#And this is MAJOR project
#To predict the prices of google stocks !!!!
#I have implemented both Linear model and polynomial model to predict!!!


import quandl
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

#Extracting dataset using quandl...!!!!

g=quandl.get("WIKI/GOOGL")
g=g.reset_index()
g=g.reset_index()
df=pd.DataFrame(g)

#Taking columns info for training dataset
index_=df['index']
days_=np.array(index_[:,np.newaxis])
priceopen=df['Adj. Open']
price_train=np.array(priceopen[:])

#Linear regression model for predicting stockes !!!!
regres_simple=linear_model.LinearRegression()
regres_simple.fit(days_,price_train)

#Polynomial model for predicting stockes !!!!
reg_with_poly=linear_model.LinearRegression()
poly=PolynomialFeatures(4)
x_transform=poly.fit_transform(days_)
reg_with_poly.fit(x_transform,price_train)

#Extracting info for predicting  values from both models !!!!
tes=pd.read_csv('data_to_be_predicted.csv')
test_date=np.array(tes.NO[:,np.newaxis])
dates_predict=np.array(tes.DATE_PREDICTED[:])

#Predicting future stocks by Linear model
pred_by_simple_regression=regres_simple.predict(test_date)

#Predicting future stocks by Polynomial model
xtest_trans=poly.fit_transform(test_date)
pred_by_poly_regression=reg_with_poly.predict(xtest_trans)


#Building dataframe for predicted values by both models !!!!
dataframe_data=np.array([pred_by_poly_regression,pred_by_simple_regression])
index=np.array(tes.DATE_PREDICTED[:])
columns=['A    ','B    ']
pred_dataframe = pd.DataFrame(dataframe_data.T, index=index, columns=columns)
print("A = google stock prices(Adj .Open) predicted by Polynomial model")
print("B = google stock prices(Adj .Open) predicted by Linear model")
print(pred_dataframe)
print("   ")
print("CONCLUSION : 'POLYNOMIAL model predicts better as well as  "
      "fits the graph better ..as it can be clearly seen in graph")


#Plotting the graphs based on training dataset
plt.scatter(days_,price_train,color='black',s=1,label='data')
plt.plot(days_,regres_simple.predict(days_),color='blue',linewidth=2.0,label='linear model')
plt.plot(days_,reg_with_poly.predict(x_transform),color='green',linewidth=2.0,label='Polynomial model')
plt.xlabel("Days(Approx)" )
plt.ylabel('stock price in USD')
plt.title('Implementation on training data')
plt.legend()
plt.autoscale(tight='True')
plt.show()


#Thanku !!!!!




