import pandas as pd
import numpy as np
from sklearn import linear_model


df = pd.read_csv("homeprice.csv")

print(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
print(df)

reg = linear_model.LinearRegression()
reg.fit(df.drop('price',axis='columns'),df.price)

print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[3000,3,40]]))

