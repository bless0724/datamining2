import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
#读取数据集，选择标称属性和数值属性
winemag = pd.read_csv("winemag-data_first150k.csv",delimiter=',',low_memory=False)
nominal = winemag[['country','designation','province','region_1','region_2','variety','winery']]
numeric = winemag[['points','price']]
#对标称属性计算频数
print(nominal['country'].value_counts(),"\n")
print(nominal['designation'].value_counts(),"\n")
print(nominal['province'].value_counts(),"\n")
print(nominal['region_1'].value_counts(),"\n")
print(nominal['region_2'].value_counts(),"\n")
print(nominal['variety'].value_counts(),"\n")
print(nominal['winery'].value_counts(),"\n")
#对数值属性，计算5数概括和缺失值个数,用dropna()将缺失数据去除。
print("缺失值个数为：\n", numeric.isnull().sum())
numeric_drop = numeric.dropna().astype(int)
points = numeric_drop['points'].to_numpy()
price = numeric_drop['price'].to_numpy()
mean = np.mean(price)
mode = stats.mode(price)[0][0]
points_percentile = np.percentile(points, (0,25,50,75,100), interpolation='lower')
price_percentile = np.percentile(price, (0,25,50,75,100), interpolation='lower')
print("points的五数概括为：",points_percentile)
print("price的五数概括为：",price_percentile)
#用直方图和盒图检查数值属性的分布和离群点
plt.hist(points)
plt.xlabel('points')
plt.show()

plt.boxplot(points)
plt.ylabel('points')
plt.show()

plt.figure(figsize=(10,10),dpi=64)
plt.hist(price,bins=2300)
plt.xscale('log')
plt.xlabel('price')
plt.show()

plt.boxplot(price)
plt.ylabel('price')
plt.yscale('log')
#只有price属性存在缺失值，我们用最高频率值来填补缺失值。
numeric_fill= numeric.fillna(mode).astype(int)
price_fill = numeric_fill['price'].to_numpy()
#重新绘制出直方图和箱图
plt.figure(figsize=(10,10),dpi=64)
plt.hist(price_fill,bins=2300)
plt.xscale('log')
plt.xlabel('price fill with mode')
plt.show()

plt.boxplot(price)
plt.ylabel('price fill with mode')
plt.yscale('log')
#第二种方法，用线性回归填补缺失值
from sklearn.linear_model import LinearRegression
numeric_nan = numeric[np.isnan(numeric['price'])]
lineR = LinearRegression()
lineR.fit(points.reshape(-1,1), price)
line = lineR.predict(numeric_nan['points'].to_numpy().reshape(-1,1))
numeric_nan_copy=numeric_nan.copy()
numeric_nan_copy['price']= line
numeric_linear = numeric_drop.append(numeric_nan_copy)
numeric_linear = numeric_linear.astype(int)
price_linear = numeric_linear['price'].to_numpy()
#重新绘制出直方图和箱图
plt.figure(figsize=(10,10),dpi=64)
plt.hist(price_linear,bins=2300)
plt.xscale('log')
plt.xlabel('price fill with LinearRegression')
plt.show()

plt.boxplot(price_linear)
plt.ylabel('price fill with LinearRegression')
plt.yscale('log')
#第三种方法，用KNN填补缺失值
from sklearn.neighbors import KNeighborsRegressor
clf = KNeighborsRegressor(n_neighbors = 10, weights = "distance")
clf.fit(points.reshape(-1,1), price)
knn = clf.predict(numeric_nan['points'].to_numpy().reshape(-1,1))
numeric_nan_copy=numeric_nan.copy()
numeric_nan_copy['price']= knn
numeric_knn = numeric_drop.append(numeric_nan_copy)
numeric_knn = numeric_knn.astype(int)
price_knn = numeric_knn['price'].to_numpy()
#重新绘制出直方图和箱图
plt.figure(figsize=(10,10),dpi=64)
plt.hist(price_knn,bins=2300)
plt.xscale('log')
plt.xlabel('price fill with KNN')
plt.show()

plt.boxplot(price_knn)
plt.ylabel('price fill with KNN')
plt.yscale('log')