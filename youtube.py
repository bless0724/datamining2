import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#读取数据，并查看有多少属性
ca_youtube = pd.read_csv("CAvideos.csv",delimiter=',',low_memory=False)
print(ca_youtube.nunique())
#我们划分出标称属性和数值属性
nominal = ca_youtube[['trending_date','category_id','tags','comments_disabled','ratings_disabled','video_error_or_removed']]
numeric = ca_youtube[['views','likes','dislikes','comment_count']]
#对每一个标称属性计算频数
print(nominal['trending_date'].value_counts(),"\n")
print(nominal['category_id'].value_counts(),"\n")
print(nominal['tags'].value_counts(),"\n")
print(nominal['comments_disabled'].value_counts(),"\n")
print(nominal['ratings_disabled'].value_counts(),"\n")
print(nominal['video_error_or_removed'].value_counts(),"\n")
#对数值属性，计算5数概括，并查询缺失值
views = numeric['views'].to_numpy()
likes = numeric['likes'].to_numpy()
dislikes = numeric['dislikes'].to_numpy()
comment_count = numeric['comment_count'].to_numpy()
#对数值属性，计算5数概括，并查询缺失值
views_percentile = np.percentile(views, (0,25,50,75,100), interpolation='lower')
likes_percentile = np.percentile(likes, (0,25,50,75,100), interpolation='lower')
dislikes_percentile = np.percentile(dislikes, (0,25,50,75,100), interpolation='lower')
comment_count_percentile = np.percentile(comment_count, (0,25,50,75,100), interpolation='lower')
print("views的五数概括为：",views_percentile)
print("likes的五数概括为：",likes_percentile)
print("dislikes的五数概括为：",dislikes_percentile)
print("comment_count的五数概括为：",comment_count_percentile)

print("缺失值个数为：\n", numeric.isnull().sum())

#可视化
plt.boxplot(numeric['views'])
plt.ylabel('views')
plt.yscale('log')

plt.boxplot(numeric['likes'])
plt.ylabel('likes')
plt.yscale('log')

plt.boxplot(numeric['dislikes'])
plt.ylabel('dislikes')
plt.yscale('log')

plt.boxplot(numeric['comment_count'])
plt.ylabel('comment_count')
plt.yscale('log')


