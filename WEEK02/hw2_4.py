import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymysql
import pylab as P
from matplotlib import mlab


df=pd.read_excel('./db_score.xlsx')
subset = df[[ 'attendance', 'homework','discussion','midterm','final','score']]

atd = subset.sort_values(['score'], ascending=[False])
atd=subset['score']

arr=[]
for i in range(92):
    arr.append(atd[i])

d=np.sort(arr)
p=np.array([0,10,20,30,40,50,60,70,80,90,100])
plt.title('score')
plt.plot(np.percentile(d,p),marker='o')
plt.xticks(p/10,map(str,p))
plt.show()
