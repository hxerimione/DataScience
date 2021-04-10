import pymysql
import pandas as pd

df=pd.read_excel('./db_score.xlsx')
subset = df[[ 'attendance', 'homework','discussion','midterm','final','score']]

sum=0
med=0
atd = subset.sort_values(['score'], ascending=[False])
atd=subset['score']
print(type(atd))
for i in range(92):
    if(i==45 or i==46):
        med+=atd[i]
    sum+=atd[i]
    
mean=sum/92
print("mean : ",mean)
median=med/2
print("median : " ,median)
