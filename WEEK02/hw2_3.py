import pymysql
import numpy
import pandas as pd

df=pd.read_excel('./db_score.xlsx')
subset = df[[ 'attendance', 'homework','discussion','midterm','final','score']]

atd = subset.sort_values(['score'], ascending=[False])
atd=subset['score']
sum=0
value=[]
for i in range(92):
    value.append(atd[i])
    sum+=atd[i]

mean=sum/92
var=numpy.var(value) #분산
print("Var : ",var)
std=numpy.std(value) #표준편차
print("Std : ",std)
sum2=0
med=0
value.sort()
for i in range(92):
    value[i]=abs(value[i]-mean)
    sum2+=value[i]
med=value[45]+value[46]
aad=sum2/92 #AAD
print("AAD : ",aad)
mad=med/2 #MAD
print("MAD : ",mad)
