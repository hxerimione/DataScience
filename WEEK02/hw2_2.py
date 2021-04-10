import pymysql
import pandas as pd
from collections import Counter

df=pd.read_excel('./db_score.xlsx')
subset = df[[ 'grade']]

grd = subset.sort_values(['grade'], ascending=[False])
grd=subset['grade']

a=[]
for i in range(92):
    a.append(grd[i])
cnt=Counter(a)

b=cnt.most_common()
mode=[]
for i in range(len(b)):
    c=b[i]
    if(i==0):
        max=c[1]
        mode.append(c[0])
    elif(c[1]==max):
        mode.append(c[0])

print("mode : ",mode)
