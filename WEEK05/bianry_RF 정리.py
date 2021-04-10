import pymysql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#함수 정의

def classification_perfomance_eval(y,y_predict):
    tp,tn,fp,fn=0,0,0,0
    for y,yp in zip(y,y_predict):
        if y==1 and yp==1:
            tp+=1
        elif y==1 and yp==-1:  
            fn+=1
        elif y==-1 and yp==1:
            fp+=1
        else:
            tn+=1
    print("TP:" ,tp)
    print("FN:",fn)
    print("FP:",fp)
    print("TN:",tn)
    print('------')
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision=(tp)/(tp+fp)
    recall=(tp)/(tp+fn)
    f1_score=2*precision*recall/(precision+recall)

    return accuracy,precision,recall,f1_score


#DB연결

conn=pymysql.connect(host='localhost',user='root',password='purejang98',db='practice')
curs=conn.cursor(pymysql.cursors.DictCursor)
curs.execute("select * from ds_exam")

data=curs.fetchall()
curs.close()
conn.close()

#train_test_split

X=[(t['homework'],t['discussion'],t['midterm'],ord(t['grade']))for t in data]
X=np.array(X) 
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples,7*n_features)]

y=[1 if(t['grade']=='B')else -1 for t in data]
y=np.array(y)
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)

from sklearn import ensemble
es=ensemble.RandomForestClassifier()
model=es.fit(X_train,y_train)
y_predict = model.predict(X_test)

acc,prec,rec,f1=classification_perfomance_eval(y_test,y_predict)
print("accuracy=%f" %acc)
print("precision=%f" %prec)
print("recall=%f" %rec)
print("f1_score=%f" %f1)


#K-Fold
from sklearn.model_selection import KFold

accuracy=[]
precision=[]
recall=[]
f1_score=[]
kf=KFold(n_splits=5,random_state=42,shuffle=True)

for train_index, test_index in kf.split(X):
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]

    from sklearn import ensemble
    es=ensemble.RandomForestClassifier()
    model=es.fit(X_train,y_train)
    y_predict = model.predict(X_test)

    
    acc,prec,rec,f1=classification_perfomance_eval(y_test,y_predict)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)



import statistics

print("average_accuracy", statistics.mean(accuracy))
print("average_precision", statistics.mean(precision))
print("average_recall", statistics.mean(recall))
print("average_f1_score", statistics.mean(f1_score))
