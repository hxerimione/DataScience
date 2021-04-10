import pymysql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

#함수 정의

def classification_perfomance_eval(y,y_predict):
    tp_a,tn_a,fp_a,fn_a=0,0,0,0
    tp_b,tn_b,fp_b,fn_b=0,0,0,0
    tp_c,tn_c,fp_c,fn_c=0,0,0,0
    
    tp,tn,fp,fn=0,0,0,0
    for y,yp in zip(y,y_predict):
        if yp==1 and y==1:
            tp_a+=1
        if (yp==2 or 3) and(y==2 or 3):
            tn_a+=1
        if yp==1 and (y==2 or 3):
            fp_a+=1
        if (yp==2 or 3) and y==1:
            fn_a+=1

        
        if yp==2 and y==2:
            tp_b+=1
        if (yp==1 or 3) and(y==1 or 3):
            tn_b+=1
        if yp==2 and (y==1 or 3):
            fp_b+=1
        if (yp==1 or 3) and y==2:
            fn_b+=1

            
        if yp==3 and y==3:
            tp_c+=1
        if (yp==1 or 2) and(y==1 or 2):
            tn_c+=1
        if yp==3 and (y==1 or 2):
            fp_c+=1
        if (yp==1 or 2) and y==3:
            fn_c+=1
   

    tp=tp_a+tp_b+tp_c
    tn=tn_a+tn_b+tn_c
    fp=fp_a+fp_b+fp_c
    fn=fn_a+fn_b+fn_c
    
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision_a=(tp_a)/(tp_a+fp_a)
    precision_b=(tp_b)/(tp_b+fp_b)
    precision_c=(tp_c)/(tp_c+fp_c)
    recall_a=(tp_a)/(tp_a+fn_a)
    recall_b=(tp_b)/(tp_b+fn_b)
    recall_c=(tp_c)/(tp_c+fn_c)
    f1_score_a=2*precision_a*recall_a/(precision_a+recall_a)
    f1_score_b=2*precision_b*recall_b/(precision_b+recall_b)
    f1_score_c=2*precision_c*recall_c/(precision_c+recall_c)
    
    return accuracy,precision_a,precision_b,precision_c,recall_a,recall_b,recall_c,f1_score_a,f1_score_b,f1_score_c


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
X = np.c_[X, random_state.randn(n_samples,22*n_features)]

y=[1 if(t['grade']=='A') else(2 if(t['grade']=='B')else 3)for t in data]
y=np.array(y)
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.33,random_state=43)


from sklearn import ensemble
es=ensemble.RandomForestClassifier()
model=es.fit(X_train,y_train)
y_predict = model.predict(X_test)


acc,prec_a,prec_b,prec_c,rec_a,rec_b,rec_c,f1_a,f1_b,f1_c=classification_perfomance_eval(y_test,y_predict)
print("accuracy : ",acc)
print("A의 precision : ",prec_a)
print("B의 precision : ",prec_b)
print("C의 precision : ",prec_c)
print("A의 recall : ",rec_a)
print("B의 recall : ",rec_b)
print("C의 recall : ",rec_c)
print("A의 f1 score :",f1_a)
print("B의 f1 score :",f1_a)
print("C의 f1 score :",f1_a)


#K-Fold
from sklearn.model_selection import KFold

accuracy=[]
precision_a=[]
precision_b=[]
precision_c=[]
recall_a=[]
recall_b=[]
recall_c=[]
f1_score_a=[]
f1_score_b=[]
f1_score_c=[]

kf=KFold(n_splits=5,random_state=43,shuffle=True)

for train_index, test_index in kf.split(X):
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]
   
    from sklearn import ensemble
    es=ensemble.RandomForestClassifier()
    model=es.fit(X_train,y_train)
    y_predict = model.predict(X_test)

   
    
    acc,prec_a,prec_b,prec_c,rec_a,rec_b,rec_c,f1_a,f1_b,f1_c=classification_perfomance_eval(y_test,y_predict)
    accuracy.append(acc)
    precision_a.append(prec_a)
    precision_b.append(prec_b)
    precision_c.append(prec_c)
    recall_a.append(rec_a)
    recall_b.append(rec_b)
    recall_c.append(rec_c)
    f1_score_a.append(f1_a)
    f1_score_b.append(f1_b)
    f1_score_c.append(f1_c)



import statistics

print("average_accuracy", statistics.mean(accuracy))
print("average_precision_A", statistics.mean(precision_a))
print("average_precision_B", statistics.mean(precision_b))
print("average_precision_C", statistics.mean(precision_c))
print("average_recall_A", statistics.mean(recall_a))
print("average_recall_B", statistics.mean(recall_b))
print("average_recall_C", statistics.mean(recall_c))
print("average_f1_score_A", statistics.mean(f1_score_a))
print("average_f1_score_B", statistics.mean(f1_score_b))
print("average_f1_score_C", statistics.mean(f1_score_c))
