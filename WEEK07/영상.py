import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pymysql
import time
from celluloid import Camera


def load_dbscore_data():
    conn=pymysql.connect(host='localhost',user='root',password='purejang98',db='practice')
    curs=conn.cursor(pymysql.cursors.DictCursor)
    curs.execute("select * from db_score")
    
    data  = curs.fetchall()
    
    curs.close()
    conn.close()
    
    #X = [ (t['attendance'], t['homework'], t['midterm'] ) for t in data ]
    X = [ ( t['midterm'] ) for t in data ]
    X = np.array(X)
    
    y = [ (t['score']) for t in data]
    y = np.array(y)

    return X, y

X, y = load_dbscore_data()

'''
plt.scatter(X, y) 
plt.show()
'''
print(min(X))
print(max(X))
# y = mx + c

import statsmodels.api as sm
X_const = sm.add_constant(X)

model = sm.OLS(y, X_const)
ls = model.fit()

#print(ls.summary())

ls_c = ls.params[0]
ls_m = ls.params[1]

mm=[]
cc=[]
def gradient_descent_naive(X, y):

    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001
    
    m = 0.0
    c = 0.0
    k=0
    n = len(y)
    
    c_grad = 0.0
    m_grad = 0.0
    
    for epoch in range(epochs):
        
        for i in range(n):
            y_pred = m * X[i] + c #y의 예측값
            m_grad += 2*(y_pred-y[i]) * X[i]
            c_grad += 2*(y_pred - y[i])

        c_grad /= n
        m_grad /= n
        
        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad

        
        if ( epoch % 1000 == 0):
            print("epoch %d: m_grad=%f, c_grad=%f, m=%f, c=%f" %(epoch, m_grad, c_grad, m, c) )
            mm.append(m)
            cc.append(c)
            k=k+1
            
        if ( abs(m_grad) < min_grad and abs(c_grad) < min_grad ):
            break

    
    
    

    return m,c,k
m, c, k = gradient_descent_naive(X, y)
fig,ax=plt.subplots()
camera=Camera(fig)

for i in range(k):
    plt.scatter(X, y,color='blue')
    ax2=ax.plot([0,35],[0*mm[i]+cc[i],35*mm[i]+cc[i]],color='red')
    plt.legend(ax2,['m :{:.3f} c :{:.3f}'.format(mm[i],cc[i])])
    camera.snap()
    fig
animation=camera.animate(interval=100,repeat=True)
plt.show()

animation.save('animation.mp4')
