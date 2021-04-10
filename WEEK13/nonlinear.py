import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras import initializers
#y=a + b*x1 + c*x2^2 + d*x3^3 + e

def gen_sequential_model(): #2
    model = Sequential([
        Input(3,name='input_layer'), #2개의 input
        
        Dense(9,activation='relu',name='hidden_layer1',kernel_initializer=initializers.RandomNormal(mean=00 ,stddev=0.05,seed=42)),
        Dense(9,activation='relu',name='hidden_layer2',kernel_initializer=initializers.RandomNormal(mean=00 ,stddev=0.05,seed=42)),
        Dense(1,activation='relu',name='output_layer',kernel_initializer=initializers.RandomNormal(mean=00 ,stddev=0.05,seed=42))
        ])
    model.summary()
    #print(model.layers[0].get_weights())
    #print(model.layers[1].get_weights())
    model.compile(optimizer='sgd',loss='mse')
    return model

#데이터 생성하는 함수 (400개 이상은 있어야지) #1
def gen_nonlinear_regression_dataset(numofsamples=500,a=1,b=3,c=5,d=10,e=20):
    np.random.seed(42)
    X=np.random.rand(numofsamples,3)#2차원 array로 리턴해달라 (샘플의 개수만큼)

    #print(X)
    #print(X.shape)
    X[0:numofsamples ,1:2]= X[0:numofsamples ,1:2]**2
    X[0:numofsamples ,2:3]= X[0:numofsamples ,2:3]**3
    #print(X)
    #y값 만들기
    coef=np.array([b,c,d])
    bias=a + e

    #print(coef)
    #print(coef.shape)

    y=np.matmul(X,coef.transpose())+bias

    #X=(numofsamples,3),coef.transpose() = (3,1)
    #print(y)
    #print(y.shape)

    return X,y

def plot_loss_curve(history): #4

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15,10))

    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upper right')
    plt.show()

#5
def predict_new_sample(model,x,a=1,b=3,c=5,d=10,e=20):
    x=x.reshape(1,3)
    
    y_pred=model.predict(x)[0][0]
    y_actual=b*x[0][0] + c*x[0][1] + d*x[0][2] +a+e
    print("y actual value=",y_actual)
    print("y predicted value=",y_pred)


def training():
    
    model=gen_sequential_model()
    X,y=gen_nonlinear_regression_dataset(numofsamples=1000)
    #트레이
    history=model.fit(X,y,epochs = 200,verbose=2,validation_split=0.3)

    return model,history

model,history=training()
#plot_loss_curve(history)
print("train loss=",history.history['loss'][-1])
print("test loss=",history.history['val_loss'][-1])
predict_new_sample(model, np.array([0.1,0.2,0.3]))

