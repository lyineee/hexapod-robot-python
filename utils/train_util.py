import numpy as np 


def train_set_percentage():
    test=ImageTrain()
    label,_=test.get_data_range(0,32)
    label=np.argmax(label,axis=1)
    a=np.size(np.where(label==2))
    b=np.size(np.where(label==3))
    c=np.size(np.where(label==5))
    d=np.size(np.where(label==6))
    e=np.size(np.where(label==0))
    f=np.size(np.where(label==1))
    g=np.size(np.where(label==4))
    total=a+b+c+d+e
    print('2:{} 3:{} 5:{} 6:{} 0:{} 1:{} 4:{}'.format(a,b,c,d,e,f,g))