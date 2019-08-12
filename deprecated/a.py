def rrr(self):
    model=keras.models.load_model('result.h5')
    label,data=self.get_data_range(0,173)
    label,data=self.rechoice(label,data)
    model.fit(data, label, batch_size=32, epochs=50,
              validation_split=0.1)
    model.save('result.h5')


def get_loi(image_array,line=7):
row_img=image_array[line]
for i in range(len(row_img)):
    if row_img[i]>128:
        row_img[i]=1
    else:
        row_img[i]=0
return row_img

def get_state(loi,th=7):
if th <0:
    raise AttributeError('argument out of range')
length=len(loi)
if length%2 == 0:
    half=length/2
    left=loi[:int(half)]
    right=loi[int(half):]        
else:
    half=(length+1)/2
    left=loi[:int(half)-1]
    right=loi[int(half):]
delta=int(left.sum())-int(right.sum())
print(delta)
if delta>th:
    state=3
elif delta<-1*th:
    state=6
else:
    state=0
return state