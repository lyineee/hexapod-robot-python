# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import tensorflow as tf
import cv2
import numpy as np 
import os
import sys
import hashlib


#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#%%
class ImageLabeled(object):
    def __init__(self,file_dir='./'):
        # a-left d-right s-middle
        self.input_key_list=['d','f','j','k','g']
        self.file_dir=file_dir
        self.window_name='labeled-app'

    def get_key(self):
        key=-1
        try:
            while True:
                try:
                    key=chr(cv2.waitKey(0))
                except:
                    print('exiting')
                    sys.exit(0)
                if key in self.input_key_list:
                    break
        except:
            print('exiting')
            self.end_labeled()
            raise
        return key

    def get_state(self):
        state=-1
        key=self.get_key()
        # convert 
        if key == 'g':
            state=0
        elif key == 'f':
            state=2
        elif key == 'd':
            state=3
        elif key == 'j':
            state=5
        elif key == 'k':
            state=6
        return state

    def image_data_gen(self, img_path='./image/1/',filter_tag=True,rescale=True):
        # get file name
        file_name_list = os.listdir(img_path)
#         file_name_list=sorted(file_name_list,key=lambda x:int(os.path.splitext(x)[0].split('_')[-2]))

        for file_name in file_name_list:
            # label
            label_name = int(os.path.splitext(file_name)[0].split('_')[-1])
            if label_name != -1 and filter_tag:
                continue
            # data
            file_name = img_path+file_name
            img = cv2.imread(file_name)
            if rescale:
                img=cv2.resize(img,(640,640))
            # img_array = np.array(img)

            yield file_name,img
    
    def dir_array_gen(self,path='./'):
        for root,dirs,_ in os.walk(path):
            if dirs:
                for dir_name in dirs:
                    yield root+dir_name+'/'
                break

    def name_label(self,file_name,label,data):
        split_name=os.path.splitext(file_name)
        hash_md5 = hashlib.md5(file_name.encode()).hexdigest()
        new_name=split_name[0].split('_')[:-2]+[str(hash_md5)]+[str(label)]
        os.rename(file_name,'_'.join(new_name)+split_name[1])
    
    def clear_labels(self,label,rotate=0,flip=False):
        for dir_name in self.dir_array_gen(self.file_dir):
            for file_name,img in self.image_data_gen(dir_name,filter_tag=False,rescale=False):
                # delete first image
                if os.path.splitext(file_name)[0].split('_')[-2]=='0':
                    os.remove(file_name)
                    continue
                # if os.path.splitext(file_name)[0].split('_')[-1]=='5':
                #     self.name_label(file_name,2)
                # if os.path.splitext(file_name)[0].split('_')[-1]=='6':
                #     self.name_label(file_name,3)
                #rotate image
                for _ in range(rotate):
                    img=np.rot90(img)

                #filp image
                if flip:
                    img=cv2.flip(img,1)
                
                # from utils.cv2_util import edge_det
                # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # img=edge_det(img)

                cv2.imwrite(file_name,img)
                self.name_label(file_name,str(label))

    def start_labeled(self):
        cv2.namedWindow(self.window_name)
        for dir_name in self.dir_array_gen(self.file_dir):
            for file_name,img in self.image_data_gen(dir_name):
                try:
                    img_line=cv2.line(img,(0,320),(640,320),(0,255,0),thickness=3)
                    cv2.imshow(self.window_name,img_line)
                    state=self.get_state()
                    self.name_label(file_name,state)
                    print('labeled "{}" to {}'.format(file_name,state))
                except:
                    sys.exit(0)

    def end_labeled(self):
        cv2.destroyAllWindows()
    
    def file_gen(self,filter_tag=False,rescale=False):
        for dir_name in self.dir_array_gen(self.file_dir):
            for file_name,img in self.image_data_gen(dir_name,filter_tag,rescale):
                yield file_name,img


#%%
datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
#         shear_range=0.2,
        zoom_range=[0.8,1],
        fill_mode='nearest')


#%%
image=ImageLabeled('./image/d/')
for file_name,img in image.file_gen():
    img=cv2.flip(img.astype(np.float32),1)
    img=img.astype(np.uint8)
    img=img.reshape(1,64,64,3)
    # label_name = int(os.path.splitext(file_name)[0].split('_')[-1])
    label_name = 3
#     print(label_name)
    i=0
    for batch in datagen.flow(img, batch_size=1,
                              save_to_dir='./image/gen_image_t/',save_prefix='{}'.format(str(label_name)), save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely


#%%

label=ImageLabeled('./gen_image/')
num=0
for file_name,image in label.image_data_gen('./gen_image/',filter_tag=False):
#     print(os.path.splitext(file_name)[0].split('/')[-1].split('_')[0])
    if image.max()- image.min()<50:
        os.remove(file_name)
        num+=1
        continue
    # label_name = int(os.path.splitext(file_name)[0].split('/')[-1].split('_')[0])
    # label.name_label(file_name,label_name,image)
print(num)
    




#%%
