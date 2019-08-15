import cv2
import numpy as np
import os
import sys


class ImageLabeled(object):
    def __init__(self, file_dir='./'):
        # a-left d-right s-middle
        self.input_key_list = ['d', 'f', 'j', 'k', 'g']
        self.file_dir = file_dir
        self.window_name = 'labeled-app'

    def get_key(self):
        key = -1
        try:
            while True:
                try:
                    key = chr(cv2.waitKey(0))
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
        state = -1
        key = self.get_key()
        # convert
        if key == 'g':
            state = 0
        elif key == 'f':
            state = 2
        elif key == 'd':
            state = 3
        elif key == 'j':
            state = 5
        elif key == 'k':
            state = 6
        return state

    def image_data_gen(self, img_path='./image/1/', filter_tag=True, rescale=True):
        # get file name
        file_name_list = os.listdir(img_path)
        file_name_list = sorted(file_name_list, key=lambda x: int(
            os.path.splitext(x)[0].split('_')[-2]))

        for file_name in file_name_list:
            # label
            label_name = int(os.path.splitext(file_name)[0].split('_')[-1])
            if label_name != -1 and filter_tag:
                continue
            # data
            file_name = img_path+file_name
            img = cv2.imread(file_name)
            if rescale:
                img = cv2.resize(img, (640, 640))
            # img_array = np.array(img)

            yield file_name, img

    def dir_array_gen(self, path='./'):
        for root, dirs, _ in os.walk(path):
            if dirs:
                for dir_name in dirs:
                    yield root+dir_name+'/'
                break

    def name_label(self , file_name, label):
        split_name = os.path.splitext(file_name)
        new_name = split_name[0].split('_')[:-1]+[str(label)]
        os.rename(file_name, '_'.join(new_name)+split_name[1])

    def clear_labels(self, label, rotate=0, flip=False):
        for dir_name in self.dir_array_gen(self.file_dir):
            for file_name, img in self.image_data_gen(dir_name, filter_tag=False, rescale=False):
                # delete first image
                if os.path.splitext(file_name)[0].split('_')[-2] == '0':
                    os.remove(file_name)
                    continue
                # if os.path.splitext(file_name)[0].split('_')[-1]=='5':
                #     self.name_label(file_name,2)
                # if os.path.splitext(file_name)[0].split('_')[-1]=='6':
                #     self.name_label(file_name,3)
                # rotate image
                for _ in range(rotate):
                    img = np.rot90(img)

                # filp image
                if flip:
                    img = cv2.flip(img, 1)

                # from utils.cv2_util import edge_det
                # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # img=edge_det(img)

                cv2.imwrite(file_name, img)
                # self.name_label(file_name, str(label))

    def start_labeled(self):
        cv2.namedWindow(self.window_name)
        for dir_name in self.dir_array_gen(self.file_dir):
            for file_name, img in self.image_data_gen(dir_name):
                try:
                    img_line = cv2.line(
                        img, (0, 320), (640, 320), (0, 255, 0), thickness=3)
                    cv2.imshow(self.window_name, img_line)
                    state = self.get_state()
                    self.name_label(file_name, state)
                    print('labeled "{}" to {}'.format(file_name, state))
                except:
                    sys.exit(0)

    def end_labeled(self):
        cv2.destroyAllWindows()

    def file_gen(self,filter_tag=False, rescale=False):
        for dir_name in self.dir_array_gen(self.file_dir):
            for file_name, img in self.image_data_gen(dir_name, filter_tag, rescale):
                yield file_name, img

if __name__ == "__main__":
    test = ImageLabeled('./image/c/')
    # test.clear_labels('0',rotate=1,flip=True)
    # test.clear_labels('0',rotate=0,flip=True)
    # test.start_labeled()
