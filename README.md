# hexapod robot python code

see [notice](./NOTICE.md)

## 文件结构

### 有用的文件

#### python

[hexapod.py](./hexapod.py) - vrep机器人仿真控制类

[by_opt.py](.by_opt.py) - 贝叶斯优化程序

[control.py](./control.py) - 主控制程序,跑真赛道的时候用

[bluetooth_control.py](./bluetooth_control.py) - 蓝牙信息发送类

[image_train.py](./image_train.py) 包括:

1. 循迹神经网络训练程序

    ```python
    train = ImageTrain()
    train.train(data=data,label=label) # data为数据 label为标签
    ```

2. vrep仿真,使用`test_robot()`这行代码来在vrep中仿真.(不要改`'turn_straight_cla_v0.2.h5'`这个)

[image_label.py](./image_label.py) - 用于标记图片

#### arduino

None

### 没用的文件(夹)

tet.py
./deprecated/
./map/
./logs/
___

## 下面的似乎没什么用了

___

## 运行 image-capture 采集数据

### 更改转弯角度

在`image-capture.py`文件中更改下面的东西

``` python
while True:
    if STATE==0:
        rb.one_step(0.003)
    if STATE==1:
        rb.turn_left([20, 30])
    if STATE==2:
        rb.turn_right([20,36])
```

## 摄像头参数

1. 角度 60度
2. 分辨率 1920*1080

## 怎样转向

两种方法

1. 直接分类四种转向状态
2. 间接获取数据计算转向状态(四种状态,两种状态?)

### 获取什么间接数据

1. 轨迹线的中线
2. 轨迹线的边线

## 训练结果很受杂音的影响,模型抓不到重点

## cv2 notice

cv2 图片似乎支持两种格式

1. `np.int8`:用于图片显示
2. `np.float`:用于图片处理
3. `np.uint8`:用于数据保存,图片显示,读取

两者可以通过`np.astype`转换

*注意: np.int8范围是 -128 - 128 如果搞错可能会出现画面变白*

## 其他

1. 训练集数组为`np.uint8`

## 解决使用cv2是pylint的报错  `Module 'cv2' has no **** member`

1. 使用`from cv2 import cv2`
2. 更改pylint设置`pylint --extension-pkg-whitelist=cv2`

## 赛道制作

1. pv纸 80cm 宽:制作大弯
2. 0.6 * 0.9 似乎最便宜
