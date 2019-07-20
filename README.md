# hexapod robot python code

see [notice](./NOTICE.md)

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
