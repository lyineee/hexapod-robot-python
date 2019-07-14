# 注意事项

## 一定记得要把连接各个元器件的电压调到正确的值

## 原件信息

1. 舵机
   - 型号:LDX-218
2. 舵机控制板
   - 控制方式:SPI总线
3. 主控板:Arduino
4. 蓝牙:zs-040
5. 稳压模块x2
6. 图传模块

## 电池充电

因为电池电流大所以电池放电特别快,要经常使用`检测器`检测电池电压,电池电压一定要**大于**最低电压

### 电压

使用:2s电池
>1s = 3.6v-4.2v => 2s = 7.2v-8.4v

### 充电电流

以`C`为单位
>1C=2200mah=2.2A
>充电电流: 3300 mah = 3.3 A 且不会超过5A

### 充电流程

1. 连接两种接口到充电板上
2. 选择"平衡充"`banlance`
3. 选择充电电压(1s)
4. 选择充电电流[见上](#%E5%85%85%E7%94%B5%E7%94%B5%E6%B5%81)
5. **长按**最右的按钮`start`,开始充电
6. 将电池放入**防爆袋**中,*注意*:充电器不要放进防爆袋

>充电器上排文字为**长按**选择,下排**短按**选择

## 舵机

注意舵机的型号,电压.使用`降压模块`上的拨码开关选择对应的电压

### [资料](https://www.amazon.com/LewanSoul-LDX-218-Standard-Digital-Bearing/dp/B07LF652M7)

>#### basic
>
>Product description:
>
>Dimension: 40*20*40.5mm(1.57*0.78*1.59inch)  
>
>Speed: 0.16sec/60°(7.4V)  
>
>Accuracy: 0.3°
>Torque: 15 kg·cm (208 oz·in) @6.6V;17 kg·cm (236 oz·in) @7.4V
>
>Working Voltage: 6-7.4V
>
>MIn Working Current: 1A
>
>No-Load Current: 100mA
>
>The default wire: 35cm(13.78inch) in length
>180 degree rotation. Controllable angle range from 0 to 180 degrees, Excellent linearity, precision control. Can be rotated within 360 degrees when power off
>
>#### Control Specifications
>
>Control Method: PWM
>
>Pulse Width: 500~2500
>
>Duty Ratio: 0.5ms~2.5ms
>
>Pulse Period: 20ms
>
>Please make sure that the duty cycle of the controller you are using conform >to our specifications, otherwise the servo can't turn up to claimed degree
>
>#### Notice for use
>
>1. It's recommended that use lithium polymer battery with high rate discharge>(min 5C), please don't use dry battery.
>2. Please use a short and thick power cord, don't use Dupont cord

## 图传

图传有单独的电源模块,连上电池打开开关就可以使用
>使用wifi进行图像传输

## 顺序

建模->仿真->行走参数调整->图像识别模块->蓝牙传输控制模块->Arduino蓝牙接收与控制代码->实际测试

## 其他注意

### 测试舵机

先用**便宜的板**来测试*单个*舵机,然后再用机器人上安好的舵机控制板
