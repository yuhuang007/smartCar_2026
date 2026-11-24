# UART1 串口指令文档

## 1. 文档说明

本文档适用于工程 `E02_02_drv8701e_double_motor_contro_demo`。

当前工程通过 `UART1` 实现以下功能：

- 双电机前进、倒车、普通停止和急停控制
- 基于编码器反馈的速度环控制
- 舵机角度控制
- 编码器测速状态回传
- `P7.3` 按键触发串口发送脚本启动命令给上位机

本文档内容以当前 [main.c](file:///e:/STM32/STC/Motherboard_Demo/E02_02_drv8701e_double_motor_contro_demo/user/main.c) 的实际实现为准。

## 2. 串口参数

- 串口号：`UART1`
- 波特率：`115200`
- 数据位：`8`
- 停止位：`1`
- 校验位：`无`
- 命令结束：建议发送 `\r\n`

## 3. 硬件连接

### 3.1 串口

- `UART1_TX -> P31`
- `UART1_RX -> P30`
- `GND -> GND`

### 3.2 编码器

- `ENC_PULSE -> P04`
- `ENC_DIR -> P53`
- `VCC -> 3.3V`
- `GND -> GND`

### 3.3 舵机

- `SERVO_PWM -> P33`
- `VCC -> 舵机电源`
- `GND -> GND`

### 3.4 按键

- `KEY_IN -> P73`
- 按键另一端接 `GND`
- 输入模式：内部上拉输入
- 触发方式：低电平按下有效

## 4. 指令格式

- 指令采用 ASCII 字符串
- 支持大写和小写
- 每条指令建议带回车或回车换行结束
- 当前支持的命令有：`F`、`S`、`Dxx`、`Axx`、`B`、`E`
- 指令缓冲区长度为 `16` 字节，超出会返回错误

## 5. 当前控制方式说明

当前代码已经不是简单的固定占空比开环控制，而是：

- `Dxx` 用来设置速度目标等级
- 程序根据 `Dxx` 自动换算目标编码器计数 `target`
- `pit_handler()` 中根据编码器反馈做速度环调节
- 实际输出到电机的 PWM 占空比是 `out`

也就是说：

- `D10` 不再只是“固定输出 10% 占空比”
- 而是“以 10% 占空比对应的实测编码器值作为目标速度基准”

当前速度环基准参数见 [main.c](file:///e:/STM32/STC/Motherboard_Demo/E02_02_drv8701e_double_motor_contro_demo/user/main.c#L95-L103)：

```c
#define SPEED_BASE_DUTY     (10)
#define SPEED_BASE_COUNT    (1260)
```

含义是：

- 以 `10%` 占空比时实测的编码器稳定值 `1260 count/100ms` 作为基准目标

因此当前目标速度换算关系约为：

- `D10 -> target = 1260`
- `D11 -> target = 1386`
- `D12 -> target = 1512`
- `D15 -> target = 1890`
- `D20 -> target = 2520`

## 6. 指令总览

| 指令 | 功能 | 参数 | 示例 |
| --- | --- | --- | --- |
| `F` | 前进运行 | 无 | `F` |
| `S` | 普通停止 | 无 | `S` |
| `Dxx` | 设置速度目标等级 | `0~100` | `D10` |
| `Axx` | 设置舵机角度 | `0~180` | `A90` |
| `B` | 倒车运行 | 无 | `B` |
| `E` | 急停刹车 | 无 | `E` |

## 7. 指令详细说明

### 7.1 `F` 前进运行

- 功能：让电机以前进方向运行
- 参数：无
- 方向行为：将 `motor_direction` 设为前进
- 运行行为：使能电机，并重置速度环积分项
- 控制方式：按当前 `Dxx` 对应的目标速度运行

执行效果：

- 左右电机按前进方向输出
- 启动时会根据当前 `Dxx` 重新装载目标速度
- 速度环自动调整实际输出占空比

示例：

```text
D10
F
```

返回示例：

```text
OK F target=1260 duty=10%
```

说明：

- `target` 是当前目标编码器计数
- `duty` 是当前命令给定值，不是实际输出值

### 7.2 `S` 普通停止

- 功能：普通停机
- 参数：无
- 行为：关闭电机输出，清零速度环积分

执行效果：

- 电机 PWM 输出为 0
- 小车会因惯性产生一定滑行
- 不执行反向制动

示例：

```text
S
```

返回示例：

```text
OK S
```

说明：

- `S` 是普通停止，不是急停
- 地面跑动时执行 `S` 后，车辆可能继续滑行一小段

### 7.3 `Dxx` 设置速度目标等级

- 功能：设置电机目标速度等级
- 参数范围：`0~100`
- 格式：`D` 后直接跟十进制数字，例如 `D10`

当前实现说明：

- 程序先根据 `Dxx` 计算目标编码器计数 `target`
- 速度环再根据 `target` 和实际反馈值自动调节输出占空比

典型示例：

```text
D10
```

返回示例：

```text
OK D10 target=1260
```

再例如：

```text
D12
```

返回示例：

```text
OK D12 target=1512
```

注意事项：

- 现在 `Dxx` 更接近“速度给定”，不是单纯固定 PWM
- 参数只能是数字
- 超过 `0~100` 范围会被判为非法命令
- 若重新标定了 `10%` 对应的实测速度，需要同步修改 [main.c](file:///e:/STM32/STC/Motherboard_Demo/E02_02_drv8701e_double_motor_contro_demo/user/main.c#L99-L100) 中的 `SPEED_BASE_DUTY` 和 `SPEED_BASE_COUNT`

### 7.4 `Axx` 设置舵机角度

- 功能：设置舵机目标角度
- 参数范围：`0~180`
- 格式：`A` 后直接跟十进制数字，例如 `A90`

执行效果：

- 更新舵机目标角度
- 立即刷新舵机 PWM 输出

示例：

```text
A90
A120
```

返回示例：

```text
OK A90
OK A120
```

说明：

- 当前打印的是目标角度，不是反馈角度
- 虽然程序允许 `0~180`，但实际机械结构未必支持全范围

### 7.5 `B` 倒车运行

- 功能：让电机以倒车方向运行
- 参数：无
- 方向行为：将 `motor_direction` 设为倒车
- 运行行为：使能电机，并重置速度环积分项

执行效果：

- 电机按倒车方向运行
- 速度环按当前目标值自动调节实际输出占空比

示例：

```text
D10
B
```

返回示例：

```text
OK B target=1260 duty=10%
```

### 7.6 `E` 急停刹车

- 功能：执行急停
- 参数：无
- 当前实现方式：两段反向制动

当前急停参数见 [main.c](file:///e:/STM32/STC/Motherboard_Demo/E02_02_drv8701e_double_motor_contro_demo/user/main.c#L93-L98)：

- 第一段强刹：`28% / 60ms`
- 第二段弱刹：`10% / 40ms`

执行逻辑：

1. 若当前电机未运行，则直接返回
2. 设置 `brake_busy = 1`
3. 先反向输出第一段强刹脉冲
4. 再反向输出第二段弱刹脉冲
5. 最后停止电机并退出急停状态

返回示例：

```text
OK E brake1=28% 60ms brake2=10% 40ms
```

说明：

- `E` 是急停，不是普通停止
- 适合车辆跑动时快速减小滑行距离
- 当速度提高后，急停参数可能仍需继续调试

### 7.7 `P73` 按键触发串口发送 `RUN`

- 功能：按下 `P73` 按键后，通过 `UART1` 主动给上位机发送一条脚本启动命令
- 触发引脚：`P73`
- 触发条件：低电平按下
- 当前发送内容：`RUN\r\n`

相关参数见 [main.c](file:///e:/STM32/STC/Motherboard_Demo/E02_02_drv8701e_double_motor_contro_demo/user/main.c#L104-L108)：

```c
#define KEY_PIN             (IO_P73)
#define KEY_ACTIVE_LEVEL    (GPIO_LOW)
#define KEY_RELEASE_LEVEL   (GPIO_HIGH)
#define KEY_DEBOUNCE_MS     (20)
#define KEY_UART_COMMAND    ("RUN\r\n")
```

执行逻辑：

1. `P73` 配置为内部上拉输入
2. 默认松开为高电平
3. 检测到按下后先延时 `20ms` 消抖
4. 消抖后若仍为低电平，则通过 `UART1` 发送 `RUN\r\n`
5. 长按不会重复连续发送，需松开后再次按下才会再次触发

串口发送效果：

```text
RUN
```

上电提示：

```text
key ready: P73 low -> send RUN
```

## 8. 状态回传说明

当前代码的周期状态打印已经改成英文格式：

```text
encoder_cnt=xxx rpm=xxx target=xxx cmd=xx% out=xx% state=run/stop dir=forward/reverse angle=xx
```

字段含义如下：

- `encoder_cnt`：当前测速周期内采集到的编码器计数
- `rpm`：根据编码器计数换算得到的转速
- `target`：当前目标编码器计数
- `cmd`：串口命令给定值，也就是当前 `Dxx`
- `out`：速度环当前实际输出到电机的 PWM 占空比
- `state`：`run` 表示运行，`stop` 表示停止
- `dir`：`forward` 表示前进，`reverse` 表示倒车
- `angle`：当前舵机目标角度

状态打印示例：

```text
encoder_cnt=-1254 rpm=-1929 target=1260 cmd=10% out=15% state=run dir=forward angle=90
```

这个例子的含义是：

- 当前目标速度是 `1260`
- 速度环当前把实际输出拉到了 `15%`
- 方向为前进
- 舵机角度为 `90`

## 9. 错误返回

### 9.1 非法指令

当输入命令不符合当前支持格式时，返回：

```text
ERR xxx
```

示例：

```text
ERR X12
```

### 9.2 指令过长

当输入指令超过缓冲区长度时，返回：

```text
ERR command too long
```

## 10. 推荐调试顺序

### 10.1 速度环基础测试

```text
D10
F
```

观察：

- `target`
- `encoder_cnt`
- `out`

若 `encoder_cnt` 长期低于 `target`，说明输出偏小；若长期远高于 `target`，说明环路参数偏猛。

### 10.2 倒车速度环测试

```text
D10
B
```

### 10.3 急停测试

```text
D10
F
E
```

```text
D10
B
E
```

### 10.4 舵机测试

```text
A90
A120
A60
```

### 10.5 按键触发上位机脚本测试

上电后直接按下 `P73`：

```text
RUN
```

## 11. 当前支持功能汇总

```text
F        前进运行
S        普通停止
Dxx      设置速度目标等级
Axx      设置舵机角度
B        倒车运行
E        急停刹车
P73      低电平按下后发送 RUN\r\n 给上位机
```
