# K230 智能门锁系统 / K230 Smart Door Lock System

一个本科阶段的小项目 利用K230开发板 MicroPython 实现智能开门

- 自娱自乐 正在逐渐迭代 反正没人看
- 部分代码依赖AI生成 风险自辩（甚至连这个都是AI顺手给我生成的）

## 功能特性 / Features

- 🔍 **人脸检测与识别** - 实时人脸检测，支持多人脸识别
- 🌐 **网络连接** - WiFi连接，TCP通信，NTP时间同步
- 🔐 **门禁控制** - 步进电机控制，自动开关门
- 🔊 **音频反馈** - 语音提示，蜂鸣器警告
- 💤 **智能休眠** - 无人时自动休眠，降低功耗
- 📝 **日志记录** - 完整的访问日志，云端同步
- 🔄 **多线程** - 异步处理，系统响应快速

## 系统架构 / System Architecture

```
K230 Controller/
├── main.py              # 主程序入口
├── boot.py              # 启动脚本
├── config.json          # 系统配置
├── core/               # 核心模块
│   ├── face_detect.py      # 人脸检测
│   ├── network_manager.py  # 网络管理
│   ├── motor_control.py    # 电机控制
│   └── audio_manager.py    # 音频管理
├── utils/              # 工具模块
│   ├── config_loader.py    # 配置加载
│   ├── logger.py           # 日志管理
│   └── sleep_manager.py    # 休眠管理
├── modules/            # 业务模块
│   └── door_controller.py  # 门禁控制
└── libs/               # 库文件（系统提供）
```

## 硬件连接 / Hardware Connection

### GPIO引脚分配

| 功能 | 引脚 | 说明 |
|------|------|------|
| 步进电机PUL | GPIO42 | 脉冲信号 |
| 步进电机DIR | GPIO33 | 方向信号 |
| 步进电机EN | GPIO32 | 使能信号 |
| 扬声器 | GPIO35 | PWM音频输出 |

### 电机参数

- 输入电压：3.3V-28V
- 步进信号频率：2-400KHz
- 步进脉冲宽度：250ns
- 方向信号宽度：62.5us

## 配置说明 / Configuration

编辑 `config.json` 文件进行系统配置：

```json
{
    "network": {
        "wifi_ssid": "YOUR_WIFI_NAME",
        "wifi_password": "YOUR_PASSWORD"
    },
    "face_recognition": {
        "confidence_threshold": 0.5,
        "recognition_threshold": 0.65
    }
}
```

## 使用方法 / Usage

### 1. 准备工作

1. 将代码上传到 `/sdcard/` 目录
2. 确保模型文件在 `/sdcard/kmodel/` 目录
3. 配置WiFi信息

### 2. 启动系统

系统会自动运行 `boot.py`，然后启动 `main.py`

### 3. 人脸注册

```python
# 将人脸照片放在指定目录
/data/photo/YOUR_ID/
# 系统会自动注册
```

### 4. 功能测试

系统启动后会自动进行测试：
- 音频测试 - 播放不同音调
- 电机测试 - 正反转测试
- 开关门测试 - 完整流程测试

## API说明 / API Reference

### DoorController

```python
door_ctrl = DoorController()

# 授权访问
door_ctrl.grant_access(person_name, method, confidence)

# 拒绝访问  
door_ctrl.deny_access(reason, person_name)

# 紧急开门
door_ctrl.emergency_open()

# 锁门
door_ctrl.lock_door()
```

### NetworkManager

```python
network = NetworkManager()

# 连接WiFi
network.connect_wifi()

# 检查连接
network.check_connection()

# TCP通信
network.create_tcp_client()
network.send_tcp_data(data)
```

## 日志说明 / Logging

日志文件保存在 `/sdcard/logs/` 目录，格式：`door_YYYYMMDD.log`

日志级别：
- DEBUG - 调试信息
- INFO - 一般信息
- WARNING - 警告信息
- ERROR - 错误信息
- CRITICAL - 严重错误

## 注意事项 / Notes

1. **电源管理** - 确保电源稳定，建议使用5V/3A电源
2. **散热** - 长时间运行需要散热片
3. **安全** - 请勿在生产环境直接使用，需要增加安全机制
4. **隐私** - 人脸数据需要加密存储

## 故障排除 / Troubleshooting

### WiFi连接失败
- 检查SSID和密码是否正确
- 确认路由器2.4G频段开启
- 查看日志文件获取详细错误

### 人脸识别不准确
- 调整置信度阈值
- 确保光线充足
- 重新注册人脸

### 电机不工作
- 检查GPIO连接
- 确认电机驱动器供电
- 测试引脚输出

## 开发计划 / Development Plan

- [ ] 云端API集成
- [ ] 讯飞语音识别
- [ ] 人脸活体检测
- [ ] 手机APP控制
- [ ] 访客模式
- [ ] 多重认证

## 贡献 / Contributing

欢迎提交Issue和Pull Request！

## 许可证 / License

MIT License

## 联系方式 / Contact

项目维护：K230 Development Team

---

**安全提示：本系统仅供学习研究使用，请勿用于实际门锁控制。**