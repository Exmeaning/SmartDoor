在zeabur上 尝试按以下方式部署一个中转服务器
### 因为没有公网IP 实际上是需要作为HTTP服务器 接受发送客户端 和 接受客户端
### 发送客户端 和 接受客户端 都会采用1s的轮询机制 实现类似于“双向”通信
### 因为接受客户端的NTP有问题 你需要额外部署一个查询时间戳的端点
### 注意需要控制服务器内存 以确保不会内存泄漏 节省PAAS费用
以下是接受客户端的文档
# K230 自动开门机 HTTP API 文档

## 1. 概述

### 1.1 基础信息
- **协议**: HTTP/1.0
- **编码**: UTF-8
- **数据格式**: JSON (除图片上传外)
- **认证方式**: Bearer Token
- **服务器地址**: 配置在 `config.json` 中的 `http_server_url`
- **轮询间隔**: 1秒（可配置）
- **心跳间隔**: 30秒（可配置）

### 1.2 认证机制

#### Token 认证
所有请求都需要在 Header 中包含认证令牌：

```
Authorization: Bearer <token>
```

**Token 配置方式**：
1. 在 `config.json` 中设置 `network.http_token` 字段
2. Token 格式建议：
   - UUID格式: `550e8400-e29b-41d4-a716-446655440000`
   - JWT格式: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
   - 自定义密钥: `your-secret-token-here`

**Token 验证失败响应**：
```json
{
    "success": false,
    "code": 401,
    "message": "Unauthorized: Invalid or missing token",
    "timestamp": 1699999999
}
```

### 1.3 双向通信机制

#### 轮询机制（Polling）
设备通过定期轮询实现"准实时"的双向通信：

1. **设备 → 服务器**（主动上报）：
   - 事件上报（访问记录、告警等）
   - 心跳保活
   - 状态同步

2. **服务器 → 设备**（轮询获取）：
   - 设备每秒轮询一次 `/api/device/poll` 端点
   - 服务器返回待执行的命令队列
   - 设备执行命令并上报结果

**轮询流程图**：
```
设备                          服务器
 |                              |
 |----GET /api/device/poll----->|
 |<---返回命令队列或空响应--------|
 |                              |
 |----执行命令------------------->|
 |----POST /api/event(结果)---->|
 |<---确认响应-------------------|
 |                              |
 |----(1秒后重复)--------------->|
```

### 1.4 通用响应格式

```json
{
    "success": true,
    "code": 200,
    "message": "操作成功",
    "data": {},
    "timestamp": 1699999999
}
```

### 1.4 错误码定义

| 错误码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 401 | 认证失败 |
| 403 | 权限不足 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |
| 503 | 服务暂时不可用 |

---

## 2. API 接口详细说明

### 2.1 设备轮询接口（核心接口）

**端点**: `GET /api/device/poll`

**功能**: 设备定期轮询服务器，获取待执行的命令（实现服务器到设备的通信）

**请求频率**: 每秒1次

**请求头**:
```
Authorization: Bearer <token>
```

**请求参数**:
```json
{
    "device_id": "K230_001",  // 可选，通过URL参数传递
    "last_poll": 1699999998   // 可选，上次轮询时间戳
}
```

**响应示例（有命令）**:
```json
{
    "success": true,
    "code": 200,
    "message": "Command pending",
    "data": {
        "command": "open_door",
        "command_id": "cmd_1699999999_001",
        "params": {
            "duration": 5,
            "reason": "Remote control"
        },
        "priority": "normal",  // low/normal/high/urgent
        "timestamp": 1699999999
    },
    "timestamp": 1699999999
}
```

**响应示例（无命令）**:
```json
{
    "success": true,
    "code": 200,
    "message": "No pending commands",
    "data": null,
    "timestamp": 1699999999
}
```

#### 支持的命令类型

| 命令 | 说明 | 参数 |
|------|------|------|
| `open_door` | 远程开门 | `duration`: 开门持续时间（秒） |
| `register_face` | 注册人脸 | `name`: 人员姓名 |
| `delete_face` | 删除人脸 | `name`: 人员姓名 |
| `list_faces` | 列出所有人脸 | 无 |
| `update_config` | 更新配置 | `key`: 配置键, `value`: 配置值 |
| `capture_image` | 捕获当前画面 | 无 |
| `reboot` | 重启设备 | `delay`: 延迟时间（秒） |
| `get_status` | 获取设备状态 | 无 |
| `sync_database` | 同步人脸数据库 | 无 |
| `emergency_open` | 紧急开门 | `duration`: 持续时间 |

**命令执行反馈**:
设备执行命令后，应通过事件上报接口反馈执行结果：

```json
{
    "device_id": "K230_001",
    "event_type": "command_executed",
    "timestamp": 1699999999,
    "data": {
        "command_id": "cmd_1699999999_001",
        "command": "open_door",
        "success": true,
        "message": "Door opened successfully",
        "execution_time": 0.125
    }
}
```

### 2.2 心跳接口

**端点**: `GET /api/heartbeat`

**功能**: 设备定期发送心跳，保持连接活跃

**请求头**:
```
Authorization: Bearer <token>
```

**请求参数**: 无

**响应示例**:
```json
{
    "success": true,
    "code": 200,
    "message": "OK",
    "data": {
        "server_time": 1699999999,
        "config_version": "1.0.0",
        "update_available": false
    },
    "timestamp": 1699999999
}
```

---

### 2.2 事件上报接口

**端点**: `POST /api/event`

**功能**: 上报各类设备事件

**请求头**:
```
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体**:
```json
{
    "device_id": "K230_001",
    "event_type": "access_granted",
    "timestamp": 1699999999,
    "data": {
        // 事件特定数据
    }
}
```

#### 2.2.1 设备上线事件 (device_online)

**请求示例**:
```json
{
    "device_id": "K230_001",
    "event_type": "device_online",
    "timestamp": 1699999999,
    "data": {
        "device_id": "K230_001",
        "device_name": "前门",
        "firmware_version": "1.0.0",
        "ip_address": "192.168.1.100",
        "mac_address": "AA:BB:CC:DD:EE:FF"
    }
}
```

#### 2.2.2 设备离线事件 (device_offline)

**请求示例**:
```json
{
    "device_id": "K230_001",
    "event_type": "device_offline",
    "timestamp": 1699999999,
    "data": {
        "reason": "shutdown"  // shutdown/network_error/power_off
    }
}
```

#### 2.2.3 访问授权事件 (access_granted)

**请求示例**:
```json
{
    "device_id": "K230_001",
    "event_type": "access_granted",
    "timestamp": 1699999999,
    "data": {
        "person": "张三",
        "method": "face",
        "confidence": 0.95,
        "door_id": "main",
        "time": 1699999999,
        "face_image_url": "/api/upload/granted/xxx.jpg"  // 可选
    }
}
```

#### 2.2.4 访问拒绝事件 (access_denied)

**请求示例**:
```json
{
    "device_id": "K230_001",
    "event_type": "access_denied",
    "timestamp": 1699999999,
    "data": {
        "person": "unknown",
        "reason": "unregistered",  // unregistered/low_confidence/blacklist
        "door_id": "main",
        "time": 1699999999,
        "face_image_url": "/api/upload/denied/xxx.jpg"  // 可选
    }
}
```

#### 2.2.5 门锁事件 (door_locked/door_opened)

**请求示例**:
```json
{
    "device_id": "K230_001",
    "event_type": "door_locked",
    "timestamp": 1699999999,
    "data": {
        "door_id": "main",
        "action": "lock",  // lock/unlock
        "method": "auto",  // auto/manual/remote
        "time": 1699999999
    }
}
```

#### 2.2.6 紧急开门事件 (emergency_open)

**请求示例**:
```json
{
    "device_id": "K230_001",
    "event_type": "emergency_open",
    "timestamp": 1699999999,
    "data": {
        "door_id": "main",
        "triggered_by": "button",  // button/remote/system
        "time": 1699999999
    }
}
```

**通用响应**:
```json
{
    "success": true,
    "code": 200,
    "message": "事件已接收",
    "data": {
        "event_id": "evt_1699999999_001",
        "received_at": 1699999999
    },
    "timestamp": 1699999999
}
```

---

### 2.3 图片上传接口

#### 2.3.1 上传授权通过图片

**端点**: `POST /api/upload/granted`

**功能**: 上传授权通过的人脸图片

**请求头**:
```
Authorization: Bearer <token>
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW
```

**请求体** (multipart/form-data):
```
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="file"; filename="granted_张三_1699999999.jpg"
Content-Type: image/jpeg

<二进制图片数据>
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="metadata"

{
    "person": "张三",
    "confidence": 0.95,
    "timestamp": 1699999999,
    "device_id": "K230_001"
}
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

#### 2.3.2 上传拒绝访问图片

**端点**: `POST /api/upload/denied`

**功能**: 上传拒绝访问的人脸图片（用于安全审计）

**请求格式同上**

**响应示例**:
```json
{
    "success": true,
    "code": 200,
    "message": "图片上传成功",
    "data": {
        "file_id": "img_1699999999_001",
        "file_path": "/storage/images/2024/01/granted_张三_1699999999.jpg",
        "file_size": 102400,
        "url": "https://server.com/images/xxx.jpg"
    },
    "timestamp": 1699999999
}
```

---

## 3. 设备配置更新

为了支持Token认证，需要在 `config.json` 中添加以下配置：

```json
{
    "network": {
        "wifi_ssid": "AzumaSeren",
        "wifi_password": "masheihanjian",
        "http_server_url": "http://10.100.228.5:8080",
        "http_token": "your-secret-token-here",
        "http_timeout": 10,
        "ntp_sync": true,
        "auto_reconnect": true,
        "reconnect_interval": 5
    }
}
```

---

## 4. 人脸注册相关接口

### 4.1 注册人脸

#### 方式一：Multipart上传（推荐）

**端点**: `POST /api/face/register`

**功能**: 注册新的人脸数据（二进制格式）

**请求头**:
```
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**请求体** (multipart/form-data):
- `image`: 人脸图片文件（JPEG格式，二进制）
- `feature`: 人脸特征文件（二进制）
- `metadata`: 元数据（JSON字符串）

**元数据格式**:
```json
{
    "device_id": "K230_001",
    "person_name": "张三",
    "timestamp": 1699999999,
    "feature_size": 512
}
```

#### 方式二：JSON上传（十六进制编码）

**端点**: `POST /api/face/register/hex`

**功能**: 注册新的人脸数据（十六进制编码格式）

**请求头**:
```
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体**:
```json
{
    "device_id": "K230_001",
    "person_name": "张三",
    "timestamp": 1699999999,
    "image_hex": "ffd8ffe000104a464946...",  // JPEG图像的十六进制编码
    "feature_hex": "3f8000003f800000...",    // 特征向量的十六进制编码
    "image_size": 102400,
    "feature_size": 512
}
```

**注意**: 十六进制编码会使数据量翻倍，建议在网络条件较好时使用

**响应示例**:
```json
{
    "success": true,
    "code": 200,
    "message": "Face registered successfully",
    "data": {
        "face_id": "face_1699999999_001",
        "person_name": "张三",
        "device_id": "K230_001"
    },
    "timestamp": 1699999999
}
```

### 4.2 删除人脸

**端点**: `DELETE /api/face/{person_name}`

**功能**: 删除指定人员的人脸数据

**响应示例**:
```json
{
    "success": true,
    "code": 200,
    "message": "Face deleted successfully",
    "data": {
        "person_name": "张三",
        "deleted_count": 1
    },
    "timestamp": 1699999999
}
```

### 4.3 获取人脸列表

**端点**: `GET /api/face/list`

**功能**: 获取所有已注册的人脸列表

**请求参数** (Query):
- `device_id`: 设备ID（可选，筛选特定设备的人脸）
- `page`: 页码（默认1）
- `limit`: 每页数量（默认50）

**响应示例**:
```json
{
    "success": true,
    "code": 200,
    "message": "Success",
    "data": {
        "faces": [
            {
                "name": "张三",
                "face_id": "face_1699999999_001",
                "device_id": "K230_001",
                "registered_at": 1699999999
            },
            {
                "name": "李四",
                "face_id": "face_1699999998_002",
                "device_id": "K230_001",
                "registered_at": 1699999998
            }
        ],
        "total": 2,
        "page": 1,
        "limit": 50
    },
    "timestamp": 1699999999
}
```

### 4.4 下载人脸特征

**端点**: `GET /api/face/download/{person_name}`

**功能**: 下载指定人员的人脸特征文件

**响应格式选项**:

1. **二进制格式**（推荐）:
   - Content-Type: `application/octet-stream`
   - 直接返回二进制特征数据

2. **十六进制格式**:
   - Content-Type: `application/json`
   ```json
   {
       "success": true,
       "code": 200,
       "data": {
           "person_name": "张三",
           "feature_hex": "48656c6c6f20576f726c64...",  // 十六进制编码
           "feature_size": 512
       }
   }
   ```

**注意**: K230设备不支持Base64编码，请使用二进制或十六进制格式

### 4.5 同步人脸数据库

**端点**: `POST /api/face/sync`

**功能**: 批量同步设备和服务器的人脸数据库

**请求体**:
```json
{
    "device_id": "K230_001",
    "local_faces": ["张三", "李四", "王五"],
    "request_missing": true  // 是否请求下载缺失的人脸
}
```

**响应示例**:
```json
{
    "success": true,
    "code": 200,
    "message": "Sync completed",
    "data": {
        "to_download": ["赵六", "钱七"],  // 服务器有但设备没有
        "to_upload": ["王五"],             // 设备有但服务器没有
        "synced": 5,
        "total": 7
    },
    "timestamp": 1699999999
}
```

---

## 5. 实施指南

### 5.1 Token 配置

在 `config.json` 中配置Token：
```json
{
    "network": {
        "wifi_ssid": "YourWiFiSSID",
        "wifi_password": "YourWiFiPassword",
        "http_server_url": "http://192.168.1.100:8080",
        "http_token": "550e8400-e29b-41d4-a716-446655440000",
        "http_timeout": 10,
        "poll_interval": 1,     // 轮询间隔（秒）
        "heartbeat_interval": 30 // 心跳间隔（秒）
    }
}
```

### 5.2 服务器端实现建议

#### 5.2.1 命令队列管理

服务器应为每个设备维护一个命令队列：

```python
# 示例Python实现
class DeviceCommandQueue:
    def __init__(self):
        self.queues = {}  # {device_id: deque()}
    
    def add_command(self, device_id, command):
        """添加命令到设备队列"""
        if device_id not in self.queues:
            self.queues[device_id] = deque()
        
        command_obj = {
            'command_id': f'cmd_{int(time.time())}_{uuid.uuid4().hex[:8]}',
            'command': command['command'],
            'params': command.get('params', {}),
            'priority': command.get('priority', 'normal'),
            'timestamp': int(time.time()),
            'status': 'pending'
        }
        
        self.queues[device_id].append(command_obj)
        return command_obj['command_id']
    
    def get_next_command(self, device_id):
        """获取设备的下一个待执行命令"""
        if device_id in self.queues and self.queues[device_id]:
            return self.queues[device_id].popleft()
        return None
```

#### 5.2.2 Token 验证中间件

```python
# Flask示例
from functools import wraps
from flask import request, jsonify

def require_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # 从Header获取Token
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                pass
        
        if not token:
            return jsonify({
                'success': False,
                'code': 401,
                'message': 'Missing token'
            }), 401
        
        # 验证Token
        if not verify_token(token):
            return jsonify({
                'success': False,
                'code': 401,
                'message': 'Invalid token'
            }), 401
        
        return f(*args, **kwargs)
    
    return decorated_function
```

#### 5.2.3 轮询端点实现

```python
@app.route('/api/device/poll', methods=['GET'])
@require_token
def device_poll():
    """设备轮询端点"""
    device_id = request.args.get('device_id', 'unknown')
    
    # 更新设备最后活跃时间
    update_device_last_seen(device_id)
    
    # 获取下一个命令
    command = command_queue.get_next_command(device_id)
    
    if command:
        return jsonify({
            'success': True,
            'code': 200,
            'message': 'Command pending',
            'data': command,
            'timestamp': int(time.time())
        })
    else:
        return jsonify({
            'success': True,
            'code': 200,
            'message': 'No pending commands',
            'data': None,
            'timestamp': int(time.time())
        })
```

#### 5.2.4 处理十六进制编码的数据

```python
@app.route('/api/face/register/hex', methods=['POST'])
@require_token
def register_face_hex():
    """接收十六进制编码的人脸数据"""
    data = request.json
    
    # 解码十六进制数据
    try:
        image_bytes = bytes.fromhex(data['image_hex'])
        feature_bytes = bytes.fromhex(data['feature_hex'])
        
        # 保存到文件或数据库
        person_name = data['person_name']
        save_face_data(person_name, image_bytes, feature_bytes)
        
        return jsonify({
            'success': True,
            'code': 200,
            'message': 'Face registered successfully',
            'data': {
                'person_name': person_name,
                'image_size': len(image_bytes),
                'feature_size': len(feature_bytes)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'code': 400,
            'message': f'Invalid hex data: {str(e)}'
        }), 400

@app.route('/api/face/download/<person_name>', methods=['GET'])
@require_token  
def download_face(person_name):
    """下载人脸特征（支持多种格式）"""
    accept_format = request.args.get('format', 'binary')
    
    # 获取特征数据
    feature_bytes = get_face_feature(person_name)
    if not feature_bytes:
        return jsonify({
            'success': False,
            'code': 404,
            'message': 'Face not found'
        }), 404
    
    if accept_format == 'hex':
        # 返回十六进制编码的JSON
        return jsonify({
            'success': True,
            'code': 200,
            'data': {
                'person_name': person_name,
                'feature_hex': feature_bytes.hex(),
                'feature_size': len(feature_bytes)
            }
        })
    else:
        # 返回二进制数据
        return Response(
            feature_bytes,
            mimetype='application/octet-stream',
            headers={
                'Content-Disposition': f'attachment; filename={person_name}.bin'
            }
        )
```

### 5.3 数据编码说明

#### K230设备限制
由于K230的MicroPython环境限制，不支持以下编码：
- ❌ Base64编码（无`base64`模块）
- ❌ 复杂的压缩算法

#### 支持的编码方式
1. **二进制直传**（推荐）：
   - 适用于multipart/form-data
   - 传输效率最高
   - 无编码开销

2. **十六进制编码**：
   - 适用于JSON传输
   - 简单可靠
   - 数据量翻倍

3. **实现示例**：
```python
# MicroPython中的十六进制编码
def to_hex(data):
    """字节转十六进制字符串"""
    return ''.join(['%02x' % b for b in data])

def from_hex(hex_str):
    """十六进制字符串转字节"""
    return bytes.fromhex(hex_str)
```

### 5.4 安全建议

1. **Token管理**：
   - 使用强随机Token（至少32字符）
   - 定期更换Token
   - 每个设备使用独立的Token

2. **HTTPS支持**：
   - 生产环境建议使用HTTPS
   - 可使用自签名证书或Let's Encrypt

3. **访问限制**：
   - 实施请求频率限制
   - 记录异常访问行为
   - 设置IP白名单（可选）

4. **数据传输**：
   - 优先使用二进制直传
   - 大数据使用分块传输
   - 考虑使用简单的XOR加密

### 5.4 性能优化

1. **轮询优化**：
   - 空闲时可降低轮询频率
   - 有命令时立即响应，无命令时可使用长轮询

2. **批量处理**：
   - 支持批量命令下发
   - 批量事件上报

3. **缓存机制**：
   - 缓存人脸特征数据
   - 缓存设备状态信息

4. **连接池**：
   - 使用HTTP连接池复用连接
   - 减少TCP握手开销

---

## 6. 故障排查

### 6.1 常见问题

#### Token认证失败
- 检查Token是否正确配置
- 确认Authorization header格式正确
- 查看服务器端Token验证逻辑

#### 轮询无响应
- 检查网络连接状态
- 确认服务器端轮询端点正常
- 查看设备日志中的错误信息

#### 命令执行失败
- 检查命令格式是否正确
- 确认设备支持该命令
- 查看命令执行日志

### 6.2 调试模式

在 `config.json` 中启用调试模式：
```json
{
    "system": {
        "debug_mode": true,
        "log_level": "DEBUG"
    }
}
```

### 6.3 日志位置

- 设备端日志: `/logs/` 目录
- 服务器端日志: 根据服务器配置
- 网络通信日志: `/logs/network.log`




发送客户端没有相关文档 你可能要自行撰写一个

同时，您需要在服务器端提供一个时间戳接口。这里是一个简单的Python Flask服务器示例：(实际在zeabur使用nodejs部署以提高效率)

```Python

# server_time_endpoint.py (服务器端)
from flask import Flask, jsonify
import time

app = Flask(__name__)

@app.route('/api/time', methods=['GET'])
def get_time():
    """返回当前时间戳"""
    return jsonify({
        'timestamp': int(time.time()),
        'datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```
请按此格式输出：

FILE: 路径
SEARCH:
<<<
原代码
>>>
REPLACE:
<<<
新代码
>>>

⚠️ 重要：
1. 创建新文件时，SEARCH的<<<和>>>之间必须有空行
2. SEARCH内容要包含足够上下文（5-10行）确保唯一
3. 如果代码重复出现，加ANCHOR_BEFORE定位
4. 如果需要删除较长的代码，可以尝试注释掉开头和结尾来实现

