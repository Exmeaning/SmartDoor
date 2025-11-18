# K230 智能门禁系统 HTTP API 文档

## 1. 概述

### 1.1 基础信息
- **协议**: HTTP/1.0
- **编码**: UTF-8
- **数据格式**: JSON (除图片上传外)
- **认证方式**: Bearer Token
- **服务器地址**: 配置在 `config.json` 中的 `http_server_url`

### 1.2 认证机制

所有请求都需要在 Header 中包含认证令牌：

```
Authorization: Bearer <token>
```

Token 应在设备初始化时配置，建议使用 JWT 或 UUID 格式。

### 1.3 通用响应格式

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

### 2.1 心跳接口

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

## 4. NetworkManager 更新

需要在 `network_manager.py` 中添加 Token 支持：

FILE: C:\Python_Project\K230 Controller\core\network_manager.py
SEARCH:
<<<
    def init_http_client(self):
        """初始化HTTP客户端配置 / Initialize HTTP client configuration"""
        self.http_server_url = self.config.get('network.http_server_url')
        self.http_timeout = self.config.get('network.http_timeout', 10)
        
        if not self.http_server_url:
            self.logger.error("HTTP服务器URL未配置")
            return False
        
        self.logger.info(f"HTTP客户端配置: {self.http_server_url}")
        return True
>>>
REPLACE:
<<<
    def init_http_client(self):
        """初始化HTTP客户端配置 / Initialize HTTP client configuration"""
        self.http_server_url = self.config.get('network.http_server_url')
        self.http_timeout = self.config.get('network.http_timeout', 10)
        self.http_token = self.config.get('network.http_token', '')
        
        if not self.http_server_url:
            self.logger.error("HTTP服务器URL未配置")
            return False
        
        if not self.http_token:
            self.logger.warning("HTTP Token未配置，可能导致认证失败")
        
        self.logger.info(f"HTTP客户端配置: {self.http_server_url}")
        return True
>>>

FILE: C:\Python_Project\K230 Controller\core\network_manager.py
SEARCH:
<<<
    def _build_http_request(self, method, path, host, data=None, headers=None):
        """构建HTTP请求 / Build HTTP request"""
        request = f"{method} {path} HTTP/1.0\r\n"
        request += f"Host: {host}\r\n"
        
        if headers:
            for key, value in headers.items():
                request += f"{key}: {value}\r\n"
>>>
REPLACE:
<<<
    def _build_http_request(self, method, path, host, data=None, headers=None):
        """构建HTTP请求 / Build HTTP request"""
        request = f"{method} {path} HTTP/1.0\r\n"
        request += f"Host: {host}\r\n"
        
        # 添加认证Token
        if hasattr(self, 'http_token') and self.http_token:
            request += f"Authorization: Bearer {self.http_token}\r\n"
        
        if headers:
            for key, value in headers.items():
                request += f"{key}: {value}\r\n"
>>>

---

## 5. Python 服务器示例代码

以下是一个简单的 Python Flask 服务器示例，用于接收 K230 设备的请求：

```python
from flask import Flask, request, jsonify
from functools import wraps
import time
import os

app = Flask(__name__)

# 配置
SECRET_TOKEN = "your-secret-token-here"
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def require_token(f):
    """Token验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '')
        if not token.startswith('Bearer ') or token[7:] != SECRET_TOKEN:
            return jsonify({
                "success": False,
                "code": 401,
                "message": "认证失败",
                "timestamp": int(time.time())
            }), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/heartbeat', methods=['GET'])
@require_token
def heartbeat():
    """心跳接口"""
    return jsonify({
        "success": True,
        "code": 200,
        "message": "OK",
        "data": {
            "server_time": int(time.time()),
            "config_version": "1.0.0",
            "update_available": False
        },
        "timestamp": int(time.time())
    })

@app.route('/api/event', methods=['POST'])
@require_token
def handle_event():
    """事件处理接口"""
    data = request.json
    event_type = data.get('event_type')
    device_id = data.get('device_id')
    
    # 记录事件到数据库或日志
    print(f"收到事件: {event_type} from {device_id}")
    print(f"事件数据: {data}")
    
    # 根据不同事件类型处理
    if event_type == 'access_granted':
        # 处理授权访问
        pass
    elif event_type == 'access_denied':
        # 处理拒绝访问，可能需要告警
        pass
    elif event_type == 'emergency_open':
        # 紧急开门，发送告警
        pass
    
    return jsonify({
        "success": True,
        "code": 200,
        "message": "事件已接收",
        "data": {
            "event_id": f"evt_{int(time.time())}_{device_id}",
            "received_at": int(time.time())
        },
        "timestamp": int(time.time())
    })

@app.route('/api/upload/<upload_type>', methods=['POST'])
@require_token
def upload_image(upload_type):
    """图片上传接口"""
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "code": 400,
            "message": "没有文件",
            "timestamp": int(time.time())
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "success": False,
            "code": 400,
            "message": "文件名为空",
            "timestamp": int(time.time())
        }), 400
    
    # 保存文件
    filename = f"{upload_type}_{int(time.time())}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # 解析元数据（如果有）
    metadata = request.form.get('metadata')
    if metadata:
        import json
        metadata = json.loads(metadata)
        print(f"图片元数据: {metadata}")
    
    return jsonify({
        "success": True,
        "code": 200,
        "message": "图片上传成功",
        "data": {
            "file_id": f"img_{int(time.time())}",
            "file_path": filepath,
            "file_size": os.path.getsize(filepath),
            "url": f"http://server.com/images/{filename}"
        },
        "timestamp": int(time.time())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
```

---

## 6. 安全建议

1. **使用 HTTPS**: 生产环境应使用 HTTPS 协议
2. **Token 管理**: 
   - 使用强随机Token或JWT
   - 定期更换Token
   - 不同设备使用不同Token
3. **访问控制**: 实现IP白名单
4. **日志审计**: 记录所有访问日志
5. **数据加密**: 敏感数据应加密传输
6. **限流**: 实现请求频率限制防止DoS攻击

---

## 7. 测试工具

可以使用 curl 命令测试API：

```bash
# 心跳测试
curl -H "Authorization: Bearer your-secret-token-here" \
     http://localhost:8080/api/heartbeat

# 事件上报测试
curl -X POST \
     -H "Authorization: Bearer your-secret-token-here" \
     -H "Content-Type: application/json" \
     -d '{"device_id":"K230_001","event_type":"device_online","timestamp":1699999999,"data":{"device_name":"Test"}}' \
     http://localhost:8080/api/event

# 图片上传测试
curl -X POST \
     -H "Authorization: Bearer your-secret-token-here" \
     -F "file=@test.jpg" \
     -F 'metadata={"person":"测试","confidence":0.95}' \
     http://localhost:8080/api/upload/granted
```

这个文档提供了完整的HTTP API规范，包括认证机制、所有接口的详细说明、请求响应格式、错误码定义，以及一个可以直接运行的Python服务器示例代码。