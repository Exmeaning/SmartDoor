# K230 智能门禁系统部署指南

## 1. 首次部署步骤

### 1.1 配置文件准备
1. 复制 `config.template.json` 为 `config.json`
2. 修改以下必要配置：
   - `device_id`: 设置唯一的设备ID（如：K230_001）
   - `device_name`: 设置设备名称（如：前门、后门等）
   - `device_secret`: 设置设备密钥（用于签名验证）
   - `wifi_ssid`: WiFi名称
   - `wifi_password`: WiFi密码
   - `http_server_url`: 服务器地址
   - `http_token`: API访问令牌

### 1.2 Token生成建议

#### 方法1：使用UUID（推荐）
```python
import uuid
token = str(uuid.uuid4())
print(f"Generated Token: {token}")
```

#### 方法2：使用随机字符串
```python
import secrets
token = secrets.token_urlsafe(32)
print(f"Generated Token: {token}")
```

#### 方法3：使用设备特定Token
```python
import hashlib
device_id = "K230_001"
secret = "your-secret-key"
token = hashlib.sha256(f"{device_id}:{secret}".encode()).hexdigest()
print(f"Device Token: {token}")
```

### 1.3 安全配置检查清单

- [ ] 修改默认的 `device_secret`
- [ ] 设置强密码的 `http_token`
- [ ] 验证 `http_server_url` 使用HTTPS（生产环境）
- [ ] 确认 `device_id` 全局唯一
- [ ] 删除或保护 `config.template.json`

## 2. 网络安全配置

### 2.1 防火墙规则（服务器端）
```bash
# 只允许特定IP范围访问
iptables -A INPUT -p tcp --dport 8080 -s 192.168.1.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -j DROP
```

### 2.2 Nginx反向代理配置（推荐）
```nginx
server {
    listen 443 ssl;
    server_name door.example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location /api/ {
        proxy_pass http://localhost:8080/api/;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Device-ID $http_x_device_id;
        
        # Token验证
        if ($http_authorization !~ "Bearer .+") {
            return 401;
        }
    }
}
```

## 3. 设备注册流程

### 3.1 服务器端设备注册
```sql
-- 设备注册表
CREATE TABLE devices (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    token VARCHAR(255) UNIQUE,
    secret VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP,
    status ENUM('active', 'inactive', 'banned') DEFAULT 'active'
);

-- 插入新设备
INSERT INTO devices (id, name, token, secret) 
VALUES ('K230_001', '前门', 'your-token-here', 'device-secret');
```

### 3.2 批量部署脚本
```python
#!/usr/bin/env python3
"""批量生成设备配置"""

import json
import uuid
import hashlib

def generate_device_config(device_id, device_name, server_url, base_config):
    """生成单个设备配置"""
    config = json.loads(base_config)
    
    # 生成唯一的token和secret
    token = str(uuid.uuid4())
    secret = hashlib.sha256(f"{device_id}:{uuid.uuid4()}".encode()).hexdigest()[:32]
    
    # 更新配置
    config['system']['device_id'] = device_id
    config['system']['device_name'] = device_name
    config['system']['device_secret'] = secret
    config['network']['http_server_url'] = server_url
    config['network']['http_token'] = token
    
    return config, token, secret

# 使用示例
with open('config.template.json', 'r') as f:
    template = f.read()

devices = [
    ('K230_001', '前门'),
    ('K230_002', '后门'),
    ('K230_003', '侧门'),
]

server_url = "https://door.example.com"
device_registry = []

for device_id, device_name in devices:
    config, token, secret = generate_device_config(
        device_id, device_name, server_url, template
    )
    
    # 保存配置文件
    with open(f'config_{device_id}.json', 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    # 记录设备信息
    device_registry.append({
        'id': device_id,
        'name': device_name,
        'token': token,
        'secret': secret
    })

# 保存设备注册表
with open('device_registry.json', 'w') as f:
    json.dump(device_registry, f, indent=4)

print(f"已生成 {len(devices)} 个设备配置")
```

## 4. 监控和维护

### 4.1 日志监控
```bash
# 查看设备日志
tail -f /sdcard/logs/door_access.log

# 查看错误日志
grep ERROR /sdcard/logs/door_access.log
```

### 4.2 健康检查脚本
```python
import requests
import time

def check_device_health(device_id, token, server_url):
    """检查设备健康状态"""
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        response = requests.get(f"{server_url}/api/device/{device_id}/status", 
                               headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            last_seen = data.get('last_seen', 0)
            if time.time() - last_seen > 300:  # 5分钟未心跳
                return 'WARNING', '设备离线超过5分钟'
            return 'OK', '设备正常'
        else:
            return 'ERROR', f'状态码: {response.status_code}'
    except Exception as e:
        return 'ERROR', str(e)
```

## 5. 故障排除

### 常见问题

1. **认证失败 (401)**
   - 检查 `http_token` 是否正确
   - 确认服务器端token配置

2. **网络连接失败**
   - 验证WiFi配置
   - 检查服务器地址和端口
   - 确认防火墙规则

3. **设备离线**
   - 检查网络连接
   - 查看设备日志
   - 验证心跳功能

### 调试模式
在 `config.json` 中启用调试模式：
```json
{
    "system": {
        "debug_mode": true,
        "log_level": "DEBUG"
    }
}
```

## 6. 安全最佳实践

1. **定期更换Token**：建议每3-6个月更换一次
2. **使用HTTPS**：生产环境必须使用SSL/TLS
3. **IP白名单**：限制服务器访问IP范围
4. **日志审计**：定期审查访问日志
5. **备份配置**：加密保存配置备份
6. **最小权限原则**：每个设备使用独立的token

## 7. 更新和升级

### 7.1 配置更新（不中断服务）
1. 修改 `config.json`
2. 重启应用：`supervisorctl restart k230_door`

### 7.2 固件更新
1. 备份当前配置
2. 上传新固件
3. 恢复配置
4. 验证功能

## 联系支持

- 技术支持邮箱：support@example.com
- 紧急电话：xxx-xxxx-xxxx
- 文档地址：https://docs.example.com