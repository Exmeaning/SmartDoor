import network
import utime
import socket
import json
import uhashlib as hashlib
from utils.logger import get_logger
from utils.config_loader import ConfigLoader
import binascii

class NetworkManager:
    """网络管理器 / Network Manager"""
    
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = get_logger()
        self.sta = None
        self.is_connected = False
        self.http_server_url = None
        self.http_token = None
        self.http_timeout = 10
        self.http_client = None
        
    def connect_wifi(self, max_retry=3):
        """连接WiFi网络 / Connect to WiFi network"""
        ssid = self.config.get('network.wifi_ssid')
        password = self.config.get('network.wifi_password')
    
        if not ssid:
            self.logger.error("WiFi SSID未配置")
            return False
    
        self.logger.info(f"正在连接WiFi: {ssid}")
    
        try:
            # 创建STA接口
            self.sta = network.WLAN(network.STA_IF)
            self.sta.active(True)
        
            # 尝试连接
            for attempt in range(max_retry):
                self.sta.connect(ssid, password)
            
                # 等待连接
                timeout = 10
                start_time = utime.time()
            
                while not self.sta.isconnected():
                    if utime.time() - start_time > timeout:
                        self.logger.warning(f"WiFi连接超时 (尝试 {attempt + 1}/{max_retry})")
                        break
                    utime.sleep(1)
            
                if self.sta.isconnected():
                    self.is_connected = True
                    ip_info = self.sta.ifconfig()
                    self.logger.info(f"WiFi连接成功! IP: {ip_info[0]}")
                
                    # 初始化HTTP配置
                    self.http_server_url = self.config.get('network.http_server_url')
                    self.http_token = self.config.get('network.http_token', '')
                    self.http_timeout = self.config.get('network.http_timeout', 10)
                
                    # HTTP时间同步
                    if self.config.get('network.ntp_sync', True):
                        self.sync_time()
                
                    return True
        
            self.logger.error("WiFi连接失败，进入离线模式")
            return False
        
        except Exception as e:
            self.logger.error(f"WiFi连接异常: {e}")
            return False
    
    def disconnect_wifi(self):
        """断开WiFi连接 / Disconnect WiFi"""
        if self.sta:
            try:
                self.sta.disconnect()
                self.sta.active(False)
                self.is_connected = False
                self.logger.info("WiFi已断开")
            except Exception as e:
                self.logger.error(f"断开WiFi失败: {e}")
    
    # core/network_manager.py

    def sync_time(self):
        """从HTTP服务器同步时间戳"""
        try:
            # 检查是否启用时间同步
            if not self.config.get('network.ntp_sync', True):
                self.logger.info("时间同步已禁用")
                return False
        
            # 获取时间同步服务器URL
            time_server_url = self.config.get('network.time_server_url', 
                                              self.config.get('network.http_server_url'))
        
            if not time_server_url:
                self.logger.warning("时间服务器URL未配置")
                return False
        
            # 从HTTP服务器获取时间戳
            self.logger.info("正在从HTTP服务器同步时间...")
        
            # 直接使用socket请求，避免循环依赖
            try:
                # 解析URL
                url_parts = self._parse_url(time_server_url + '/api/time')
            
                # DNS解析
                addr_info = socket.getaddrinfo(url_parts['host'], url_parts['port'])
                addr = addr_info[0][-1]
            
                # 创建socket连接
                s = socket.socket()
                s.settimeout(5)  # 时间同步使用较短超时
                s.connect(addr)
            
                # 发送简单的GET请求
                request = f"GET {url_parts['path']} HTTP/1.0\r\n"
                request += f"Host: {url_parts['host']}\r\n"
                if self.http_token:
                    request += f"Authorization: Bearer {self.http_token}\r\n"
                request += "\r\n"
            
                s.send(request.encode())
            
                # 读取响应
                response = b""
                while True:
                    chunk = s.recv(1024)
                    if not chunk:
                        break
                    response += chunk
                    if b"\r\n\r\n" in response:
                        # 找到header结束，继续读取body
                        header_end = response.find(b"\r\n\r\n")
                        headers_raw = response[:header_end].decode()
                    
                        # 获取Content-Length
                        content_length = 0
                        for line in headers_raw.split('\r\n'):
                            if line.startswith('Content-Length:'):
                                content_length = int(line.split(':')[1].strip())
                                break
                    
                        # 读取完整的body
                        body_start = header_end + 4
                        if content_length > 0:
                            while len(response) - body_start < content_length:
                                chunk = s.recv(1024)
                                if not chunk:
                                    break
                                response += chunk
                        break
            
                s.close()
            
                # 解析响应
                if b"200 OK" in response:
                    # 提取body
                    body_start = response.find(b"\r\n\r\n") + 4
                    body = response[body_start:].decode()
                
                    # 解析JSON响应
                    try:
                        time_data = json.loads(body)
                        timestamp = time_data.get('timestamp')
                    except:
                        # 如果不是JSON，尝试直接解析为数字
                        try:
                            timestamp = int(body.strip())
                        except:
                            self.logger.error(f"无法解析时间响应: {body}")
                            return False
                
                    if timestamp:
                        # 设置系统时间
                        from machine import RTC
                        rtc = RTC()
                    
                        # 将时间戳转换为时间元组 (年,月,日,时,分,秒,星期,年日)
                        time_tuple = utime.localtime(timestamp)
                    
                        # 设置RTC时间
                        # RTC.datetime() 格式: (year, month, day, weekday, hours, minutes, seconds, subseconds)
                        rtc.datetime((time_tuple[0], time_tuple[1], time_tuple[2], 
                                     time_tuple[6], time_tuple[3], time_tuple[4], 
                                     time_tuple[5], 0))
                    
                        self.logger.info(f"✅ HTTP时间同步成功: {time_tuple[0]}-{time_tuple[1]:02d}-{time_tuple[2]:02d} {time_tuple[3]:02d}:{time_tuple[4]:02d}:{time_tuple[5]:02d}")
                    
                        # 验证时间是否设置成功
                        utime.sleep(1)
                        current = utime.localtime()
                        if current[0] > 2020:
                            self.logger.info(f"🎉 系统时间已更新: {current[:6]}")
                            return True
                        else:
                            self.logger.warning("⚠️ 时间设置可能失败")
                            return False
                    else:
                        self.logger.error("服务器响应中没有时间戳")
                        return False
                else:
                    self.logger.error(f"HTTP时间同步请求失败")
                    return False
                
            except Exception as e:
                self.logger.error(f"HTTP时间同步异常: {e}")
                return False
            
        except Exception as e:
            self.logger.error(f"时间同步异常: {e}")
            return False

    
    def check_connection(self):
        """检查网络连接状态 / Check network connection status"""
        if self.sta and self.sta.isconnected():
            return True
        return False
    
    def auto_reconnect(self):
        """自动重连WiFi / Auto reconnect WiFi"""
        if not self.check_connection():
            self.logger.info("检测到WiFi断开，尝试重连...")
            return self.connect_wifi()
        return True
    
    def init_http_client(self):
        """初始化HTTP客户端配置 / Initialize HTTP client configuration"""
        self.http_server_url = self.config.get('network.http_server_url')
        self.http_token = self.config.get('network.http_token', '')
        self.http_timeout = self.config.get('network.http_timeout', 10)
        
        if not self.http_server_url:
            self.logger.error("HTTP服务器URL未配置")
            return False
        
        if not self.http_token:
            self.logger.warning("HTTP Token未配置，可能导致认证失败")
        
        # 创建HTTP客户端实例
        try:
            from core.http_client import HTTPClient
            self.http_client = HTTPClient(self.http_server_url, self.http_token, self.http_timeout)
            self.logger.info(f"HTTP客户端配置完成: {self.http_server_url}")
            return True
        except Exception as e:
            self.logger.error(f"创建HTTP客户端失败: {e}")
            return False
    
    def http_request(self, method, endpoint, data=None, headers=None):
        """发送HTTP请求 / Send HTTP request"""
        if not self.is_connected:
            self.logger.error("网络未连接")
            return None
        
        try:
            # 解析URL
            url = self.http_server_url + endpoint
            url_parts = self._parse_url(url)
            
            # 获取主机地址
            addr_info = socket.getaddrinfo(url_parts['host'], url_parts['port'])
            addr = addr_info[0][-1]
            
            # 创建socket连接
            s = socket.socket()
            s.settimeout(self.http_timeout)
            s.connect(addr)
            
            # 构建请求
            request = self._build_http_request(method, url_parts['path'], 
                                              url_parts['host'], data, headers)
            
            # 发送请求
            s.send(request.encode())
            
            # 读取响应
            response = self._read_http_response(s)
            s.close()
            
            return response
            
        except Exception as e:
            self.logger.error(f"HTTP请求失败: {e}")
            return None
    
    def http_get(self, endpoint, headers=None):
        """发送GET请求 / Send GET request"""
        return self.http_request('GET', endpoint, headers=headers)
    
    def http_post(self, endpoint, data, headers=None):
        """发送POST请求 / Send POST request"""
        if headers is None:
            headers = {}
        
        if isinstance(data, dict):
            # JSON数据
            headers['Content-Type'] = 'application/json'
            data = json.dumps(data)
        elif isinstance(data, bytes):
            # 二进制数据（如图片）
            headers['Content-Type'] = 'application/octet-stream'
        else:
            # 文本数据
            headers['Content-Type'] = 'text/plain'
            
        return self.http_request('POST', endpoint, data, headers)
    
    def upload_image(self, endpoint, image_data, filename="image.jpg"):
        """上传图片 / Upload image"""
        try:
            # 使用multipart/form-data格式上传
            boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
            
            # 构建multipart body
            body = []
            body.append(f'------{boundary}')
            body.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"')
            body.append('Content-Type: image/jpeg')
            body.append('')
            
            # 合并文本部分
            text_part = '\r\n'.join(body).encode() + b'\r\n'
            
            # 添加图片数据
            image_part = image_data
            
            # 添加结束边界
            end_part = f'\r\n------{boundary}--\r\n'.encode()
            
            # 合并所有部分
            full_body = text_part + image_part + end_part
            
            # 设置headers
            headers = {
                'Content-Type': f'multipart/form-data; boundary={boundary}',
                'Content-Length': str(len(full_body))
            }
            
            # 发送请求
            response = self.http_request('POST', endpoint, full_body, headers)
            
            if response and response['status_code'] == 200:
                self.logger.info(f"图片上传成功: {filename}")
                return True
            else:
                self.logger.error(f"图片上传失败: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"图片上传异常: {e}")
            return False
    
    def poll_server(self):
        """轮询服务器获取命令"""
        if not self.http_client or not self.is_connected:
            return None
            
        return self.http_client.poll_server()
    
    def get_ip_address(self):
        """获取IP地址"""
        if self.sta and self.sta.isconnected():
            return self.sta.ifconfig()[0]
        return "0.0.0.0"
    
    def get_mac_address(self):
        """获取MAC地址"""
        if self.sta:
            import ubinascii
            mac = ubinascii.hexlify(self.sta.config('mac'), ':').decode()
            return mac.upper()
        return "00:00:00:00:00:00"
    
    def send_event(self, event_type, event_data):
        """发送事件到服务器 / Send event to server"""
        try:
            # 添加设备认证信息
            device_id = self.config.get('system.device_id', 'unknown')
            device_secret = self.config.get('system.device_secret', '')
            
            data = {
                'device_id': device_id,
                'event_type': event_type,
                'timestamp': utime.time(),
                'data': event_data
            }
            
            # 可选：添加设备签名（简单的HMAC）
            if device_secret:
                # 创建简单的签名

                signature_data = f"{device_id}:{event_type}:{utime.time()}:{device_secret}"
                h = hashlib.sha256(signature_data.encode())
                signature = binascii.hexlify(h.digest()).decode('utf-8')[:16]
                data['signature'] = signature
            
            response = self.http_post('/api/event', data)
            
            if response and response['status_code'] == 200:
                return True
            else:
                self.logger.warning(f"事件发送失败: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"发送事件异常: {e}")
            return False
    
    def _parse_url(self, url):
        """解析URL / Parse URL"""
        # 简单的URL解析
        if url.startswith('http://'):
            url = url[7:]
        elif url.startswith('https://'):
            url = url[8:]
        
        # 查找端口
        if ':' in url:
            parts = url.split(':', 1)
            host = parts[0]
            remaining = parts[1]
            if '/' in remaining:
                port_str, path = remaining.split('/', 1)
                port = int(port_str)
                path = '/' + path
            else:
                port = int(remaining)
                path = '/'
        else:
            if '/' in url:
                host, path = url.split('/', 1)
                path = '/' + path
            else:
                host = url
                path = '/'
            port = 80
        
        return {'host': host, 'port': port, 'path': path}
    
    def _build_http_request(self, method, path, host, data=None, headers=None):
        """构建HTTP请求 / Build HTTP request"""
        request = f"{method} {path} HTTP/1.0\r\n"
        request += f"Host: {host}\r\n"
        
        # 添加认证Token
        if self.http_token:
            request += f"Authorization: Bearer {self.http_token}\r\n"
        
        if headers:
            for key, value in headers.items():
                # 如果headers中已有Authorization，跳过（允许覆盖）
                if key.lower() == 'authorization' and self.http_token:
                    continue
                request += f"{key}: {value}\r\n"
        
        if data:
            if isinstance(data, str):
                data = data.encode()
            request += f"Content-Length: {len(data)}\r\n"
        
        request += "\r\n"
        
        if data:
            if isinstance(request, str):
                request = request.encode()
            if isinstance(data, str):
                data = data.encode()
            return request + data
        
        return request
    
    def _read_http_response(self, sock):
        """读取HTTP响应 / Read HTTP response"""
        try:
            # 读取响应头
            response_data = b""
            while b"\r\n\r\n" not in response_data:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                response_data += chunk
            
            # 解析响应头
            header_end = response_data.find(b"\r\n\r\n")
            headers_raw = response_data[:header_end].decode()
            body_start = response_data[header_end + 4:]
            
            # 解析状态行
            lines = headers_raw.split('\r\n')
            status_line = lines[0]
            status_parts = status_line.split(' ', 2)
            status_code = int(status_parts[1]) if len(status_parts) > 1 else 0
            
            # 解析headers
            headers = {}
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip()] = value.strip()
            
            # 读取body
            body = body_start
            content_length = int(headers.get('Content-Length', 0))
            
            if content_length > 0:
                while len(body) < content_length:
                    chunk = sock.recv(min(4096, content_length - len(body)))
                    if not chunk:
                        break
                    body += chunk
            
            # 尝试解析JSON响应
            try:
                body_text = body.decode()
                if headers.get('Content-Type', '').startswith('application/json'):
                    body = json.loads(body_text)
                else:
                    body = body_text
            except:
                pass  # 保持原始字节数据
            
            return {
                'status_code': status_code,
                'headers': headers,
                'body': body
            }
            
        except Exception as e:
            self.logger.error(f"读取HTTP响应失败: {e}")
            return None