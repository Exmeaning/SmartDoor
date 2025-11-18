#!/usr/bin/env micropython
"""
HTTP客户端模块 / HTTP Client Module
用于处理与服务器的HTTP通信
Author: System
Version: 1.0.0
"""

import socket
import json
import utime
from utils.logger import get_logger

class HTTPClient:
    """HTTP客户端类"""
    
    def __init__(self, server_url, token, timeout=10):
        """
        初始化HTTP客户端
        
        Args:
            server_url: 服务器URL
            token: 认证令牌
            timeout: 超时时间（秒）
        """
        self.logger = get_logger()
        self.server_url = server_url
        self.token = token
        self.timeout = timeout
        self.last_poll_time = 0
        self.poll_interval = 1  # 轮询间隔（秒）
        
    def parse_url(self, url):
        """解析URL"""
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
    
    def request(self, method, endpoint, data=None, headers=None):
        """发送HTTP请求"""
        try:
            # 构建完整URL
            url = self.server_url + endpoint
            url_parts = self.parse_url(url)
            
            # DNS解析
            addr_info = socket.getaddrinfo(url_parts['host'], url_parts['port'])
            addr = addr_info[0][-1]
            
            # 创建socket
            s = socket.socket()
            s.settimeout(self.timeout)
            s.connect(addr)
            
            # 构建请求
            request = f"{method} {url_parts['path']} HTTP/1.0\r\n"
            request += f"Host: {url_parts['host']}\r\n"
            
            # 添加Token认证
            if self.token:
                request += f"Authorization: Bearer {self.token}\r\n"
            
            # 添加自定义headers
            if headers:
                for key, value in headers.items():
                    if key.lower() != 'authorization':  # 避免覆盖token
                        request += f"{key}: {value}\r\n"
            
            # 处理数据
            if data:
                if isinstance(data, dict):
                    data = json.dumps(data)
                    request += "Content-Type: application/json\r\n"
                
                if isinstance(data, str):
                    data = data.encode()
                    
                request += f"Content-Length: {len(data)}\r\n"
            
            request += "\r\n"
            
            # 发送请求
            if isinstance(request, str):
                request = request.encode()
            s.send(request)
            
            if data:
                s.send(data)
            
            # 读取响应
            response = self.read_response(s)
            s.close()
            
            return response
            
        except Exception as e:
            self.logger.error(f"HTTP请求失败: {e}")
            return None
    
    def read_response(self, sock):
        """读取HTTP响应"""
        try:
            # 读取响应头
            response_data = b""
            while b"\r\n\r\n" not in response_data:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                response_data += chunk
            
            # 解析响应
            header_end = response_data.find(b"\r\n\r\n")
            if header_end == -1:
                return None
                
            headers_raw = response_data[:header_end].decode()
            body_data = response_data[header_end + 4:]
            
            # 解析状态码
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
            
            # 读取完整body
            content_length = int(headers.get('Content-Length', 0))
            while len(body_data) < content_length:
                chunk = sock.recv(min(4096, content_length - len(body_data)))
                if not chunk:
                    break
                body_data += chunk
            
            # 解析响应body
            body = body_data
            try:
                content_type = headers.get('Content-Type', '')
                if content_type.startswith('application/json'):
                    # JSON响应
                    body = json.loads(body_data.decode())
                elif content_type.startswith('application/octet-stream'):
                    # 二进制响应，保持原样
                    body = body_data
                else:
                    # 其他格式，尝试解码为文本
                    try:
                        body = body_data.decode() if body_data else ""
                    except:
                        # 解码失败，保持二进制
                        body = body_data
            except:
                body = body_data
            
            return {
                'success': status_code == 200,
                'code': status_code,
                'headers': headers,
                'data': body
            }
            
        except Exception as e:
            self.logger.error(f"读取响应失败: {e}")
            return None
    
    def poll_server(self):
        """
        轮询服务器获取命令
        返回: 命令字典或None
        """
        current_time = utime.time()
        
        # 检查是否到达轮询时间
        if current_time - self.last_poll_time < self.poll_interval:
            return None
        
        self.last_poll_time = current_time
        
        try:
            # 发送轮询请求
            response = self.request('GET', '/api/device/poll')
            
            if response and response['success']:
                data = response.get('data', {})
                
                # 检查是否有待执行的命令
                if isinstance(data, dict) and data.get('command'):
                    self.logger.info(f"收到服务器命令: {data['command']}")
                    return data
                    
            return None
            
        except Exception as e:
            self.logger.debug(f"轮询失败: {e}")
            return None
    
    def hex_encode(self, data):
        """将字节数据转换为十六进制字符串（替代base64）"""
        if isinstance(data, bytes):
            return ''.join(['%02x' % b for b in data])
        return ''
    
    def hex_decode(self, hex_string):
        """将十六进制字符串转换为字节数据"""
        try:
            return bytes.fromhex(hex_string)
        except:
            return b''
    
    def upload_multipart(self, endpoint, files, fields=None, use_hex=False):
        """
        上传multipart/form-data数据
        
        Args:
            endpoint: API端点
            files: 文件字典 {'field_name': (filename, file_data)}
            fields: 其他字段字典
            use_hex: 是否使用十六进制编码二进制数据
        """
        try:
            boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
            body = b""
            
            # 添加普通字段
            if fields:
                for key, value in fields.items():
                    body += f"------{boundary}\r\n".encode()
                    body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
                    if isinstance(value, dict):
                        value = json.dumps(value)
                    body += f"{value}\r\n".encode()
            
            # 添加文件
            for field_name, (filename, file_data) in files.items():
                body += f"------{boundary}\r\n".encode()
                body += f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode()
                body += b"Content-Type: application/octet-stream\r\n\r\n"
                body += file_data
                body += b"\r\n"
            
            # 结束边界
            body += f"------{boundary}--\r\n".encode()
            
            # 设置headers
            headers = {
                'Content-Type': f'multipart/form-data; boundary={boundary}',
                'Content-Length': str(len(body))
            }
            
            # 发送请求
            return self.request('POST', endpoint, body, headers)
            
        except Exception as e:
            self.logger.error(f"Multipart上传失败: {e}")
            return None