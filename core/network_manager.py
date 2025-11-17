import network
import utime
import socket
from utils.logger import get_logger
from utils.config_loader import ConfigLoader

class NetworkManager:
    """网络管理器 / Network Manager"""
    
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = get_logger()
        self.sta = None
        self.is_connected = False
        self.tcp_socket = None
        
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
                    
                    # NTP时间同步
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
    
    def sync_time(self):
        """同步网络时间 / Sync network time"""
        try:
            import utime
            if utime.ntp_sync():
                self.logger.info("NTP时间同步成功")
                return True
            else:
                self.logger.warning("NTP时间同步失败")
                return False
        except Exception as e:
            self.logger.error(f"NTP时间同步异常: {e}")
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
    
    def create_tcp_client(self):
        """创建TCP客户端 / Create TCP client"""
        try:
            server_ip = self.config.get('network.tcp_server_ip')
            server_port = self.config.get('network.tcp_server_port', 60000)
            
            # 获取服务器地址信息
            addr_info = socket.getaddrinfo(server_ip, server_port)
            addr = addr_info[0][-1]
            
            # 创建socket
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect(addr)
            
            self.logger.info(f"TCP连接成功: {server_ip}:{server_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"TCP连接失败: {e}")
            return False
    
    def send_tcp_data(self, data):
        """发送TCP数据 / Send TCP data"""
        if not self.tcp_socket:
            self.logger.error("TCP未连接")
            return False
        
        try:
            if isinstance(data, str):
                data = data.encode()
            self.tcp_socket.send(data)
            return True
        except Exception as e:
            self.logger.error(f"TCP发送失败: {e}")
            return False
    
    def close_tcp_client(self):
        """关闭TCP连接 / Close TCP connection"""
        if self.tcp_socket:
            try:
                self.tcp_socket.close()
                self.tcp_socket = None
                self.logger.info("TCP连接已关闭")
            except Exception as e:
                self.logger.error(f"关闭TCP失败: {e}")