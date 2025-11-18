import utime
import gc
from utils.logger import get_logger
from utils.config_loader import ConfigLoader
from core.motor_control import MotorController
from core.audio_manager import AudioManager

class DoorController:
    """门禁控制器 / Door Access Controller
    
    整合各个模块，实现门禁控制逻辑
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = get_logger()
        
        # 初始化各个模块
        self.motor = MotorController()
        self.audio = AudioManager()
        
        # 门禁状态
        self.is_locked = True
        self.last_access_time = 0
        self.access_count = 0
        
        # 访问记录
        self.access_history = []
        self.max_history_size = 100
        
        # 图片缓存（用于上传）
        self.last_captured_image = None
        
        # 获取网络管理器实例（如果可用）
        self.network_mgr = None
        self._init_network()
    
    def _init_network(self):
        """初始化网络管理器引用"""
        try:
            # 获取全局网络管理器实例（不创建新的）
            import sys
            if hasattr(sys.modules.get('__main__', None), 'network_mgr'):
                self.network_mgr = sys.modules['__main__'].network_mgr
                self.logger.debug("网络管理器已连接")
        except Exception as e:
            self.logger.debug(f"网络管理器不可用: {e}")
        
    def grant_access(self, person_name, method="face", confidence=0.0):
        """授权访问 / Grant access
        
        Args:
            person_name: 访问者姓名
            method: 访问方式（face/card/password等）
            confidence: 置信度
        """
        try:
            # 记录日志
            self.logger.log_door_event("ACCESS_GRANTED", person_name, method, 
                                     {"confidence": confidence})
            
            # 播放欢迎音频
            self.audio.play_feedback('access_granted')
            
            # 开门
            success = self.motor.open_door()
            
            if success:
                self.is_locked = False
                self.last_access_time = utime.time()
                self.access_count += 1
                
                # 添加到访问历史
                self._add_to_history({
                    "time": self.last_access_time,
                    "person": person_name,
                    "method": method,
                    "confidence": confidence,
                    "result": "granted"
                })
                
                # 发送到云端
                cloud_data = {
                    "event": "access_granted",
                    "person": person_name,
                    "method": method,
                    "confidence": confidence,
                    "time": self.last_access_time,
                    "door_id": self.config.get('door.id', 'main'),
                    "device_id": self.config.get('system.device_id')
                }
                self._send_to_cloud(cloud_data)
                
                # 如果有捕获的图像，上传到服务器
                if self.last_captured_image and self.network_mgr and self.network_mgr.is_connected:
                    try:
                        filename = f"granted_{person_name}_{int(self.last_access_time)}.jpg"
                        self.network_mgr.upload_image('/api/upload/granted', 
                                                     self.last_captured_image, 
                                                     filename)
                        self.logger.debug(f"已上传访问图像: {filename}")
                    except Exception as e:
                        self.logger.debug(f"上传图像失败: {e}")
                
                return True
            else:
                self.logger.error("开门失败")
                return False
                
        except Exception as e:
            self.logger.error(f"授权访问失败: {e}")
            return False
    
    def deny_access(self, reason="unknown", person_name="unknown"):
        """拒绝访问 / Deny access
        
        Args:
            reason: 拒绝原因
            person_name: 尝试访问者
        """
        try:
            current_time = utime.time()
            
            # 记录日志
            self.logger.log_door_event("ACCESS_DENIED", person_name, "face", 
                                     {"reason": reason})
            
            # 播放拒绝音频
            self.audio.play_feedback('access_denied')
            
            # 添加到访问历史
            self._add_to_history({
                "time": current_time,
                "person": person_name,
                "reason": reason,
                "result": "denied"
            })
            
            # 发送警告到云端
            cloud_data = {
                "event": "access_denied",
                "person": person_name,
                "reason": reason,
                "time": current_time,
                "door_id": self.config.get('door.id', 'main'),
                "device_id": self.config.get('system.device_id')
            }
            self._send_to_cloud(cloud_data)
            
            # 如果有捕获的图像，上传到服务器作为安全记录
            if self.last_captured_image and self.network_mgr and self.network_mgr.is_connected:
                try:
                    filename = f"denied_{person_name}_{int(current_time)}.jpg"
                    self.network_mgr.upload_image('/api/upload/denied', 
                                                 self.last_captured_image, 
                                                 filename)
                    self.logger.debug(f"已上传拒绝访问图像: {filename}")
                except Exception as e:
                    self.logger.debug(f"上传图像失败: {e}")
            
        except Exception as e:
            self.logger.error(f"记录拒绝访问失败: {e}")
    
    def emergency_open(self):
        """紧急开门 / Emergency open"""
        try:
            self.logger.warning("执行紧急开门")
            
            # 直接开门
            self.motor.enable()
            success = self.motor.open_door()
            
            # 播放警告音
            self.audio.play_beep(5, 2000, 100)
            
            # 记录事件
            self.logger.log_door_event("EMERGENCY_OPEN", "SYSTEM", "emergency")
            
            # 发送紧急事件到云端
            self._send_to_cloud({
                "event": "emergency_open",
                "time": utime.time(),
                "door_id": self.config.get('door.id', 'main'),
                "device_id": self.config.get('system.device_id')
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"紧急开门失败: {e}")
            return False
    
    def lock_door(self):
        """锁门 / Lock door"""
        try:
            if self.is_locked:
                self.logger.info("门已经是锁定状态")
                return True
            
            success = self.motor.close_door()
            
            if success:
                self.is_locked = True
                self.logger.info("门已锁定")
                
                # 记录事件
                self.logger.log_door_event("DOOR_LOCKED", "SYSTEM", "auto")
                
                # 发送关门事件到云端
                self._send_to_cloud({
                    "event": "door_locked",
                    "time": utime.time(),
                    "door_id": self.config.get('door.id', 'main'),
                    "device_id": self.config.get('system.device_id')
                })
                
            return success
            
        except Exception as e:
            self.logger.error(f"锁门失败: {e}")
            return False
    
    def _add_to_history(self, record):
        """添加访问记录 / Add access record"""
        try:
            self.access_history.append(record)
            
            # 限制历史记录大小
            if len(self.access_history) > self.max_history_size:
                self.access_history = self.access_history[-self.max_history_size:]
            
            # 定期垃圾回收
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"添加访问记录失败: {e}")
    
    def _send_to_cloud(self, data):
        """发送数据到云端 / Send data to cloud"""
        try:
            # 如果网络管理器可用且已连接，通过HTTP发送
            if self.network_mgr and self.network_mgr.is_connected:
                # 使用send_event方法发送事件
                event_type = data.get('event', 'unknown')
                self.network_mgr.send_event(event_type, data)
                self.logger.debug(f"已发送到云端: {event_type}")
            else:
                # 网络不可用，仅记录日志
                self.logger.debug(f"离线模式，数据已缓存: {data}")
                
                # TODO: 可以将数据存储到本地，等网络恢复后批量上传
                
        except Exception as e:
            self.logger.error(f"发送云端数据失败: {e}")
    
    def set_captured_image(self, image_data):
        """设置捕获的图像数据（供人脸识别模块调用）
        
        Args:
            image_data: 图像数据（bytes格式）
        """
        self.last_captured_image = image_data
    
    def get_statistics(self):
        """获取统计信息 / Get statistics"""
        return {
            "is_locked": self.is_locked,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "history_size": len(self.access_history)
        }
    
    def get_recent_history(self, count=10):
        """获取最近的访问记录 / Get recent access history"""
        return self.access_history[-count:] if self.access_history else []
    
    def test_system(self):
        """测试系统 / Test system"""
        self.logger.info("开始系统测试...")
        
        try:
            # 测试音频
            self.logger.info("测试音频系统...")
            self.audio.test_audio()
            utime.sleep(1)
            
            # 测试电机
            self.logger.info("测试电机系统...")
            self.motor.test_motor()
            utime.sleep(1)
            
            # 测试开关门流程
            self.logger.info("测试开关门流程...")
            self.grant_access("TEST_USER", "test", 1.0)
            utime.sleep(3)
            self.lock_door()
            
            self.logger.info("系统测试完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统测试失败: {e}")
            return False
    
    def deinit(self):
        """释放资源 / Release resources"""
        try:
            self.motor.deinit()
            self.audio.deinit()
            self.logger.info("门禁控制器已释放")
        except Exception as e:
            self.logger.error(f"释放门禁控制器失败: {e}")