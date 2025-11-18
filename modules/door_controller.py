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
        self.failed_attempts = 0
        
        # 访问记录
        self.access_history = []
        self.max_history_size = 100
        
        # 图片缓存（用于上传）
        self.last_captured_image = None
        self.last_captured_path = None
        
        # 门配置
        self.door_id = self.config.get('door.id', 'main')
        
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
            current_time = utime.time()
            
            # 记录日志
            self.logger.log_door_event("ACCESS_GRANTED", person_name, method, 
                                     {"confidence": confidence})
            
            # 播放欢迎音频
            self.audio.play_feedback('access_granted')
            
            # 开门
            success = self.motor.open_door()
            
            if success:
                self.is_locked = False
                self.last_access_time = current_time
                self.access_count += 1
                
                # 添加到访问历史
                self._add_to_history({
                    "time": self.last_access_time,
                    "person": person_name,
                    "method": method,
                    "confidence": confidence,
                    "result": "granted"
                })
                
                # 准备完整的事件数据
                event_data = {
                    "person": person_name,
                    "method": method,
                    "confidence": confidence,
                    "door_id": self.door_id,
                    "time": current_time,
                    "access_count": self.access_count
                }
                
                # 发送事件和图像到服务器
                self._send_access_event("access_granted", event_data, person_name, True)
                
                # 清空缓存的图像
                self.clear_captured_image()
                
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
            self.failed_attempts += 1
            
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
            
            # 准备完整的事件数据
            event_data = {
                "person": person_name,
                "reason": reason,
                "door_id": self.door_id,
                "time": current_time,
                "failed_attempts": self.failed_attempts
            }
            
            # 发送事件和图像到服务器
            self._send_access_event("access_denied", event_data, person_name, False)
            
            # 清空缓存的图像
            self.clear_captured_image()
            
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
    
    def set_captured_image(self, image_data, image_path=None):
        """设置捕获的图像数据（供人脸识别模块调用）
        
        Args:
            image_data: 图像数据（bytes格式）
            image_path: 图像文件路径（可选）
        """
        self.last_captured_image = image_data
        self.last_captured_path = image_path
    
    def clear_captured_image(self):
        """清空缓存的图像数据"""
        self.last_captured_image = None
        self.last_captured_path = None
    
    def _send_access_event(self, event_type, event_data, person_name, is_granted):
        """发送访问事件到服务器（包括JSON和图像）
        
        Args:
            event_type: 事件类型（access_granted/access_denied）
            event_data: 事件数据
            person_name: 人员名称
            is_granted: 是否授权通过
        """
        try:
            # 检查网络是否可用
            if not self.network_mgr or not self.network_mgr.is_connected:
                self.logger.debug("网络不可用，事件已缓存")
                # TODO: 实现离线缓存机制
                return
            
            # 检查HTTP客户端是否可用
            if not hasattr(self.network_mgr, 'http_client') or not self.network_mgr.http_client:
                self.logger.debug("HTTP客户端不可用")
                return
            
            image_url = None
            
            # 上传图像（如果有）
            if self.last_captured_image:
                try:
                    # 生成文件名
                    status = "granted" if is_granted else "denied"
                    safe_name = person_name.replace(" ", "_").replace("/", "_")
                    timestamp = int(event_data['time'])
                    filename = f"{status}_{safe_name}_{timestamp}.jpg"
                    
                    # 准备multipart上传数据
                    files = {
                        'file': (filename, self.last_captured_image)
                    }
                    
                    # 准备元数据
                    metadata = {
                        'person': person_name,
                        'timestamp': timestamp,
                        'device_id': self.config.get('system.device_id', 'unknown'),
                        'confidence': event_data.get('confidence', 0)
                    }
                    
                    if not is_granted:
                        metadata['reason'] = event_data.get('reason', 'unknown')
                    
                    fields = {
                        'metadata': metadata
                    }
                    
                    # 选择上传端点
                    upload_endpoint = '/api/upload/granted' if is_granted else '/api/upload/denied'
                    
                    # 上传图像
                    response = self.network_mgr.http_client.upload_multipart(
                        upload_endpoint,
                        files,
                        fields
                    )
                    
                    if response and response.get('success'):
                        image_url = response.get('data', {}).get('url')
                        self.logger.info(f"图像已上传: {filename}")
                    else:
                        self.logger.warning(f"图像上传失败: {response}")
                        
                except Exception as e:
                    self.logger.error(f"上传图像异常: {e}")
            
            # 添加图像URL到事件数据
            if image_url:
                event_data['face_image_url'] = image_url
            
            # 如果有本地图像路径，也添加
            if self.last_captured_path:
                event_data['local_image_path'] = self.last_captured_path
            
            # 发送事件JSON到服务器
            self.network_mgr.send_event(event_type, event_data)
            
            self.logger.info(f"事件已发送到服务器: {event_type}")
            
        except Exception as e:
            self.logger.error(f"发送访问事件失败: {e}")
    
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