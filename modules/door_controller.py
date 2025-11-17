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
                self._send_to_cloud({
                    "event": "access_granted",
                    "person": person_name,
                    "time": self.last_access_time,
                    "device_id": self.config.get('system.device_id')
                })
                
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
            # 记录日志
            self.logger.log_door_event("ACCESS_DENIED", person_name, "face", 
                                     {"reason": reason})
            
            # 播放拒绝音频
            self.audio.play_feedback('access_denied')
            
            # 添加到访问历史
            self._add_to_history({
                "time": utime.time(),
                "person": person_name,
                "reason": reason,
                "result": "denied"
            })
            
            # 发送警告到云端
            self._send_to_cloud({
                "event": "access_denied",
                "person": person_name,
                "reason": reason,
                "time": utime.time(),
                "device_id": self.config.get('system.device_id')
            })
            
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
            # TODO: 实现云端数据发送
            # 这里先记录日志
            self.logger.debug(f"发送到云端: {data}")
            
        except Exception as e:
            self.logger.error(f"发送云端数据失败: {e}")
    
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