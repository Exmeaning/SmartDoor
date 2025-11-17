import utime
import gc
from utils.logger import get_logger

class SleepManager:
    """休眠管理器 / Sleep Manager
    
    管理系统休眠策略，在无人时降低系统功耗
    """
    
    def __init__(self, check_interval_ms=1000):
        self.logger = get_logger()
        self.check_interval_ms = check_interval_ms
        self.is_sleeping = False
        self.last_activity_time = utime.ticks_ms()
        self.sleep_timeout_ms = 30000  # 30秒无活动进入休眠
        self.face_detector = None
        self.pipeline = None
        
    def set_face_detector(self, detector):
        """设置人脸检测器 / Set face detector"""
        self.face_detector = detector
        
    def set_pipeline(self, pipeline):
        """设置视频管道 / Set video pipeline"""
        self.pipeline = pipeline
    
    def update_activity(self):
        """更新活动时间 / Update activity time"""
        self.last_activity_time = utime.ticks_ms()
        if self.is_sleeping:
            self.wake_up()
    
    def check_sleep_condition(self):
        """检查是否需要进入休眠 / Check if need to sleep"""
        if not self.is_sleeping:
            elapsed = utime.ticks_diff(utime.ticks_ms(), self.last_activity_time)
            if elapsed > self.sleep_timeout_ms:
                self.enter_sleep()
                return True
        return False
    
    def enter_sleep(self):
        """进入休眠模式 / Enter sleep mode"""
        if self.is_sleeping:
            return
        
        self.logger.info("系统进入休眠模式")
        self.is_sleeping = True
        
        # 降低检测频率
        # 这里可以关闭一些不必要的模块
        gc.collect()
    
    def wake_up(self):
        """唤醒系统 / Wake up system"""
        if not self.is_sleeping:
            return
        
        self.logger.info("系统从休眠模式唤醒")
        self.is_sleeping = False
        
    def sleep_check_loop(self):
        """休眠检测循环 / Sleep check loop
        
        在休眠模式下，降低摄像头调用频率
        每秒检测一次是否有人
        """
        while self.is_sleeping:
            try:
                # 快速获取一帧
                if self.pipeline and self.face_detector:
                    img = self.pipeline.get_frame()
                    
                    # 快速人脸检测
                    faces = self.face_detector.run(img)
                    
                    if faces and len(faces) > 0:
                        # 检测到人脸，唤醒系统
                        self.update_activity()
                        self.logger.info("检测到人脸，唤醒系统")
                        break
                
                # 休眠间隔
                utime.sleep_ms(self.check_interval_ms)
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"休眠检测错误: {e}")
                utime.sleep_ms(self.check_interval_ms)
    
    def get_sleep_status(self):
        """获取休眠状态 / Get sleep status"""
        return {
            "is_sleeping": self.is_sleeping,
            "last_activity": self.last_activity_time,
            "timeout_ms": self.sleep_timeout_ms
        }
    
    def set_sleep_timeout(self, timeout_ms):
        """设置休眠超时时间 / Set sleep timeout"""
        self.sleep_timeout_ms = timeout_ms
        self.logger.info(f"休眠超时设置为: {timeout_ms}ms")