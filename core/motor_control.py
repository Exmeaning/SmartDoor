from machine import Pin, Timer
import utime
from utils.logger import get_logger
from utils.config_loader import ConfigLoader

class MotorController:
    """步进电机控制器 / Stepper Motor Controller"""
    
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = get_logger()
        
        # 读取引脚配置
        pul_pin = self.config.get('motor.pul_pin', 43)
        dir_pin = self.config.get('motor.dir_pin', 33)
        en_pin = self.config.get('motor.en_pin', 32)
        
        # 初始化引脚
        self.pul = Pin(pul_pin, Pin.OUT, value=0)
        self.dir = Pin(dir_pin, Pin.OUT, value=0)
        self.en = Pin(en_pin, Pin.OUT, value=1)  # 默认禁用
        
        # 电机参数
        self.frequency = self.config.get('motor.frequency', 1000)
        self.open_duration = self.config.get('motor.open_duration', 3000)
        self.close_duration = self.config.get('motor.close_duration', 3000)
        
        self.is_open = False
        self.is_running = False
        self.timer = None
        
    def enable(self):
        """使能电机 / Enable motor"""
        self.en.value(0)  # 低电平使能
        utime.sleep_ms(10)
        self.logger.debug("电机已使能")
    
    def disable(self):
        """禁用电机 / Disable motor"""
        self.en.value(1)  # 高电平禁用
        self.logger.debug("电机已禁用")
    
    def generate_pulse(self, steps, direction=1):
        """生成脉冲信号 / Generate pulse signal
        
        Args:
            steps: 脉冲数量
            direction: 方向 1-正向 0-反向
        """
        if self.is_running:
            self.logger.warning("电机正在运行中")
            return False
        
        self.is_running = True
        self.dir.value(direction)
        utime.sleep_us(63)  # 方向信号建立时间
        
        try:
            # 生成指定数量的脉冲
            pulse_period_us = 1000000 // self.frequency
            
            for _ in range(steps):
                self.pul.value(1)
                utime.sleep_us(pulse_period_us // 2)
                self.pul.value(0)
                utime.sleep_us(pulse_period_us // 2)
            
        except Exception as e:
            self.logger.error(f"脉冲生成失败: {e}")
        finally:
            self.is_running = False
        
        return True
    
    def open_door(self, callback=None):
        """开门 / Open door"""
        if self.is_open:
            self.logger.warning("门已经是开启状态")
            return False
        
        self.logger.info("正在开门...")
        
        try:
            # 使能电机
            self.enable()
            
            # 计算需要的脉冲数
            steps = (self.frequency * self.open_duration) // 1000
            
            # 正向旋转开门
            self.generate_pulse(steps, direction=1)
            
            self.is_open = True
            
            # 记录开门事件
            self.logger.log_door_event("OPEN", method="motor")
            
            # 设置自动关门定时器
            self.schedule_close_door()
            
            # 执行回调
            if callback:
                callback()
            
            return True
            
        except Exception as e:
            self.logger.error(f"开门失败: {e}")
            return False
        finally:
            # 暂时保持使能，等待自动关门
            pass
    
    def close_door(self, callback=None):
        """关门 / Close door"""
        if not self.is_open:
            self.logger.warning("门已经是关闭状态")
            return False
        
        self.logger.info("正在关门...")
        
        try:
            # 计算需要的脉冲数
            steps = (self.frequency * self.close_duration) // 1000
            
            # 反向旋转关门
            self.generate_pulse(steps, direction=0)
            
            self.is_open = False
            
            # 记录关门事件
            self.logger.log_door_event("CLOSE", method="motor")
            
            # 执行回调
            if callback:
                callback()
            
            return True
            
        except Exception as e:
            self.logger.error(f"关门失败: {e}")
            return False
        finally:
            # 关门后禁用电机
            self.disable()
    
    def schedule_close_door(self, delay_ms=5000):
        """定时自动关门 / Schedule auto close door
        
        Args:
            delay_ms: 延迟时间（毫秒）
        """
        try:
            if self.timer:
                self.timer.deinit()
            
            self.timer = Timer(-1)
            self.timer.init(period=delay_ms, mode=Timer.ONE_SHOT, 
                          callback=lambda t: self.close_door())
            
            self.logger.debug(f"已设置{delay_ms}ms后自动关门")
            
        except Exception as e:
            self.logger.error(f"设置自动关门失败: {e}")
    
    def emergency_stop(self):
        """紧急停止 / Emergency stop"""
        self.is_running = False
        self.pul.value(0)
        self.disable()
        if self.timer:
            self.timer.deinit()
        self.logger.warning("电机紧急停止")
    
    def test_motor(self):
        """测试电机 / Test motor"""
        self.logger.info("开始电机测试...")
        
        try:
            # 测试使能
            self.enable()
            utime.sleep(1)
            
            # 测试正向旋转
            self.logger.info("测试正向旋转...")
            self.generate_pulse(200, direction=1)
            utime.sleep(1)
            
            # 测试反向旋转
            self.logger.info("测试反向旋转...")
            self.generate_pulse(200, direction=0)
            utime.sleep(1)
            
            # 禁用电机
            self.disable()
            
            self.logger.info("电机测试完成")
            return True
            
        except Exception as e:
            self.logger.error(f"电机测试失败: {e}")
            return False
        
    def deinit(self):
        """释放资源 / Release resources"""
        self.emergency_stop()
        self.logger.info("电机控制器已释放")