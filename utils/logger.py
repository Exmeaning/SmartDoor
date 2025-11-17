import utime
import os
import gc

class Logger:
    """日志管理器 / Logger manager"""
    
    LEVELS = {
        'DEBUG': 0,
        'INFO': 1,
        'WARNING': 2,
        'ERROR': 3,
        'CRITICAL': 4
    }
    
    def __init__(self, log_dir='/sdcard/logs/', log_level='INFO'):
        self.log_dir = log_dir
        self.log_level = self.LEVELS.get(log_level, 1)
        self.ensure_log_dir()
        self.current_log_file = None
        self.open_log_file()
    
    def ensure_log_dir(self):
        """确保日志目录存在 / Ensure log directory exists"""
        try:
            os.stat(self.log_dir)
        except OSError:
            try:
                os.mkdir(self.log_dir)
                print(f"创建日志目录: {self.log_dir}")
            except OSError as e:
                print(f"无法创建日志目录: {e}")
    
    def open_log_file(self):
        """打开当天的日志文件 / Open today's log file"""
        try:
            # 获取当前日期
            t = utime.localtime()
            date_str = "{:04d}{:02d}{:02d}".format(t[0], t[1], t[2])
            log_filename = "{}door_{}.log".format(self.log_dir, date_str)
            
            # 打开日志文件（追加模式）
            self.current_log_file = log_filename
        except Exception as e:
            print(f"无法打开日志文件: {e}")
    
    def log(self, level, message, extra_data=None):
        """写入日志 / Write log"""
        if self.LEVELS.get(level, 0) < self.log_level:
            return
        
        try:
            t = utime.localtime()
            timestamp = "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(
                t[0], t[1], t[2], t[3], t[4], t[5]
            )
            
            log_entry = f"[{timestamp}] [{level}] {message}"
            if extra_data:
                log_entry += f" | {extra_data}"
            
            # 打印到控制台
            print(log_entry)
            
            # 写入文件
            if self.current_log_file:
                try:
                    with open(self.current_log_file, 'a') as f:
                        f.write(log_entry + '\n')
                except Exception as e:
                    print(f"写入日志失败: {e}")
            
            # 定期垃圾回收
            gc.collect()
            
        except Exception as e:
            print(f"日志记录错误: {e}")
    
    def debug(self, message, extra_data=None):
        self.log('DEBUG', message, extra_data)
    
    def info(self, message, extra_data=None):
        self.log('INFO', message, extra_data)
    
    def warning(self, message, extra_data=None):
        self.log('WARNING', message, extra_data)
    
    def error(self, message, extra_data=None):
        self.log('ERROR', message, extra_data)
    
    def critical(self, message, extra_data=None):
        self.log('CRITICAL', message, extra_data)
    
    def log_door_event(self, event_type, person_name=None, method="face", extra_info=None):
        """记录门禁事件 / Log door access event"""
        event_data = {
            "type": event_type,
            "person": person_name or "unknown",
            "method": method,
            "info": extra_info
        }
        self.info(f"DOOR_EVENT: {event_type}", event_data)
    
    def cleanup_old_logs(self, retention_days=30):
        """清理旧日志文件 / Clean up old log files"""
        try:
            current_time = utime.time()
            retention_seconds = retention_days * 24 * 3600
            
            for filename in os.listdir(self.log_dir):
                if filename.endswith('.log'):
                    filepath = self.log_dir + filename
                    file_stat = os.stat(filepath)
                    # 简单的时间比较，实际可能需要更精确的处理
                    if (current_time - file_stat[8]) > retention_seconds:
                        os.remove(filepath)
                        self.info(f"删除旧日志文件: {filename}")
        except Exception as e:
            self.error(f"清理日志失败: {e}")

# 全局日志实例
logger = None

def get_logger():
    """获取全局日志实例 / Get global logger instance"""
    global logger
    if logger is None:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader()
        log_level = config.get('system.log_level', 'INFO')
        logger = Logger(log_level=log_level)
    return logger