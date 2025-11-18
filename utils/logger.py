import utime
import uos as os
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
    
    def __init__(self, log_dir=None, log_level='INFO'):
        # 如果未指定路径，自动选择最佳可写路径
        if log_dir is None:
            log_dir = self._get_best_log_dir()
        
        self.log_dir = log_dir.rstrip('/') + '/'  # 确保以 / 结尾
        self.log_level = self.LEVELS.get(log_level, 1)
        self.ensure_log_dir()
        self.current_log_file = None
        self.open_log_file()
    
    def _get_best_log_dir(self):
        """自动选择最佳可写日志目录"""
        candidates = [
            '/sd/logs/',      # 优先 SD 卡
            '/sdcard/logs/',  # 兼容部分设备命名
            '/data/logs/',    # 次选内部可写分区
            '/tmp/logs/'      # 最后使用内存盘（重启丢失）
        ]
        
        for path in candidates:
            if self._test_write_permission(path):
                print(f"✅ 选择日志目录: {path}")
                return path
        
        # 全部失败，返回第一个并打印警告
        print("⚠️ 所有候选路径均不可写，强制使用: /tmp/logs/")
        return '/tmp/logs/'
    
    def _test_write_permission(self, path):
        """测试路径是否可写"""
        try:
            # 尝试递归创建目录
            parts = path.strip('/').split('/')
            current = '/'
            for part in parts:
                if not part:
                    continue
                current = f"{current}{part}/"
                try:
                    os.stat(current)
                except OSError:
                    try:
                        os.mkdir(current)
                    except Exception:
                        return False
            
            # 尝试写入临时文件
            test_file = f"{path}.write_test.tmp"
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except Exception as e:
            print(f"路径 {path} 不可写: {e}")
            return False
    
    def ensure_log_dir(self):
        """确保日志目录存在 / Ensure log directory exists"""
        try:
            os.stat(self.log_dir)
        except OSError:
            try:
                # 递归创建多级目录
                parts = self.log_dir.strip('/').split('/')
                current = '/'
                for part in parts:
                    if not part:
                        continue
                    current = f"{current}{part}/"
                    try:
                        os.stat(current)
                    except OSError:
                        os.mkdir(current)
                print(f"📁 创建日志目录: {self.log_dir}")
            except OSError as e:
                print(f"❌ 无法创建日志目录: {e}")
                raise Exception(f"致命错误：日志目录不可用 {self.log_dir}")

    def open_log_file(self):
        """打开当天的日志文件 / Open today's log file"""
        try:
            # 获取当前日期
            t = utime.localtime()
            date_str = "{:04d}{:02d}{:02d}".format(t[0], t[1], t[2])
            log_filename = f"{self.log_dir}door_{date_str}.log"
            
            # 测试能否写入该文件
            try:
                with open(log_filename, 'a') as f:
                    pass  # 只测试能否打开
                self.current_log_file = log_filename
                print(f"📄 日志文件已就绪: {log_filename}")
            except Exception as e:
                print(f"❌ 无法打开日志文件 {log_filename}: {e}")
                self.current_log_file = None
                
        except Exception as e:
            print(f"❌ 初始化日志文件失败: {e}")
            self.current_log_file = None
    
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
                    with open(self.current_log_file, 'a', encoding='utf-8') as f:
                        f.write(log_entry + '\n')
                except Exception as e:
                    print(f"❌ 写入日志失败 ({self.current_log_file}): {e}")
                    # 可选：尝试重新 open 文件或切换路径
                    # self._fallback_log_write(log_entry)
            
            # 定期垃圾回收
            gc.collect()
            
        except Exception as e:
            print(f"❌ 日志记录错误: {e}")
    
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
            if not self.current_log_file:
                return
                
            current_time = utime.time()
            retention_seconds = retention_days * 24 * 3600
            
            try:
                files = os.listdir(self.log_dir)
            except Exception as e:
                self.error(f"无法列出日志目录: {e}")
                return
                
            for filename in files:
                if filename.startswith('door_') and filename.endswith('.log'):
                    filepath = self.log_dir + filename
                    try:
                        file_stat = os.stat(filepath)
                        if (current_time - file_stat[8]) > retention_seconds:
                            os.remove(filepath)
                            self.info(f"🗑️ 删除旧日志文件: {filename}")
                    except Exception as e:
                        self.error(f"删除日志文件失败 {filename}: {e}")
                        
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
        # ✅ 关键：不再硬编码路径，让 Logger 自动选择
        logger = Logger(log_level=log_level)
    return logger