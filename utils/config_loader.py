import ujson as json
import os

class ConfigLoader:
    """配置文件加载器 / Configuration file loader"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path='/sdcard/config.json'):
        """加载配置文件 / Load configuration file"""
        try:
            # 尝试使用ujson，如果不存在则使用json模块
            try:
                import ujson as json
            except ImportError:
                # 使用libs.Utils中的read_json方法
                from libs.Utils import read_json
                self._config = read_json(config_path)
                return
                
            with open(config_path, 'r') as f:
                self._config = json.load(f)
            print("配置文件加载成功 / Config loaded successfully")
        except Exception as e:
            print(f"配置文件加载失败，使用默认配置 / Config load failed: {e}")
            self._config = self.get_default_config()
    
    def get_default_config(self):
        """获取默认配置 / Get default configuration"""
        return {
            "system": {
                "debug_mode": 0,
                "log_level": "INFO"
            },
            "network": {
                "wifi_ssid": "TEST",
                "wifi_password": "12345678"
            },
            "face_recognition": {
                "confidence_threshold": 0.5,
                "recognition_threshold": 0.65
            },
            "motor": {
                "pul_pin": 42,
                "dir_pin": 33,
                "en_pin": 32
            }
        }
    
    def get(self, key_path, default=None):
        """获取配置项 / Get configuration item
        key_path: 使用点号分隔的配置路径，如 'network.wifi_ssid'
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path, value):
        """设置配置项 / Set configuration item"""
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save_config(self, config_path='/sdcard/config.json'):
        """保存配置文件 / Save configuration file"""
        try:
            import ujson as json
            with open(config_path, 'w') as f:
                json.dump(self._config, f)
            print("配置文件保存成功 / Config saved successfully")
        except Exception as e:
            print(f"配置文件保存失败 / Config save failed: {e}")