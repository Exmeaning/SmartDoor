from machine import Pin, PWM
import utime
import uos as os
from media.media import MediaManager
from media.pyaudio import PyAudio
import media.wave as wave
from utils.logger import get_logger
from utils.config_loader import ConfigLoader

class AudioManager:
    """音频管理器 / Audio Manager"""
    
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = get_logger()
        
        # 扬声器引脚配置 GPIO42
        speaker_pin = self.config.get('audio.speaker_pin', 42)
        self.speaker = Pin(speaker_pin, Pin.OUT, value=0)
        
        # 音频文件路径
        self.audio_files = self.config.get('audio.audio_files', {})
        
        self.is_playing = False
    
    def enable_speaker(self):
        """启用扬声器 / Enable speaker"""
        self.speaker.value(1)
        self.logger.debug("扬声器已启用")
    
    def disable_speaker(self):
        """禁用扬声器 / Disable speaker"""
        self.speaker.value(0)
        self.logger.debug("扬声器已禁用")
    
    def play_tone(self, frequency, duration_ms):
        """播放单音 / Play tone
        
        Args:
            frequency: 频率(Hz)
            duration_ms: 持续时间(毫秒)
        """
        try:
            pwm = PWM(self.speaker, freq=frequency, duty=512)
            utime.sleep_ms(duration_ms)
            pwm.deinit()
            
        except Exception as e:
            self.logger.error(f"播放音调失败: {e}")
    
    def play_beep(self, count=1, frequency=1000, duration=100):
        """播放蜂鸣声 / Play beep
        
        Args:
            count: 蜂鸣次数
            frequency: 频率(Hz)
            duration: 每次持续时间(毫秒)
        """
        for i in range(count):
            self.play_tone(frequency, duration)
            if i < count - 1:
                utime.sleep_ms(100)
    
    def play_wav_file(self, wav_file):
        """播放WAV文件 / Play WAV file"""
        if self.is_playing:
            self.logger.warning("音频正在播放中")
            return False
        
        self.is_playing = True
        
        # 检查文件是否存在
        try:
            os.stat(wav_file)
        except OSError:
            self.logger.warning(f"音频文件不存在: {wav_file}，使用蜂鸣提示")
            self._play_beep_fallback(wav_file)
            self.is_playing = False
            return False
        
        wf = None
        p = None
        stream = None
        
        try:
            self.enable_speaker()
            
            # 打开wav文件
            wf = wave.open(wav_file, 'rb')
            
            # 设置音频chunk值
            CHUNK = int(wf.get_framerate() / 25)
            
            # 初始化PyAudio
            p = PyAudio()
            p.initialize(CHUNK)
            MediaManager.init()
            
            # 创建音频输出流
            stream = p.open(
                format=p.get_format_from_width(wf.get_sampwidth()),
                channels=wf.get_channels(),
                rate=wf.get_framerate(),
                output=True,
                frames_per_buffer=CHUNK
            )
            
            # 设置音量 75%
            stream.volume(vol=75)
            
            # 读取并播放音频数据
            data = wf.read_frames(CHUNK)
            while data:
                stream.write(data)
                data = wf.read_frames(CHUNK)
            
            self.logger.info(f"播放完成: {wav_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"播放音频失败: {e}，使用蜂鸣提示")
            self._play_beep_fallback(wav_file)
            return False
            
        finally:
            # 清理资源
            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
                if p:
                    p.terminate()
                if wf:
                    wf.close()
                MediaManager.deinit()
            except:
                pass
            
            self.disable_speaker()
            self.is_playing = False
    
    def _play_beep_fallback(self, wav_file):
        """备用蜂鸣方案 / Fallback beep"""
        self.logger.info(f"播放音频提示: {wav_file}")
        
        # 根据不同的音频文件播放不同的蜂鸣模式
        if "welcome" in wav_file:
            self.play_beep(2, 800, 100)
        elif "granted" in wav_file:
            self.play_beep(1, 1000, 200)
        elif "denied" in wav_file:
            self.play_beep(3, 500, 100)
        elif "connected" in wav_file:
            self.play_tone(1200, 300)
        elif "failed" in wav_file:
            self.play_beep(2, 400, 200)
        elif "success" in wav_file:
            self.play_tone(1500, 500)
        else:
            self.play_beep(1, 1000, 100)
    
    def play_feedback(self, event_type):
        """播放反馈音 / Play feedback sound
        
        Args:
            event_type: 事件类型
        """
        audio_map = {
            'welcome': 'welcome',
            'access_granted': 'access_granted',
            'access_denied': 'access_denied',
            'network_connected': 'network_connected',
            'network_failed': 'network_failed',
            'register_success': 'register_success'
        }
        
        if event_type in audio_map:
            audio_key = audio_map[event_type]
            audio_file = self.audio_files.get(audio_key)
            
            if audio_file:
                self.play_wav_file(audio_file)
            else:
                # 备用蜂鸣声
                self.play_beep(1, 1000, 100)
        else:
            self.logger.warning(f"未知的音频事件类型: {event_type}")
    
    def test_audio(self):
        """测试音频系统 / Test audio system"""
        self.logger.info("开始音频测试...")
        
        try:
            # 测试蜂鸣声
            self.logger.info("测试蜂鸣声...")
            self.play_beep(3, 1000, 100)
            utime.sleep(1)
            
            # 测试不同音调
            self.logger.info("测试音调...")
            for freq in [500, 800, 1000, 1500]:
                self.play_tone(freq, 200)
                utime.sleep_ms(200)
            
            self.logger.info("音频测试完成")
            return True
            
        except Exception as e:
            self.logger.error(f"音频测试失败: {e}")
            return False
    
    def deinit(self):
        """释放资源 / Release resources"""
        try:
            self.disable_speaker()
            self.logger.info("音频管理器已释放")
        except Exception as e:
            self.logger.error(f"释放音频资源失败: {e}")