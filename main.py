#!/usr/bin/env micropython
"""
K230 智能门锁系统 / K230 Smart Door Lock System

主程序入口 / Main program entry
Author: Exmeaning
Version: 1.0.0
"""

import _thread
import utime
import gc
import sys

# 导入系统模块
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.sleep_manager import SleepManager
from core.network_manager import NetworkManager
from core.audio_manager import AudioManager
from modules.door_controller import DoorController

# 导入人脸识别相关
from libs.PipeLine import PipeLine
import ulab.numpy as np

# 全局变量
config = None
logger = None
network_mgr = None
sleep_mgr = None
door_ctrl = None
face_recognizer = None
pipeline = None
running = True

def init_system():
    """初始化系统 / Initialize system"""
    global config, logger, network_mgr, sleep_mgr, door_ctrl, pipeline
    
    print("=" * 50)
    print("K230 智能门锁系统启动 / K230 Smart Door Lock Starting")
    print("=" * 50)
    
    try:
        # 加载配置
        config = ConfigLoader()
        logger = get_logger()
        
        logger.info("系统初始化开始...")
        
        # 初始化休眠管理器
        sleep_mgr = SleepManager()
        
        # 初始化网络
        network_mgr = NetworkManager()
        
        # 初始化门禁控制器
        door_ctrl = DoorController()
        
        # 初始化视频管道
        rgb888p_size = config.get('display.rgb888p_size', [640, 480])
        display_size = config.get('display.display_size', [640, 480])
        display_mode = config.get('display.display_mode', 'lcd')
        
        pipeline = PipeLine(rgb888p_size=rgb888p_size,
                          display_size=display_size,
                          display_mode=display_mode)
        pipeline.create()
        
        # 设置休眠管理器的pipeline
        sleep_mgr.set_pipeline(pipeline)
        
        logger.info("系统初始化完成")
        
        # 播放欢迎音
        door_ctrl.audio.play_feedback('welcome')
        
        return True
        
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return False

def network_thread():
    """网络线程 / Network thread"""
    global network_mgr, door_ctrl, running
    
    logger = get_logger()
    logger.info("网络线程启动")
    
    # 尝试连接WiFi
    if network_mgr.connect_wifi():
        door_ctrl.audio.play_feedback('network_connected')
        
        # 启动TCP客户端
        # network_mgr.create_tcp_client()
    else:
        door_ctrl.audio.play_feedback('network_failed')
        logger.warning("网络连接失败，运行在离线模式")
    
    # 网络监控循环
    while running:
        try:
            # 自动重连检查
            if config.get('network.auto_reconnect', True):
                if not network_mgr.check_connection():
                    network_mgr.auto_reconnect()
            
            utime.sleep(5)  # 每5秒检查一次
            
        except Exception as e:
            logger.error(f"网络线程错误: {e}")
            utime.sleep(5)
    
    logger.info("网络线程退出")

def face_recognition_thread():
    """人脸识别线程 / Face recognition thread"""
    global face_recognizer, pipeline, door_ctrl, sleep_mgr, running
    
    logger = get_logger()
    logger.info("人脸识别线程启动")
    
    # 导入并初始化人脸识别模块
    try:
        from core.face_recognition import FaceRecognition
        
        # 加载配置
        config = ConfigLoader()
        face_det_kmodel_path = config.get('face_recognition.det_model_path')
        face_reg_kmodel_path = config.get('face_recognition.reg_model_path')
        anchors_path = config.get('face_recognition.anchors_path')
        database_dir = config.get('face_recognition.database_dir')
        
        # 根据设备ID创建特定的数据库目录
        device_id = config.get('system.device_id', 'default')
        database_dir = database_dir + device_id + "/"
        
        # 加载参数
        face_det_input_size = [320, 320]
        face_reg_input_size = [112, 112]
        confidence_threshold = config.get('face_recognition.confidence_threshold', 0.5)
        nms_threshold = config.get('face_recognition.nms_threshold', 0.2)
        face_recognition_threshold = config.get('face_recognition.recognition_threshold', 0.65)
        
        # 加载anchor数据
        anchors = np.fromfile(anchors_path, dtype=np.float)
        anchors = anchors.reshape((4200, 4))
        
        rgb888p_size = config.get('display.rgb888p_size', [640, 480])
        display_size = config.get('display.display_size', [640, 480])
        
        # 创建人脸识别对象
        face_recognizer = FaceRecognition(
            face_det_kmodel_path,
            face_reg_kmodel_path,
            det_input_size=face_det_input_size,
            reg_input_size=face_reg_input_size,
            database_dir=database_dir,
            anchors=anchors,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            face_recognition_threshold=face_recognition_threshold,
            rgb888p_size=rgb888p_size,
            display_size=display_size
        )
        
        logger.info("人脸识别模块初始化成功")
        
    except Exception as e:
        logger.error(f"初始化人脸识别模块失败: {e}")
        return
    
    # 人脸识别主循环
    no_face_count = 0
    max_no_face_count = 30  # 连续30次无人脸进入休眠
    
    while running:
        try:
            # 检查是否在休眠模式
            if sleep_mgr.is_sleeping:
                # 休眠模式下的低频检测
                img = pipeline.get_frame()
                det_boxes, _ = face_recognizer.run(img)
                
                if det_boxes and len(det_boxes) > 0:
                    # 检测到人脸，唤醒系统
                    sleep_mgr.update_activity()
                    logger.info("检测到人脸，系统唤醒")
                else:
                    # 继续休眠
                    utime.sleep_ms(config.get('face_recognition.sleep_check_interval', 1000))
                    
            else:
                # 正常模式下的人脸识别
                img = pipeline.get_frame()
                det_boxes, recg_res = face_recognizer.run(img)
                
                if det_boxes and len(det_boxes) > 0:
                    # 重置无人脸计数
                    no_face_count = 0
                    sleep_mgr.update_activity()
                    
                    # 绘制结果
                    face_recognizer.draw_result(pipeline, det_boxes, recg_res)
                    
                    # 处理识别结果
                    recognized_person, confidence = face_recognizer.get_recognized_person(recg_res)
                    
                    if recognized_person:
                        # 识别成功，开门
                        logger.info(f"识别成功: {recognized_person} (置信度: {confidence:.2f})")
                        door_ctrl.grant_access(recognized_person, "face", confidence)
                    else:
                        # 检测到人脸但未识别
                        if len(recg_res) > 0:
                            logger.warning("检测到未注册人脸")
                            door_ctrl.deny_access("unregistered", "unknown")
                    
                else:
                    # 未检测到人脸
                    no_face_count += 1
                    if no_face_count >= max_no_face_count:
                        # 进入休眠模式
                        sleep_mgr.enter_sleep()
                        no_face_count = 0
                
                pipeline.show_image()
            
            gc.collect()
            utime.sleep_ms(10)
            
        except Exception as e:
            logger.error(f"人脸识别线程错误: {e}")
            utime.sleep(1)
    
    # 清理资源
    try:
        if face_recognizer:
            face_recognizer.deinit()
    except:
        pass
        
    logger.info("人脸识别线程退出")

def maintenance_thread():
    """维护线程 / Maintenance thread"""
    global logger, running
    
    logger = get_logger()
    logger.info("维护线程启动")
    
    while running:
        try:
            # 定期清理日志
            logger.cleanup_old_logs(30)
            
            # 垃圾回收
            gc.collect()
            
            # 打印系统状态
            logger.debug(f"内存使用: {gc.mem_alloc()} / {gc.mem_free()}")
            
            # 每小时执行一次
            utime.sleep(3600)
            
        except Exception as e:
            logger.error(f"维护线程错误: {e}")
            utime.sleep(3600)
    
    logger.info("维护线程退出")

def main():
    """主函数 / Main function"""
    global running
    
    # 初始化系统
    if not init_system():
        print("系统初始化失败，退出程序")
        return
    
    try:
        # 启动网络线程
        _thread.start_new_thread(network_thread, ())
        utime.sleep_ms(500)
        
        # 启动维护线程
        _thread.start_new_thread(maintenance_thread, ())
        utime.sleep_ms(500)
        
        # 主线程运行人脸识别
        face_recognition_thread()
        
    except KeyboardInterrupt:
        logger.info("收到退出信号")
    except Exception as e:
        logger.error(f"主程序异常: {e}")
    finally:
        # 清理资源
        running = False
        logger.info("正在清理资源...")
        
        if face_recognizer:
            try:
                face_recognizer.deinit()
            except:
                pass
        if door_ctrl:
            door_ctrl.deinit()
        if network_mgr:
            network_mgr.disconnect_wifi()
        
        logger.info("系统已关闭")

if __name__ == "__main__":
    main()