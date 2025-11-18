#!/usr/bin/env micropython
"""
K230 智能开门系统 / K230 Smart Door System

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
sleep_mgr = 35

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
        
        # 初始化HTTP客户端
        if network_mgr.init_http_client():
            logger.info("HTTP客户端初始化成功")
            
            # 发送设备上线事件
            device_info = {
                'device_id': config.get('system.device_id', 'unknown'),
                'device_name': config.get('system.device_name', 'K230'),
                'firmware_version': config.get('system.version', '1.0.0')
            }
            network_mgr.send_event('device_online', device_info)
        else:
            logger.warning("HTTP客户端初始化失败")
    else:
        door_ctrl.audio.play_feedback('network_failed')
        logger.warning("网络连接失败，运行在离线模式")
    
    # 网络监控循环
    last_heartbeat = utime.time()
    while running:
        try:
            # 自动重连检查
            if config.get('network.auto_reconnect', True):
                if not network_mgr.check_connection():
                    if network_mgr.auto_reconnect():
                        network_mgr.init_http_client()
            
            # 发送心跳
            current_time = utime.time()
            if current_time - last_heartbeat > 30:  # 每30秒发送一次心跳
                if network_mgr.is_connected:
                    response = network_mgr.http_get('/api/heartbeat')
                    if response and response['status_code'] == 200:
                        logger.debug("心跳发送成功")
                    last_heartbeat = current_time
            
            utime.sleep(5)  # 每5秒检查一次
            
        except Exception as e:
            logger.error(f"网络线程错误: {e}")
            utime.sleep(5)
    
    # 发送设备离线事件
    if network_mgr.is_connected:
        network_mgr.send_event('device_offline', {'reason': 'shutdown'})
    
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
    last_log_time = 0  # 用于控制日志输出频率
    
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
                
                # 处理检测结果
                if det_boxes and len(det_boxes) > 0:
                    # 检测到人脸，重置无人脸计数
                    no_face_count = 0
                    sleep_mgr.update_activity()
                elif recg_res and 'in_vacuum' not in recg_res:
                    # 没有检测到人脸且不在真空期
                    no_face_count += 1
                    if no_face_count >= max_no_face_count:
                        # 进入休眠模式
                        sleep_mgr.enter_sleep()
                        no_face_count = 0
                    
                    # 清空显示
                    pipeline.show_image()
                    gc.collect()
                    utime.sleep_ms(10)
                    continue
                
                # 有人脸时的处理逻辑
                if det_boxes and len(det_boxes) > 0:
                    
                    # 保存当前帧图像（用于上传）
                    try:
                        if door_ctrl and img is not None:
                            # 将图像数据传递给door_ctrl以便后续上传
                            # 转换为JPEG格式的bytes
                            import ubinascii
                            if hasattr(img, 'tobytes'):
                                image_bytes = img.tobytes()
                            else:
                                image_bytes = bytes(img)
                            door_ctrl.set_captured_image(image_bytes)
                    except Exception as e:
                        logger.debug(f"保存图像失败: {e}")
                    
                    # 绘制结果
                    face_recognizer.draw_result(pipeline, det_boxes, recg_res)
                    
                    # 检查是否需要处理结果（触发音频和日志）
                    if face_recognizer.should_process_result(recg_res):
                        trigger_type = face_recognizer.get_trigger_type(recg_res)
                        
                        if trigger_type == 'success':
                            # 识别成功（窗口内首次成功，立即触发）
                            person = face_recognizer.last_recognized_name
                            score = face_recognizer.last_recognized_score
                            
                            logger.info(f"识别成功: {person} (置信度: {score:.2f})")
                            door_ctrl.grant_access(person, "face", score)
                            
                        elif trigger_type == 'failed':
                            # 识别失败（5秒窗口全部失败）
                            logger.warning("5秒内全部识别失败，拒绝访问")
                            door_ctrl.deny_access("unregistered", "unknown")
                    else:
                        # 在真空期或窗口内等待，仅显示实时状态
                        if 'in_vacuum' in recg_res:
                            # 真空期中，显示剩余时间
                            remaining = face_recognizer.get_vacuum_remaining_time()
                            if remaining > 0:
                                logger.debug(f"真空期剩余: {remaining:.1f}秒")
                        else:
                            # 窗口内正常识别，降低日志频率避免刷屏
                            current_time = utime.ticks_ms() / 1000.0
                            if current_time - last_log_time > 1.0:  # 每秒最多输出一次
                                person, score = face_recognizer.get_recognized_person(recg_res)
                                if person:
                                    logger.debug(f"窗口内识别到: {person} ({score:.2f})")
                                elif len(recg_res) > 0 and 'already_success' not in recg_res:
                                    elapsed = face_recognizer.get_window_elapsed_time()
                                    if elapsed > 0:
                                        logger.debug(f"识别窗口: 检测到未知人脸 ({elapsed:.1f}/5.0秒)")
                                last_log_time = current_time
                    
                # 显示画面
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