from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os
import ujson
from media.media import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import time
import image
import aidemo
import random
import gc
import sys
import math
from utils.logger import get_logger
from utils.config_loader import ConfigLoader
from core.face_recognition import FaceDetApp, FaceRegistrationApp

class FaceRegister:
    """人脸注册类 / Face registration class"""
    
    def __init__(self):
        self.logger = get_logger()
        self.config = ConfigLoader()
        
        # 加载模型路径
        self.face_det_kmodel = self.config.get('face_recognition.det_model_path')
        self.face_reg_kmodel = self.config.get('face_recognition.reg_model_path')
        self.anchors_path = self.config.get('face_recognition.anchors_path')
        self.database_dir = self.config.get('face_recognition.database_dir')
        
        # 模型参数
        self.det_input_size = [320, 320]
        self.reg_input_size = [112, 112]
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.2
        
        # 加载anchors
        self.anchors = np.fromfile(self.anchors_path, dtype=np.float)
        self.anchors = self.anchors.reshape((4200, 4))
        
        # 初始化检测和注册模型
        self.face_det = None
        self.face_reg = None
        
    def init_models(self):
        """初始化模型"""
        try:
            self.face_det = FaceDetApp(
                self.face_det_kmodel,
                model_input_size=self.det_input_size,
                anchors=self.anchors,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                debug_mode=0
            )
            
            self.face_reg = FaceRegistrationApp(
                self.face_reg_kmodel,
                model_input_size=self.reg_input_size
            )
            
            self.logger.info("人脸注册模型初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            return False
    
    def ensure_dir(self, directory):
        """递归创建目录"""
        if not directory or directory == '/':
            return

        directory = directory.rstrip('/')

        try:
            os.stat(directory)
            self.logger.debug(f'目录已存在: {directory}')
            return
        except OSError:
            # 目录不存在，需要创建
            if '/' in directory:
                parent = directory[:directory.rindex('/')]
                if parent and parent != directory:
                    self.ensure_dir(parent)

            try:
                os.mkdir(directory)
                self.logger.info(f'已创建目录: {directory}')
            except OSError as e:
                try:
                    os.stat(directory)
                    self.logger.debug(f'目录已被其他进程创建: {directory}')
                except:
                    self.logger.error(f'创建目录时出错: {e}')
                    
    def get_directory_name(self, path):
        """获取路径中的目录名"""
        parts = path.split('/')
        for part in reversed(parts):
            if part:
                return part
        return ''
    
    def register_from_image(self, img_path, person_name=None):
        """从图像文件注册人脸
        
        Args:
            img_path: 图像文件路径
            person_name: 人员名称，如果为None则从文件名提取
        """
        try:
            if not self.face_det or not self.face_reg:
                if not self.init_models():
                    return False
            
            # 读取图像
            self.logger.info(f"读取图像: {img_path}")
            img = image.Image(img_path)
            img.compress_for_ide()
            
            # 转换图像格式
            rgb888p_img_ndarry = self.image2rgb888array(img)
            
            # 配置人脸检测预处理
            self.face_det.config_preprocess(
                input_image_size=[rgb888p_img_ndarry.shape[3], rgb888p_img_ndarry.shape[2]]
            )
            
            # 执行人脸检测
            det_boxes, landms = self.face_det.run(rgb888p_img_ndarry)
            
            if det_boxes and det_boxes.shape[0] == 1:
                # 只有一张人脸时才进行注册
                if not person_name:
                    person_name = os.path.basename(img_path).split('.')[0]
                
                for landm in landms:
                    # 配置人脸注册预处理
                    self.face_reg.config_preprocess(
                        landm,
                        input_image_size=[rgb888p_img_ndarry.shape[3], rgb888p_img_ndarry.shape[2]]
                    )
                    
                    # 提取人脸特征
                    reg_result = self.face_reg.run(rgb888p_img_ndarry)
                    
                    # 确保数据库目录存在
                    self.ensure_dir(self.database_dir)
                    
                    # 保存特征到数据库
                    feature_file = self.database_dir + '{}.bin'.format(person_name)
                    with open(feature_file, "wb") as file:
                        file.write(reg_result.tobytes())
                        
                    self.logger.info(f'成功注册人脸: {person_name}')
                    return True
                    
            elif det_boxes and det_boxes.shape[0] > 1:
                self.logger.warning('检测到多张人脸，注册失败')
                return False
            else:
                self.logger.warning('未检测到人脸')
                return False
                
        except Exception as e:
            self.logger.error(f"注册失败: {e}")
            return False
    
    def register_from_directory(self, directory):
        """从目录批量注册人脸
        
        Args:
            directory: 包含人脸图片的目录
        """
        try:
            if not os.path.exists(directory):
                self.logger.error(f"目录不存在: {directory}")
                return False
            
            # 获取目录名作为数据库子目录
            dir_name = self.get_directory_name(directory)
            database_subdir = self.database_dir + dir_name + "/"
            self.ensure_dir(database_subdir)
            
            # 获取图像列表
            img_list = os.listdir(directory)
            success_count = 0
            
            for img_file in img_list:
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    full_img_path = directory + img_file
                    person_name = img_file.split('.')[0]
                    
                    self.logger.info(f"处理: {full_img_path}")
                    
                    if self.register_from_image(full_img_path, person_name):
                        success_count += 1
                    
                    gc.collect()
            
            self.logger.info(f"批量注册完成: 成功 {success_count}/{len(img_list)}")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"批量注册失败: {e}")
            return False
    
    def register_from_camera(self, person_name, pl=None):
        """从摄像头实时注册人脸
        
        Args:
            person_name: 要注册的人员名称
            pl: Pipeline对象，如果为None则创建新的
        """
        try:
            if not self.face_det or not self.face_reg:
                if not self.init_models():
                    return False
            
            # 创建Pipeline如果没有提供
            if pl is None:
                rgb888p_size = self.config.get('display.rgb888p_size', [640, 480])
                display_size = self.config.get('display.display_size', [640, 480])
                display_mode = self.config.get('display.display_mode', 'lcd')
                
                pl = PipeLine(rgb888p_size=rgb888p_size,
                            display_size=display_size,
                            display_mode=display_mode)
                pl.create()
                need_cleanup = True
            else:
                need_cleanup = False
            
            self.logger.info(f"开始从摄像头注册人脸: {person_name}")
            self.logger.info("请保持面部在镜头中央，按任意键开始捕获...")
            
            # 等待合适的人脸
            max_attempts = 100
            attempt = 0
            
            while attempt < max_attempts:
                img = pl.get_frame()
                
                # 人脸检测
                det_boxes, landms = self.face_det.run(img)
                
                if det_boxes and det_boxes.shape[0] == 1:
                    # 检测到单个人脸
                    self.face_det.draw_result(pl, det_boxes)
                    pl.show_image()
                    
                    # 注册人脸
                    for landm in landms:
                        self.face_reg.config_preprocess(landm)
                        feature = self.face_reg.run(img)
                        
                        # 确保数据库目录存在
                        self.ensure_dir(self.database_dir)
                        
                        # 保存特征
                        feature_file = self.database_dir + '{}.bin'.format(person_name)
                        with open(feature_file, "wb") as file:
                            file.write(feature.tobytes())
                        
                        self.logger.info(f'成功注册人脸: {person_name}')
                        
                        if need_cleanup:
                            pl.destroy()
                        
                        return True
                        
                elif det_boxes and det_boxes.shape[0] > 1:
                    self.logger.warning("检测到多张人脸，请确保只有一个人在镜头前")
                else:
                    self.logger.debug("未检测到人脸，请面向镜头")
                
                pl.show_image()
                gc.collect()
                attempt += 1
                time.sleep(0.1)
            
            self.logger.error("注册超时，未能捕获合适的人脸")
            
            if need_cleanup:
                pl.destroy()
                
            return False
            
        except Exception as e:
            self.logger.error(f"从摄像头注册失败: {e}")
            return False
    
    def image2rgb888array(self, img):
        """将图像转换为RGB888数组"""
        with ScopedTiming("image2rgb888array", False):
            img_data_rgb888 = img.to_rgb888()
            img_hwc = img_data_rgb888.to_numpy_ref()
            shape = img_hwc.shape
            img_tmp = img_hwc.reshape((shape[0] * shape[1], shape[2]))
            img_tmp_trans = img_tmp.transpose()
            img_res = img_tmp_trans.copy()
            img_return = img_res.reshape((1, shape[2], shape[0], shape[1]))
        return img_return
    
    def delete_registration(self, person_name):
        """删除人脸注册
        
        Args:
            person_name: 要删除的人员名称
        """
        try:
            feature_file = self.database_dir + '{}.bin'.format(person_name)
            
            if os.path.exists(feature_file):
                os.remove(feature_file)
                self.logger.info(f"已删除人脸注册: {person_name}")
                return True
            else:
                self.logger.warning(f"未找到注册信息: {person_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"删除注册失败: {e}")
            return False
    
    def list_registrations(self):
        """列出所有已注册的人脸"""
        try:
            if not os.path.exists(self.database_dir):
                self.logger.warning("数据库目录不存在")
                return []
            
            registrations = []
            db_files = os.listdir(self.database_dir)
            
            for db_file in db_files:
                if db_file.endswith('.bin'):
                    name = db_file.replace('.bin', '')
                    file_path = self.database_dir + db_file
                    file_stat = os.stat(file_path)
                    registrations.append({
                        'name': name,
                        'file': file_path,
                        'size': file_stat[6]  # 文件大小
                    })
            
            self.logger.info(f"找到 {len(registrations)} 个注册的人脸")
            return registrations
            
        except Exception as e:
            self.logger.error(f"列出注册失败: {e}")
            return []
    
    def deinit(self):
        """释放资源"""
        try:
            if self.face_det:
                self.face_det.deinit()
            if self.face_reg:
                self.face_reg.deinit()
            self.logger.info("人脸注册模块已释放")
        except Exception as e:
            self.logger.error(f"释放注册模块失败: {e}")