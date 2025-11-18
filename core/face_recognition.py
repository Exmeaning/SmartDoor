from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import uos as os
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
import re
from utils.logger import get_logger
from utils.config_loader import ConfigLoader
from libs.YbProtocol import YbProtocol
from ybUtils.YbUart import YbUart

class FaceDetApp(AIBase):
    """人脸检测应用类 / Face detection application class"""
    
    def __init__(self, kmodel_path, model_input_size, anchors, confidence_threshold=0.25,
                 nms_threshold=0.3, rgb888p_size=[640,480], display_size=[640,480], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode

        # 初始化AI2D
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                nn.ai2d_format.NCHW_FMT,
                                np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        """配置图像预处理参数"""
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self.ai2d.pad(self.get_pad_param(), 0, [104,117,123])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],
                          [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        """后处理方法"""
        with ScopedTiming("postprocess", self.debug_mode > 0):
            res = aidemo.face_det_post_process(self.confidence_threshold,
                                             self.nms_threshold,
                                             self.model_input_size[0],
                                             self.anchors,
                                             self.rgb888p_size,
                                             results)
            if len(res) == 0:
                return res, res
            else:
                return res[0], res[1]

    def get_pad_param(self):
        """计算padding参数"""
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        ratio = min(ratio_w, ratio_h)
        new_w = int(ratio * self.rgb888p_size[0])
        new_h = int(ratio * self.rgb888p_size[1])
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = int(round(0))
        bottom = int(round(dh * 2 + 0.1))
        left = int(round(0))
        right = int(round(dw * 2 - 0.1))
        return [0, 0, 0, 0, top, bottom, left, right]

class FaceRegistrationApp(AIBase):
    """人脸注册应用类 / Face registration application class"""
    
    def __init__(self, kmodel_path, model_input_size, rgb888p_size=[640,360],
                 display_size=[640,360], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode

        # 标准人脸关键点坐标
        self.umeyama_args_112 = [
            38.2946, 51.6963,
            73.5318, 51.5014,
            56.0252, 71.7366,
            41.5493, 92.3655,
            70.7299, 92.2041
        ]

        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                nn.ai2d_format.NCHW_FMT,
                                np.uint8, np.uint8)

    def config_preprocess(self, landm, input_image_size=None):
        """配置预处理参数"""
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            affine_matrix = self.get_affine_matrix(landm)
            self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, affine_matrix)
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],
                          [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        """后处理方法"""
        with ScopedTiming("postprocess", self.debug_mode > 0):
            return results[0][0]

    def svd22(self, a):
        """2x2矩阵的奇异值分解"""
        s = [0.0, 0.0]
        u = [0.0, 0.0, 0.0, 0.0]
        v = [0.0, 0.0, 0.0, 0.0]

        s[0] = (math.sqrt((a[0] - a[3]) ** 2 + (a[1] + a[2]) ** 2) +
                math.sqrt((a[0] + a[3]) ** 2 + (a[1] - a[2]) ** 2)) / 2
        s[1] = abs(s[0] - math.sqrt((a[0] - a[3]) ** 2 + (a[1] + a[2]) ** 2))

        v[2] = (math.sin((math.atan2(2 * (a[0] * a[1] + a[2] * a[3]),
                a[0] ** 2 - a[1] ** 2 + a[2] ** 2 - a[3] ** 2)) / 2)
                if s[0] > s[1] else 0)
        v[0] = math.sqrt(1 - v[2] ** 2)
        v[1] = -v[2]
        v[3] = v[0]

        u[0] = -(a[0] * v[0] + a[1] * v[2]) / s[0] if s[0] != 0 else 1
        u[2] = -(a[2] * v[0] + a[3] * v[2]) / s[0] if s[0] != 0 else 0
        u[1] = (a[0] * v[1] + a[1] * v[3]) / s[1] if s[1] != 0 else -u[2]
        u[3] = (a[2] * v[1] + a[3] * v[3]) / s[1] if s[1] != 0 else u[0]

        v[0] = -v[0]
        v[2] = -v[2]

        return u, s, v

    def image_umeyama_112(self, src):
        """使用Umeyama算法进行人脸对齐"""
        SRC_NUM = 5
        SRC_DIM = 2

        # 计算源点和目标点的均值
        src_mean = [0.0, 0.0]
        dst_mean = [0.0, 0.0]
        for i in range(0, SRC_NUM * 2, 2):
            src_mean[0] += src[i]
            src_mean[1] += src[i + 1]
            dst_mean[0] += self.umeyama_args_112[i]
            dst_mean[1] += self.umeyama_args_112[i + 1]

        src_mean[0] /= SRC_NUM
        src_mean[1] /= SRC_NUM
        dst_mean[0] /= SRC_NUM
        dst_mean[1] /= SRC_NUM

        # 去均值化
        src_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        dst_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        for i in range(SRC_NUM):
            src_demean[i][0] = src[2 * i] - src_mean[0]
            src_demean[i][1] = src[2 * i + 1] - src_mean[1]
            dst_demean[i][0] = self.umeyama_args_112[2 * i] - dst_mean[0]
            dst_demean[i][1] = self.umeyama_args_112[2 * i + 1] - dst_mean[1]

        # 计算A矩阵
        A = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(SRC_DIM):
            for k in range(SRC_DIM):
                for j in range(SRC_NUM):
                    A[i][k] += dst_demean[j][i] * src_demean[j][k]
                A[i][k] /= SRC_NUM

        # SVD分解和旋转矩阵计算
        T = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        U, S, V = self.svd22([A[0][0], A[0][1], A[1][0], A[1][1]])
        T[0][0] = U[0] * V[0] + U[1] * V[2]
        T[0][1] = U[0] * V[1] + U[1] * V[3]
        T[1][0] = U[2] * V[0] + U[3] * V[2]
        T[1][1] = U[2] * V[1] + U[3] * V[3]

        # 计算缩放因子
        scale = 1.0
        src_demean_mean = [0.0, 0.0]
        src_demean_var = [0.0, 0.0]
        for i in range(SRC_NUM):
            src_demean_mean[0] += src_demean[i][0]
            src_demean_mean[1] += src_demean[i][1]

        src_demean_mean[0] /= SRC_NUM
        src_demean_mean[1] /= SRC_NUM

        for i in range(SRC_NUM):
            src_demean_var[0] += (src_demean_mean[0] - src_demean[i][0]) ** 2
            src_demean_var[1] += (src_demean_mean[1] - src_demean[i][1]) ** 2

        src_demean_var[0] /= SRC_NUM
        src_demean_var[1] /= SRC_NUM
        scale = 1.0 / (src_demean_var[0] + src_demean_var[1]) * (S[0] + S[1])

        # 计算平移向量
        T[0][2] = dst_mean[0] - scale * (T[0][0] * src_mean[0] + T[0][1] * src_mean[1])
        T[1][2] = dst_mean[1] - scale * (T[1][0] * src_mean[0] + T[1][1] * src_mean[1])

        # 应用缩放
        T[0][0] *= scale
        T[0][1] *= scale
        T[1][0] *= scale
        T[1][1] *= scale

        return T

    def get_affine_matrix(self, sparse_points):
        """获取仿射变换矩阵"""
        with ScopedTiming("get_affine_matrix", self.debug_mode > 1):
            matrix_dst = self.image_umeyama_112(sparse_points)
            matrix_dst = [matrix_dst[0][0], matrix_dst[0][1], matrix_dst[0][2],
                         matrix_dst[1][0], matrix_dst[1][1], matrix_dst[1][2]]
            return matrix_dst
class FaceRecognition:
    """人脸识别主类 / Face recognition main class"""
    
    def __init__(self, face_det_kmodel, face_reg_kmodel, det_input_size, reg_input_size,
                 database_dir, anchors, confidence_threshold=0.25, nms_threshold=0.3,
                 face_recognition_threshold=0.75, rgb888p_size=[1280,720],
                 display_size=[640,360], debug_mode=0):
        
        self.logger = get_logger()
        self.config = ConfigLoader()
        
        # 初始化参数
        self.face_det_kmodel = face_det_kmodel
        self.face_reg_kmodel = face_reg_kmodel
        self.det_input_size = det_input_size
        self.reg_input_size = reg_input_size
        self.database_dir = database_dir
        self.anchors = anchors
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.face_recognition_threshold = face_recognition_threshold
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode
        
        # 图像捕获相关
        self.capture_dir = "/data/captures/"
        self.capture_enabled = self.config.get('face_recognition.capture_enabled', True)
        self.capture_max_files = self.config.get('face_recognition.capture_max_files', 100)
        self.last_captured_image = None  # 保存最后捕获的图像路径
        self.last_captured_data = None  # 保存最后捕获的图像数据

        # 数据库参数
        self.max_register_face = 100
        self.feature_num = 128
        self.valid_register_face = 0
        self.db_name = []
        self.db_data = []
        
        # 识别窗口和真空期控制
        self.recognition_window = 5  # 5秒识别窗口
        self.vacuum_period = 5  # 5秒真空期
        
        # 状态控制
        self.window_start_time = 0  # 窗口开始时间
        self.in_vacuum = False  # 是否在真空期
        self.vacuum_start_time = 0  # 真空期开始时间
        
        # 窗口内统计
        self.window_has_success = False  # 窗口内是否有成功识别
        self.window_face_count = 0  # 窗口内检测到的人脸数
        self.window_unknown_count = 0  # 窗口内未知人脸数
        self.last_recognized_name = None  # 最后识别的人名
        self.last_recognized_score = 0  # 最后识别的分数

        # 初始化UART通信
        try:
            self.uart = YbUart(baudrate=115200)
            self.pto = YbProtocol()
        except:
            self.logger.warning("UART初始化失败")
            self.uart = None
            self.pto = None

        # 初始化检测和注册模型
        self.face_det = FaceDetApp(self.face_det_kmodel,
                                 model_input_size=self.det_input_size,
                                 anchors=self.anchors,
                                 confidence_threshold=self.confidence_threshold,
                                 nms_threshold=self.nms_threshold,
                                 rgb888p_size=self.rgb888p_size,
                                 display_size=self.display_size,
                                 debug_mode=0)

        self.face_reg = FaceRegistrationApp(self.face_reg_kmodel,
                                          model_input_size=self.reg_input_size,
                                          rgb888p_size=self.rgb888p_size,
                                          display_size=self.display_size)

        self.face_det.config_preprocess()
        self.database_init()

    def run(self, input_np):
        """运行人脸识别 - 5秒识别窗口机制"""
        current_time = time.ticks_ms() / 1000.0  # 转换为秒
        
        # 保存当前帧供后续捕获使用
        self.current_frame = input_np
        
        # 检查是否在真空期
        if self.in_vacuum:
            time_since_vacuum = current_time - self.vacuum_start_time
            if time_since_vacuum < self.vacuum_period:
                # 仍在真空期
                remaining = self.vacuum_period - time_since_vacuum
                self.logger.debug(f"真空期中，剩余 {remaining:.1f}秒")
                return [], ['in_vacuum']
            else:
                # 真空期结束，重置状态，开始新窗口
                self.logger.info("真空期结束，开始新的识别窗口")
                self.in_vacuum = False
                self.reset_window_stats()
                self.window_start_time = current_time
        
        # 检查是否需要开始新窗口
        if self.window_start_time == 0:
            self.window_start_time = current_time
            self.reset_window_stats()
            self.logger.debug("开始新的识别窗口")
        
        # 执行人脸检测
        det_boxes, landms = self.face_det.run(input_np)
        recg_res = []
        
        # 如果没有检测到人脸
        if not det_boxes:
            # 检查窗口是否超时（5秒）
            window_elapsed = current_time - self.window_start_time
            if window_elapsed >= self.recognition_window:
                if self.window_face_count == 0:
                    # 5秒内都没有检测到人脸，重置窗口继续等待
                    self.reset_window_stats()
                    self.window_start_time = current_time
                elif not self.window_has_success and self.window_unknown_count > 0:
                    # 5秒内只检测到未知人脸，触发失败处理
                    self.logger.info(f"识别窗口结束：5秒内检测到{self.window_unknown_count}次未知人脸")
                    recg_res = ['trigger_failed']
                    # 进入真空期
                    self.enter_vacuum_period()
                else:
                    # 重置窗口
                    self.reset_window_stats()
                    self.window_start_time = current_time
            return det_boxes, recg_res
        
        # 更新窗口统计
        self.window_face_count += len(det_boxes)
        
        # 执行人脸识别
        for landm in landms:
            self.face_reg.config_preprocess(landm)
            feature = self.face_reg.run(input_np)
            res = self.database_search(feature)
            recg_res.append(res)
        
        # 检查识别结果
        recognized_person, score = self.get_recognized_person(recg_res)
        
        if recognized_person:
            # 识别成功，立即触发成功处理
            if not self.window_has_success:
                self.logger.info(f"窗口内首次识别成功: {recognized_person}, 分数: {score:.2f}")
                self.window_has_success = True
                self.last_recognized_name = recognized_person
                self.last_recognized_score = score
                
                # 捕获图像
                self.capture_face_image(recognized_person, score, True, det_boxes)
                
                # 标记需要触发成功处理
                recg_res.append('trigger_success')
                
                # 进入真空期
                self.enter_vacuum_period()
            else:
                # 窗口内已经有成功，等待真空期
                recg_res = ['already_success']
        else:
            # 识别失败（unknown）
            self.window_unknown_count += 1
            
            # 检查窗口是否超时
            window_elapsed = current_time - self.window_start_time
            if window_elapsed >= self.recognition_window and not self.window_has_success:
                # 5秒窗口结束且全部失败，触发失败处理
                self.logger.info(f"识别窗口结束：5秒内全部失败({self.window_unknown_count}次)")
                
                # 捕获图像
                self.capture_face_image("unknown", 0, False, det_boxes)
                
                recg_res.append('trigger_failed')
                
                # 进入真空期
                self.enter_vacuum_period()
        
        return det_boxes, recg_res
    
    def reset_window_stats(self):
        """重置窗口统计"""
        self.window_has_success = False
        self.window_face_count = 0
        self.window_unknown_count = 0
        self.last_recognized_name = None
        self.last_recognized_score = 0
    
    def enter_vacuum_period(self):
        """进入真空期"""
        current_time = time.ticks_ms() / 1000.0
        self.in_vacuum = True
        self.vacuum_start_time = current_time
        self.logger.info(f"进入{self.vacuum_period}秒真空期")
    def database_init(self):
        """初始化人脸数据库"""
        with ScopedTiming("database_init", self.debug_mode > 1):
            # 自动创建数据库目录
            self._ensure_directory(self.database_dir)
        
            try:
                db_file_list = os.listdir(self.database_dir)
            
                for db_file in db_file_list:
                    if not db_file.endswith('.bin'):
                        continue
                    if self.valid_register_face >= self.max_register_face:
                        break
                
                    # 拼接路径
                    if self.database_dir.endswith('/'):
                        full_db_file = self.database_dir + db_file
                    else:
                        full_db_file = self.database_dir + '/' + db_file
                
                    try:
                        with open(full_db_file, 'rb') as f:
                            data = f.read()
                    
                        feature = np.frombuffer(data, dtype=np.float32)
                        self.db_data.append(feature)
                    
                        name = db_file.split('.')[0]
                        self.db_name.append(name)
                        self.valid_register_face += 1
                    except Exception as e:
                        self.logger.warning(f"加载文件失败 {db_file}: {e}")
                        continue
            
                self.logger.info(f"加载了 {self.valid_register_face} 个人脸特征")
                
            except Exception as e:
                self.logger.error(f"数据库初始化失败: {e}")

    def _ensure_directory(self, path):
        """确保目录存在，不存在则创建"""
        try:
            os.stat(path)
            return True
        except OSError:
            self.logger.info(f"目录不存在，正在创建: {path}")
        
        # 递归创建目录
        path = path.rstrip('/')
        parts = path.split('/')
        current = ''
    
        for i, part in enumerate(parts):
            if i == 0 and not part:  # Unix根目录
                current = '/'
                continue
        
            current = current + '/' + part if current and current != '/' else (current + part if current else part)
        
            try:
                os.stat(current)
            except OSError:
                try:
                    os.mkdir(current)
                    self.logger.debug(f"创建: {current}")
                except Exception as e:
                    self.logger.error(f"创建目录失败 {current}: {e}")
                    raise Exception(f"无法创建目录: {path}")
    
        self.logger.info(f"目录创建成功: {path}")
        return True

    def database_reset(self):
        """重置数据库"""
        with ScopedTiming("database_reset", self.debug_mode > 1):
            self.logger.info("正在清空数据库...")
            self.db_name = []
            self.db_data = []
            self.valid_register_face = 0
            self.logger.info("数据库已清空")

    def database_search(self, feature):
        """在数据库中搜索匹配的人脸"""
        with ScopedTiming("database_search", self.debug_mode > 1):
            v_id = -1
            v_score_max = 0.0

            # 特征归一化
            feature /= np.linalg.norm(feature)

            # 遍历数据库进行匹配
            for i in range(self.valid_register_face):
                db_feature = self.db_data[i]
                db_feature /= np.linalg.norm(db_feature)
                v_score = np.dot(feature, db_feature)/2 + 0.5

                if v_score > v_score_max:
                    v_score_max = v_score
                    v_id = i

            # 返回识别结果
            if v_id == -1:
                return 'unknown'
            elif v_score_max < self.face_recognition_threshold:
                return 'unknown'
            else:
                result = 'name: {}, score: {}'.format(self.db_name[v_id], v_score_max)
                return result

    def draw_result(self, pl, dets, recg_results):
        """绘制识别结果"""
        pl.osd_img.clear()
        if dets:
            for i, det in enumerate(dets):
                x1, y1, w, h = map(lambda x: int(round(x, 0)), det[:4])
                x1 = x1 * self.display_size[0]//self.rgb888p_size[0]
                y1 = y1 * self.display_size[1]//self.rgb888p_size[1]
                w = w * self.display_size[0]//self.rgb888p_size[0]
                h = h * self.display_size[1]//self.rgb888p_size[1]

                recg_text = recg_results[i]
                if recg_text == 'unknown':
                    pl.osd_img.draw_rectangle(x1, y1, w, h, color=(255,0,0,255), thickness=4)
                else:
                    pl.osd_img.draw_rectangle(x1, y1, w, h, color=(255,0,255,0), thickness=4)
                pl.osd_img.draw_string_advanced(x1, y1, 32, recg_text, color=(255,255,0,0))

                # 解析识别结果并发送UART
                if self.uart and self.pto:
                    pattern = r'name: (.*), score: (.*)'
                    match = re.match(pattern, recg_text)
                    
                    if match:
                        name_value = match.group(1)
                        score_value = match.group(2)
                        pto_data = self.pto.get_face_recoginiton_data(x1, y1, w, h, name_value, score_value)
                    else:
                        pto_data = self.pto.get_face_recoginiton_data(x1, y1, w, h, recg_text, 0)
                    
                    self.uart.send(pto_data)
                    self.logger.debug(f"UART发送: {pto_data}")

    def get_recognized_person(self, recg_results):
        """获取识别到的人员信息"""
        if not recg_results:
            return None, 0.0
            
        for result in recg_results:
            if result != 'unknown':
                pattern = r'name: (.*), score: (.*)'
                match = re.match(pattern, result)
                if match:
                    name = match.group(1)
                    score = float(match.group(2))
                    return name, score
        
        return None, 0.0
    
    def is_in_vacuum_period(self):
        """检查是否在真空期"""
        return self.in_vacuum
    
    def get_vacuum_remaining_time(self):
        """获取真空期剩余时间"""
        if not self.in_vacuum:
            return 0
        
        current_time = time.ticks_ms() / 1000.0
        elapsed = current_time - self.vacuum_start_time
        remaining = max(0, self.vacuum_period - elapsed)
        return remaining
    
    def get_window_elapsed_time(self):
        """获取当前窗口已用时间"""
        if self.window_start_time == 0:
            return 0
        
        current_time = time.ticks_ms() / 1000.0
        elapsed = current_time - self.window_start_time
        return elapsed
    
    def reset_vacuum_period(self):
        """重置真空期状态"""
        self.in_vacuum = False
        self.vacuum_start_time = 0
        self.window_start_time = 0
        self.reset_window_stats()
        self.logger.info("识别状态已完全重置")
    
    def should_process_result(self, recg_results):
        """检查是否应该处理识别结果（用于触发音频和日志）"""
        if not recg_results:
            return False
        
        # 检查特殊标记
        for res in recg_results:
            if res == 'trigger_success':
                return True  # 需要处理成功
            elif res == 'trigger_failed':
                return True  # 需要处理失败
            elif res in ['in_vacuum', 'already_success']:
                return False  # 不需要处理
        
        return False  # 默认不处理
    
    def get_trigger_type(self, recg_results):
        """获取触发类型"""
        if not recg_results:
            return None
        
        for res in recg_results:
            if res == 'trigger_success':
                return 'success'
            elif res == 'trigger_failed':
                return 'failed'
        
        return None

    def capture_face_image(self, person_name, score, is_success, det_boxes):
        """捕获并保存人脸图像
        
        Args:
            person_name: 识别的人名或"unknown"
            score: 识别分数
            is_success: 是否识别成功
            det_boxes: 检测到的人脸框
        """
        if not self.capture_enabled:
            return
        
        try:
            # 确保捕获目录存在
            self._ensure_directory(self.capture_dir)
            
            # 生成文件名
            timestamp = int(time.ticks_ms())
            status = "granted" if is_success else "denied"
            safe_name = person_name.replace(" ", "_").replace("/", "_")
            
            # 根据日期创建子目录
            import utime
            date = utime.localtime()
            date_dir = f"{self.capture_dir}{date[0]:04d}{date[1]:02d}{date[2]:02d}/"
            self._ensure_directory(date_dir)
            
            # 完整文件路径
            filename = f"{status}_{safe_name}_{timestamp}.jpg"
            filepath = date_dir + filename
            
            # 保存图像
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                # 将numpy数组转换为图像
                img = self._numpy_to_image(self.current_frame, det_boxes)
                
                if img:
                    # 保存为JPEG
                    img.compress(quality=85)
                    img.save(filepath)
                    
                    # 保存图像数据供上传使用
                    self.last_captured_image = filepath
                    self.last_captured_data = img.to_bytes()
                    
                    self.logger.info(f"捕获图像保存: {filepath}")
                    
                    # 清理旧图像
                    self._cleanup_old_captures(date_dir)
            
        except Exception as e:
            self.logger.error(f"捕获图像失败: {e}")
    
    def _numpy_to_image(self, np_array, det_boxes=None):
        """将numpy数组转换为图像对象
        
        Args:
            np_array: numpy数组格式的图像
            det_boxes: 可选的人脸框，用于绘制
        """
        try:
            # 尝试直接使用当前帧作为Image对象
            # 因为pipeline.get_frame()可能返回的是Image对象而不是numpy数组
            if hasattr(np_array, 'compress'):
                # 已经是Image对象
                img = np_array
            else:
                # 创建新的图像对象
                img = image.Image(self.rgb888p_size[0], self.rgb888p_size[1], image.RGB888)
            
            # 如果有人脸框，可以在图像上绘制
            if det_boxes and len(det_boxes) > 0:
                for det in det_boxes:
                    x1, y1, w_box, h_box = map(int, det[:4])
                    # 绘制矩形框
                    img.draw_rectangle(x1, y1, w_box, h_box, 
                                      color=(0, 255, 0) if self.last_recognized_name else (255, 0, 0),
                                      thickness=2)
                    
                    # 添加文字标签
                    if self.last_recognized_name:
                        text = f"{self.last_recognized_name} ({self.last_recognized_score:.2f})"
                    else:
                        text = "Unknown"
                    img.draw_string(x1, y1 - 10, text, color=(255, 255, 255))
            
            return img
            
        except Exception as e:
            self.logger.error(f"转换图像失败: {e}")
            # 尝试直接保存原始数据
            try:
                # 创建默认图像
                img = image.Image(self.rgb888p_size[0], self.rgb888p_size[1], image.RGB888)
                return img
            except:
                return None
    
    def _cleanup_old_captures(self, directory):
        """清理旧的捕获图像
        
        Args:
            directory: 要清理的目录
        """
        try:
            files = os.listdir(directory)
            
            # 如果文件数超过限制
            if len(files) > self.capture_max_files:
                # 获取文件信息并排序
                file_info = []
                for f in files:
                    if f.endswith('.jpg'):
                        filepath = directory + f
                        try:
                            stat = os.stat(filepath)
                            file_info.append((filepath, stat[8]))  # stat[8]是修改时间
                        except:
                            continue
                
                # 按时间排序，删除最旧的文件
                file_info.sort(key=lambda x: x[1])
                
                # 删除超出数量的文件
                files_to_delete = len(file_info) - self.capture_max_files + 10  # 留10个缓冲
                for i in range(files_to_delete):
                    try:
                        os.remove(file_info[i][0])
                        self.logger.debug(f"删除旧图像: {file_info[i][0]}")
                    except:
                        continue
                        
        except Exception as e:
            self.logger.debug(f"清理旧图像失败: {e}")
    
    def get_last_captured_image(self):
        """获取最后捕获的图像信息
        
        Returns:
            tuple: (图像路径, 图像数据) 或 (None, None)
        """
        return self.last_captured_image, self.last_captured_data
    
    def clear_captured_image(self):
        """清除缓存的捕获图像"""
        self.last_captured_image = None
        self.last_captured_data = None
    
    def deinit(self):
        """释放资源"""
        try:
            self.face_det.deinit()
            self.face_reg.deinit()
            self.logger.info("人脸识别模块已释放")
        except Exception as e:
            self.logger.error(f"释放人脸识别模块失败: {e}")