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

        # 数据库参数
        self.max_register_face = 100
        self.feature_num = 128
        self.valid_register_face = 0
        self.db_name = []
        self.db_data = []

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
        """运行人脸识别"""
        det_boxes, landms = self.face_det.run(input_np)
        recg_res = []

        for landm in landms:
            self.face_reg.config_preprocess(landm)
            feature = self.face_reg.run(input_np)
            res = self.database_search(feature)
            recg_res.append(res)

        return det_boxes, recg_res

    def database_init(self):
        """初始化人脸数据库"""
        with ScopedTiming("database_init", self.debug_mode > 1):
            try:
                os.stat(self.database_dir)
                db_file_list = os.listdir(self.database_dir)
                for db_file in db_file_list:
                    if not db_file.endswith('.bin'):
                        continue
                    if self.valid_register_face >= self.max_register_face:
                        break
                        
                    full_db_file = self.database_dir + db_file
                    
                    with open(full_db_file, 'rb') as f:
                        data = f.read()
                    feature = np.frombuffer(data, dtype=np.float)
                    self.db_data.append(feature)
                    
                    name = db_file.split('.')[0]
                    self.db_name.append(name)
                    self.valid_register_face += 1
                    
                self.logger.info(f"加载了 {self.valid_register_face} 个人脸特征")
                    
            except Exception as e:
                self.logger.error(f"数据库初始化失败: {e}")

            except OSError:  
                self.logger.warning(f"数据库目录不存在: {self.database_dir}")  
                return

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

    def deinit(self):
        """释放资源"""
        try:
            self.face_det.deinit()
            self.face_reg.deinit()
            self.logger.info("人脸识别模块已释放")
        except Exception as e:
            self.logger.error(f"释放人脸识别模块失败: {e}")