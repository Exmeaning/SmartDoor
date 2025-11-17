## 这是一个基于K230开发的提示词文件 该系统基于microPython 下面是你需要做的部分

### 1.人脸识别模块 通过人脸识别调用相关API

- 以下是提供的示例代码
```python
# 导入所需库 / Import required libraries
from libs.PipeLine import PipeLine, ScopedTiming  # 导入视频处理Pipeline和计时器类 / Import video pipeline and timer classes
from libs.AIBase import AIBase                    # 导入AI基础类 / Import AI base class
from libs.AI2D import Ai2d                       # 导入AI 2D处理类 / Import AI 2D processing class
import os
import ujson
from media.media import *                        # 导入媒体处理相关库 / Import media processing libraries
from time import *
import nncase_runtime as nn                      # 导入神经网络运行时库 / Import neural network runtime library
import ulab.numpy as np                          # 导入类numpy库，用于数组操作 / Import numpy-like library for array operations
import time
import image                                     # 图像处理库 / Image processing library
import aidemo                                    # AI演示库 / AI demo library
import random
import gc                                        # 垃圾回收模块 / Garbage collection module
import sys
import math,re

# 全局变量定义 / Global variable definition
fr = None                                        # 人脸识别对象的全局变量 / Global variable for face recognition object
from libs.YbProtocol import YbProtocol
from ybUtils.YbUart import YbUart
# uart = None
uart = YbUart(baudrate=115200)
pto = YbProtocol()

class FaceDetApp(AIBase):
    """
    人脸检测应用类 / Face detection application class
    继承自AIBase基类 / Inherits from AIBase class
    """
    def __init__(self, kmodel_path, model_input_size, anchors, confidence_threshold=0.25,
                 nms_threshold=0.3, rgb888p_size=[640,480], display_size=[640,480], debug_mode=0):
        """
        初始化函数 / Initialization function
        参数说明 / Parameters:
        kmodel_path: 模型文件路径 / Model file path
        model_input_size: 模型输入尺寸 / Model input size
        anchors: 锚框参数 / Anchor box parameters
        confidence_threshold: 置信度阈值 / Confidence threshold
        nms_threshold: 非极大值抑制阈值 / Non-maximum suppression threshold
        rgb888p_size: 输入图像尺寸 / Input image size
        display_size: 显示尺寸 / Display size
        debug_mode: 调试模式标志 / Debug mode flag
        """
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        # 保存初始化参数 / Save initialization parameters
        self.kmodel_path = kmodel_path                  # kmodel文件路径 / kmodel file path
        self.model_input_size = model_input_size        # 模型输入尺寸 / Model input size
        self.confidence_threshold = confidence_threshold # 置信度阈值 / Confidence threshold
        self.nms_threshold = nms_threshold              # NMS阈值 / NMS threshold
        self.anchors = anchors                          # 锚框参数 / Anchor parameters

        # 图像尺寸处理（16字节对齐）/ Image size processing (16-byte alignment)
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]

        self.debug_mode = debug_mode                    # 调试模式 / Debug mode

        # 初始化AI2D预处理器 / Initialize AI2D preprocessor
        self.ai2d = Ai2d(debug_mode)
        # 设置AI2D参数 / Set AI2D parameters
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                nn.ai2d_format.NCHW_FMT,
                                np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        """
        配置图像预处理参数 / Configure image preprocessing parameters
        使用pad和resize操作 / Use pad and resize operations
        """
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            # 设置输入大小 / Set input size
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size

            # 配置padding参数 / Configure padding parameters
            self.ai2d.pad(self.get_pad_param(), 0, [104,117,123])
            # 配置resize参数 / Configure resize parameters
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            # 构建预处理pipeline / Build preprocessing pipeline
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],
                          [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        """
        后处理方法 / Post-processing method
        使用aidemo库处理检测结果 / Process detection results using aidemo library
        """
        with ScopedTiming("postprocess", self.debug_mode > 0):
            # 处理检测结果 / Process detection results
            res = aidemo.face_det_post_process(self.confidence_threshold,
                                             self.nms_threshold,
                                             self.model_input_size[0],
                                             self.anchors,
                                             self.rgb888p_size,
                                             results)
            # 返回检测结果 / Return detection results
            if len(res) == 0:
                return res, res
            else:
                return res[0], res[1]

    def get_pad_param(self):
        """
        计算padding参数 / Calculate padding parameters
        返回padding的边界值 / Return padding boundary values
        """
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]

        # 计算缩放比例 / Calculate scaling ratio
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        ratio = min(ratio_w, ratio_h)

        # 计算新的尺寸 / Calculate new dimensions
        new_w = int(ratio * self.rgb888p_size[0])
        new_h = int(ratio * self.rgb888p_size[1])

        # 计算padding值 / Calculate padding values
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2

        # 返回padding参数 / Return padding parameters
        top = int(round(0))
        bottom = int(round(dh * 2 + 0.1))
        left = int(round(0))
        right = int(round(dw * 2 - 0.1))
        return [0, 0, 0, 0, top, bottom, left, right]

class FaceRegistrationApp(AIBase):
    """
    人脸注册应用类 / Face registration application class
    用于人脸特征提取和注册 / For face feature extraction and registration
    """
    def __init__(self, kmodel_path, model_input_size, rgb888p_size=[640,360],
                 display_size=[640,360], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        # 初始化参数 / Initialize parameters
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode

        # 标准人脸关键点坐标 / Standard face keypoint coordinates
        self.umeyama_args_112 = [
            38.2946, 51.6963,
            73.5318, 51.5014,
            56.0252, 71.7366,
            41.5493, 92.3655,
            70.7299, 92.2041
        ]

        # 初始化AI2D / Initialize AI2D
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                nn.ai2d_format.NCHW_FMT,
                                np.uint8, np.uint8)

    def config_preprocess(self, landm, input_image_size=None):
        """
        配置预处理参数 / Configure preprocessing parameters
        使用仿射变换进行人脸对齐 / Use affine transformation for face alignment
        """
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size

            # 计算仿射变换矩阵 / Calculate affine transformation matrix
            affine_matrix = self.get_affine_matrix(landm)
            self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, affine_matrix)

            # 构建预处理pipeline / Build preprocessing pipeline
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],
                          [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        """
        后处理方法 / Post-processing method
        提取人脸特征 / Extract face features
        """
        with ScopedTiming("postprocess", self.debug_mode > 0):
            return results[0][0]

    def svd22(self, a):
        """
        2x2矩阵的奇异值分解 / Singular value decomposition for 2x2 matrix
        """
        # SVD计算 / SVD calculation
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
        """
        使用Umeyama算法进行人脸对齐 / Face alignment using Umeyama algorithm
        """
        SRC_NUM = 5  # 关键点数量 / Number of keypoints
        SRC_DIM = 2  # 坐标维度 / Coordinate dimensions

        # 计算源点和目标点的均值 / Calculate mean of source and target points
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

        # 去均值化 / De-mean
        src_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        dst_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        for i in range(SRC_NUM):
            src_demean[i][0] = src[2 * i] - src_mean[0]
            src_demean[i][1] = src[2 * i + 1] - src_mean[1]
            dst_demean[i][0] = self.umeyama_args_112[2 * i] - dst_mean[0]
            dst_demean[i][1] = self.umeyama_args_112[2 * i + 1] - dst_mean[1]

        # 计算A矩阵 / Calculate A matrix
        A = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(SRC_DIM):
            for k in range(SRC_DIM):
                for j in range(SRC_NUM):
                    A[i][k] += dst_demean[j][i] * src_demean[j][k]
                A[i][k] /= SRC_NUM

        # SVD分解和旋转矩阵计算 / SVD decomposition and rotation matrix calculation
        T = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        U, S, V = self.svd22([A[0][0], A[0][1], A[1][0], A[1][1]])
        T[0][0] = U[0] * V[0] + U[1] * V[2]
        T[0][1] = U[0] * V[1] + U[1] * V[3]
        T[1][0] = U[2] * V[0] + U[3] * V[2]
        T[1][1] = U[2] * V[1] + U[3] * V[3]

        # 计算缩放因子 / Calculate scaling factor
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

        # 计算平移向量 / Calculate translation vector
        T[0][2] = dst_mean[0] - scale * (T[0][0] * src_mean[0] + T[0][1] * src_mean[1])
        T[1][2] = dst_mean[1] - scale * (T[1][0] * src_mean[0] + T[1][1] * src_mean[1])

        # 应用缩放 / Apply scaling
        T[0][0] *= scale
        T[0][1] *= scale
        T[1][0] *= scale
        T[1][1] *= scale

        return T

    def get_affine_matrix(self, sparse_points):
        """
        获取仿射变换矩阵 / Get affine transformation matrix
        """
        with ScopedTiming("get_affine_matrix", self.debug_mode > 1):
            matrix_dst = self.image_umeyama_112(sparse_points)
            matrix_dst = [matrix_dst[0][0], matrix_dst[0][1], matrix_dst[0][2],
                         matrix_dst[1][0], matrix_dst[1][1], matrix_dst[1][2]]
            return matrix_dst

class FaceRecognition:
    """
    人脸识别类 / Face recognition class
    集成了检测和识别功能 / Integrates detection and recognition functions
    """
    def __init__(self, face_det_kmodel, face_reg_kmodel, det_input_size, reg_input_size,
                 database_dir, anchors, confidence_threshold=0.25, nms_threshold=0.3,
                 face_recognition_threshold=0.75, rgb888p_size=[1280,720],
                 display_size=[640,360], debug_mode=0):

        # 初始化参数 / Initialize parameters
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

        # 数据库参数 / Database parameters
        self.max_register_face = 100
        self.feature_num = 128
        self.valid_register_face = 0
        self.db_name = []
        self.db_data = []

        # 初始化检测和注册模型 / Initialize detection and registration models
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
        """
        运行人脸识别 / Run face recognition
        """
        # 人脸检测 / Face detection
        det_boxes, landms = self.face_det.run(input_np)
        recg_res = []

        # 对每个检测到的人脸进行识别 / Recognize each detected face
        for landm in landms:
            self.face_reg.config_preprocess(landm)
            feature = self.face_reg.run(input_np)
            res = self.database_search(feature)
            recg_res.append(res)

        return det_boxes, recg_res

    def database_init(self):
        """
        初始化人脸数据库 / Initialize face database
        """
        with ScopedTiming("database_init", self.debug_mode > 1):
            try:
                # 读取数据库文件 / Read database files
                db_file_list = os.listdir(self.database_dir)
                for db_file in db_file_list:
                    if not db_file.endswith('.bin'):
                        continue
                    if self.valid_register_face >= self.max_register_face:
                        break
                        
                    valid_index = self.valid_register_face
                    full_db_file = self.database_dir + db_file
                    
                    # 读取特征数据 / Read feature data
                    with open(full_db_file, 'rb') as f:
                        data = f.read()
                    feature = np.frombuffer(data, dtype=np.float)
                    self.db_data.append(feature)
                    
                    # 保存人名 / Save person name
                    name = db_file.split('.')[0]
                    self.db_name.append(name)
                    self.valid_register_face += 1
            except Exception as e:
                print(e)
                print("未检测到人脸数据库，请先按照教程步骤，注册人脸信息")
                print("No face database detected, please follow the tutorial steps to register the face information")
            # 读取数据库文件 / Read database files
            db_file_list = os.listdir(self.database_dir)
            for db_file in db_file_list:
                if not db_file.endswith('.bin'):
                    continue
                if self.valid_register_face >= self.max_register_face:
                    break

                valid_index = self.valid_register_face
                full_db_file = self.database_dir + db_file

                # 读取特征数据 / Read feature data
                with open(full_db_file, 'rb') as f:
                    data = f.read()
                feature = np.frombuffer(data, dtype=np.float)
                self.db_data.append(feature)

                # 保存人名 / Save person name
                name = db_file.split('.')[0]
                self.db_name.append(name)
                self.valid_register_face += 1

    def database_reset(self):
        """
        重置数据库 / Reset database
        """
        with ScopedTiming("database_reset", self.debug_mode > 1):
            print("database clearing...")
            self.db_name = []
            self.db_data = []
            self.valid_register_face = 0
            print("database clear Done!")

    def database_search(self, feature):
        """
        在数据库中搜索匹配的人脸 / Search for matching face in database
        """
        with ScopedTiming("database_search", self.debug_mode > 1):
            v_id = -1
            v_score_max = 0.0

            # 特征归一化 / Feature normalization
            feature /= np.linalg.norm(feature)

            # 遍历数据库进行匹配 / Search through database for matches
            for i in range(self.valid_register_face):
                db_feature = self.db_data[i]
                db_feature /= np.linalg.norm(db_feature)
                v_score = np.dot(feature, db_feature)/2 + 0.5

                if v_score > v_score_max:
                    v_score_max = v_score
                    v_id = i

            # 返回识别结果 / Return recognition result
            if v_id == -1:
                return 'unknown'
            elif v_score_max < self.face_recognition_threshold:
                return 'unknown'
            else:
                result = 'name: {}, score: {}'.format(self.db_name[v_id], v_score_max)
                return result

    def draw_result(self, pl, dets, recg_results):
        """
        绘制识别结果 / Draw recognition results
        """
        pl.osd_img.clear()
        if dets:
            for i, det in enumerate(dets):
                # 绘制人脸框 / Draw face box
                x1, y1, w, h = map(lambda x: int(round(x, 0)), det[:4])
                x1 = x1 * self.display_size[0]//self.rgb888p_size[0]
                y1 = y1 * self.display_size[1]//self.rgb888p_size[1]
                w = w * self.display_size[0]//self.rgb888p_size[0]
                h = h * self.display_size[1]//self.rgb888p_size[1]

                # 绘制识别结果 / Draw recognition result
                recg_text = recg_results[i]
                if recg_text == 'unknown':
                    pl.osd_img.draw_rectangle(x1, y1, w, h, color=(255,0,0,255), thickness=4)
                else:
                    pl.osd_img.draw_rectangle(x1, y1, w, h, color=(255,0,255,0), thickness=4)
                pl.osd_img.draw_string_advanced(x1, y1, 32, recg_text, color=(255,255,0,0))

                # 使用正则表达式匹配 name 和 score 的值
                pattern = r'name: (.*), score: (.*)'
                match = re.match(pattern, recg_text)

                if match:
                    name_value = match.group(1)  # 提取 name 的值
                    score_value = match.group(2)  # 提取 score 的值
                    pto_data = pto.get_face_recoginiton_data(x1, y1, w, h, name_value, score_value)
                    uart.send(pto_data)
                    print(pto_data)
                else:
                    pto_data = pto.get_face_recoginiton_data(x1, y1, w, h, recg_text, 0)
                    uart.send(pto_data)
                    print(pto_data)


def exce_demo(pl):
    """
    执行演示程序 / Execute demo program
    """
    global fr
    display_mode = pl.display_mode
    rgb888p_size = pl.rgb888p_size
    display_size = pl.display_size

    # 加载模型和配置 / Load models and configurations
    face_det_kmodel_path = "/sdcard/kmodel/face_detection_320.kmodel"
    face_reg_kmodel_path = "/sdcard/kmodel/face_recognition.kmodel"
    anchors_path = "/sdcard/utils/prior_data_320.bin"
    database_dir = "/data/face_database/2600271ef6d/"
    face_det_input_size = [320,320]
    face_reg_input_size = [112,112]
    confidence_threshold = 0.5
    nms_threshold = 0.2
    anchor_len = 4200
    det_dim = 4

    # 读取anchor数据 / Read anchor data
    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((anchor_len, det_dim))
    face_recognition_threshold = 0.65

    # 创建人脸识别对象 / Create face recognition object
    fr = FaceRecognition(face_det_kmodel_path, face_reg_kmodel_path,
                        det_input_size=face_det_input_size,
                        reg_input_size=face_reg_input_size,
                        database_dir=database_dir,
                        anchors=anchors,
                        confidence_threshold=confidence_threshold,
                        nms_threshold=nms_threshold,
                        face_recognition_threshold=face_recognition_threshold,
                        rgb888p_size=rgb888p_size,
                        display_size=display_size)

    # 主循环 / Main loop
    try:
        while True:
            with ScopedTiming("total", 1):
                # 获取图像并处理 / Get and process image
                img = pl.get_frame()
                det_boxes, recg_res = fr.run(img)
                fr.draw_result(pl, det_boxes, recg_res)
                pl.show_image()
                gc.collect()
    except Exception as e:
        print("人脸识别功能退出")
    finally:
        exit_demo()

def exit_demo():
    """
    退出程序 / Exit program
    """
    global fr
    fr.face_det.deinit()
    fr.face_reg.deinit()

if __name__ == "__main__":
    # 主程序入口 / Main program entry
    rgb888p_size=[640,480]
    display_size = [640,480]
    display_mode = "lcd"

    # 创建并启动视频处理Pipeline / Create and start video processing pipeline
    pl = PipeLine(rgb888p_size=rgb888p_size,
                 display_size=display_size,
                 display_mode=display_mode)
    pl.create()
    exce_demo(pl)

```

### 2.人脸检测模块 检测是否存在人脸 若不存在 则进入休眠，降低摄像头调用次数 控制发热

- 以下是提供的示例代码
```python
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
import utime
import image
import random
import gc
import sys
import aidemo
import _thread

face_det = None

from libs.YbProtocol import YbProtocol
from ybUtils.YbUart import YbUart
# uart = None
uart = YbUart(baudrate=115200)
pto = YbProtocol()

# 自定义人脸检测类，继承自AIBase基类
class FaceDetectionApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, anchors, confidence_threshold=0.5, nms_threshold=0.2, rgb888p_size=[224,224], display_size=[1920,1080], debug_mode=0):
        # 调用基类的构造函数 / Call parent class constructor
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        # 模型文件路径 / Path to the model file
        self.kmodel_path = kmodel_path

        # 模型输入分辨率 / Model input resolution
        self.model_input_size = model_input_size

        # 置信度阈值：检测结果的最小置信度要求 / Confidence threshold: minimum confidence requirement for detection results
        self.confidence_threshold = confidence_threshold

        # NMS阈值：非极大值抑制的阈值 / NMS threshold: threshold for Non-Maximum Suppression
        self.nms_threshold = nms_threshold

        # 锚点数据：用于目标检测的预定义框 / Anchor data: predefined boxes for object detection
        self.anchors = anchors

        # sensor给到AI的图像分辨率，宽度16对齐 / Image resolution from sensor to AI, width aligned to 16
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]

        # 显示分辨率，宽度16对齐 / Display resolution, width aligned to 16
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]

        # 调试模式标志 / Debug mode flag
        self.debug_mode = debug_mode

        # 实例化AI2D对象用于图像预处理 / Initialize AI2D object for image preprocessing
        self.ai2d = Ai2d(debug_mode)

        # 设置AI2D的输入输出格式 / Set AI2D input/output format
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            # 获取AI2D输入尺寸 / Get AI2D input size
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size

            # 获取padding参数 / Get padding parameters
            top, bottom, left, right = self.get_padding_param()

            # 设置padding: [上,下,左,右], 填充值[104,117,123] / Set padding: [top,bottom,left,right], padding value[104,117,123]
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [104, 117, 123])

            # 设置resize方法：双线性插值 / Set resize method: bilinear interpolation
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)

            # 构建预处理流程 / Build preprocessing pipeline
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],
                          [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            # 调用aidemo库进行人脸检测后处理 / Call aidemo library for face detection post-processing
            post_ret = aidemo.face_det_post_process(self.confidence_threshold,
                                                  self.nms_threshold,
                                                  self.model_input_size[1],
                                                  self.anchors,
                                                  self.rgb888p_size,
                                                  results)
            return post_ret[0] if post_ret else post_ret

    def draw_result(self, pl, dets):
        with ScopedTiming("display_draw", self.debug_mode > 0):
            if dets:
                # 清除上一帧的OSD绘制 / Clear previous frame's OSD drawing
                pl.osd_img.clear()

                for det in dets:
                    # 转换检测框坐标到显示分辨率 / Convert detection box coordinates to display resolution
                    x, y, w, h = map(lambda x: int(round(x, 0)), det[:4])
                    x = x * self.display_size[0] // self.rgb888p_size[0]
                    y = y * self.display_size[1] // self.rgb888p_size[1]
                    w = w * self.display_size[0] // self.rgb888p_size[0]
                    h = h * self.display_size[1] // self.rgb888p_size[1]

                    # 绘制黄色检测框 / Draw yellow detection box
                    pl.osd_img.draw_rectangle(x, y, w, h, color=(255, 255, 0, 255), thickness=2)

                    pto_data = pto.get_face_detect_data(x, y, w, h)
                    uart.send(pto_data)
                    print(pto_data)
            else:
                pl.osd_img.clear()

    def get_padding_param(self):
        # 计算模型输入和实际图像的缩放比例 / Calculate scaling ratio between model input and actual image
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        ratio = min(ratio_w, ratio_h)

        # 计算缩放后的新尺寸 / Calculate new dimensions after scaling
        new_w = int(ratio * self.rgb888p_size[0])
        new_h = int(ratio * self.rgb888p_size[1])

        # 计算padding值 / Calculate padding values
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2

        # 返回padding参数 / Return padding parameters
        return (int(round(0)),
                int(round(dh * 2 + 0.1)),
                int(round(0)),
                int(round(dw * 2 - 0.1)))

def exce_demo(pl):
    # 声明全局变量face_det / Declare global variable face_det
    global face_det

    # 获取显示相关参数 / Get display-related parameters
    display_mode = pl.display_mode      # 显示模式(如lcd) / Display mode (e.g., lcd)
    rgb888p_size = pl.rgb888p_size     # 原始图像分辨率 / Original image resolution
    display_size = pl.display_size      # 显示分辨率 / Display resolution

    # 设置人脸检测模型路径 / Set face detection model path
    kmodel_path = "/sdcard/kmodel/face_detection_320.kmodel"

    # 设置模型参数 / Set model parameters
    confidence_threshold = 0.5    # 置信度阈值 / Confidence threshold
    nms_threshold = 0.2          # 非极大值抑制阈值 / Non-maximum suppression threshold
    anchor_len = 4200            # 锚框数量 / Number of anchor boxes
    det_dim = 4                  # 检测维度(x,y,w,h) / Detection dimensions (x,y,w,h)

    # 加载锚框数据 / Load anchor box data
    anchors_path = "/sdcard/utils/prior_data_320.bin"
    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((anchor_len, det_dim))



    try:
        # 初始化人脸检测应用实例 / Initialize face detection application instance
        face_det = FaceDetectionApp(kmodel_path,
                                  model_input_size=[320, 320],
                                  anchors=anchors,
                                  confidence_threshold=confidence_threshold,
                                  nms_threshold=nms_threshold,
                                  rgb888p_size=rgb888p_size,
                                  display_size=display_size,
                                  debug_mode=0)

        # 配置图像预处理 / Configure image preprocessing
        face_det.config_preprocess()

        # 主循环 / Main loop
        while True:
            with ScopedTiming("total",0):    # 计时器 / Timer
                img = pl.get_frame()          # 获取摄像头帧图像 / Get camera frame
                res = face_det.run(img)       # 执行人脸检测 / Run face detection
                face_det.draw_result(pl, res) # 绘制检测结果 / Draw detection results
                pl.show_image()               # 显示处理后的图像 / Display processed image
                gc.collect()                  # 垃圾回收 / Garbage collection
                time.sleep_us(10)             # 短暂延时 / Brief delay

    except Exception as e:
        print("人脸检测功能退出")           # 异常退出提示 / Exception exit prompt
    finally:
        face_det.deinit()                   # 释放资源 / Release resources

def exit_demo():
    # 程序退出时释放资源 / Release resources when program exits
    global face_det
    face_det.deinit()

if __name__ == "__main__":
    # 设置图像和显示参数 / Set image and display parameters
    rgb888p_size=[640,480]    # 原始图像分辨率 / Original image resolution
    display_size=[640,480]      # 显示分辨率 / Display resolution
    display_mode="lcd"          # 显示模式 / Display mode

    # 初始化图像处理Pipline / Initialize image processing pipeline
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
    pl.create()  # 创建Pipline实例 / Create pipeline instance

    # 运行人脸检测demo / Run face detection demo
    exce_demo(pl)

```

### 3.网络模块 通过连接至WIFI网络 进行网络模块的通信 

- 连接成功 输出连接成功的音频
- 连接失败 输出未连接成功的音频 注明是离线模式

### 4.TCP通信模块 通过TCP客户端实现云服务器的通信

- 先简单实现 等相关内容齐全后我再给你详细提示词

### 5.人脸注册模块 通过命令实现人脸的注册

- 先简单实现
- 未来会有TCP相关命令来达成这个目标
- 示例代码如下

```python
# 导入必要的库文件 / Import necessary libraries
from libs.PipeLine import PipeLine, ScopedTiming  # 导入Pipeline和计时工具 / Import pipeline and timing tools
from libs.AIBase import AIBase     # 导入AI基础类 / Import AI base class
from libs.AI2D import Ai2d        # 导入AI 2D处理类 / Import AI 2D processing class
import os                         # 导入操作系统接口 / Import OS interface
import ujson                      # 导入JSON处理库 / Import JSON processing library
from media.media import *         # 导入媒体处理库 / Import media processing library
from time import *               # 导入时间处理库 / Import time processing library
import nncase_runtime as nn      # 导入神经网络运行时 / Import neural network runtime
import ulab.numpy as np          # 导入numpy库 / Import numpy library
import time                      # 导入时间库 / Import time library
import image                     # 导入图像处理库 / Import image processing library
import aidemo                    # 导入AI演示库 / Import AI demo library
import random                    # 导入随机数库 / Import random number library
import gc                        # 导入垃圾回收库 / Import garbage collection library
import sys                       # 导入系统库 / Import system library
import math                      # 导入数学库 / Import math library

global fr                        # 声明全局变量 / Declare global variable

class FaceDetApp(AIBase):
    """人脸检测应用类 / Face Detection Application Class

    这个类继承自AIBase，实现了人脸检测的功能
    This class inherits from AIBase and implements face detection functionality
    """

    def __init__(self, kmodel_path, model_input_size, anchors,
                 confidence_threshold=0.25, nms_threshold=0.3,
                 rgb888p_size=[1280,720], display_size=[1920,1080],
                 debug_mode=0):
        """初始化函数 / Initialization function

        参数 / Parameters:
        - kmodel_path: KPU模型的路径 / Path to KPU model
        - model_input_size: 模型输入尺寸 / Model input size
        - anchors: 锚框参数 / Anchor box parameters
        - confidence_threshold: 置信度阈值 / Confidence threshold
        - nms_threshold: NMS阈值 / NMS threshold
        - rgb888p_size: RGB888格式图像尺寸 / RGB888 format image size
        - display_size: 显示尺寸 / Display size
        - debug_mode: 调试模式 / Debug mode
        """
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        self.kmodel_path = kmodel_path                # KPU模型路径 / KPU model path
        self.model_input_size = model_input_size      # 模型输入尺寸 / Model input size
        self.confidence_threshold = confidence_threshold  # 置信度阈值 / Confidence threshold
        self.nms_threshold = nms_threshold            # NMS阈值 / NMS threshold
        self.anchors = anchors                        # 锚框参数 / Anchor box parameters

        # 设置RGB888图像尺寸，确保宽度16字节对齐 / Set RGB888 image size, ensure width is 16-byte aligned
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]

        # 设置显示尺寸，确保宽度16字节对齐 / Set display size, ensure width is 16-byte aligned
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]

        self.debug_mode = debug_mode                  # 调试模式 / Debug mode

        # 初始化AI2D对象，用于图像预处理 / Initialize AI2D object for image preprocessing
        self.ai2d = Ai2d(debug_mode)

        # 设置AI2D的数据类型和格式 / Set AI2D data type and format
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                nn.ai2d_format.NCHW_FMT,
                                np.uint8, np.uint8)
        self.image_size = []

    def config_preprocess(self, input_image_size=None):
        """配置预处理参数 / Configure preprocessing parameters

        对输入图像进行pad和resize等预处理操作
        Perform preprocessing operations such as pad and resize on input images
        """
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            # 设置输入图像尺寸 / Set input image size
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self.image_size = [input_image_size[1], input_image_size[0]]

            # 配置padding参数 / Configure padding parameters
            self.ai2d.pad(self.get_pad_param(ai2d_input_size), 0, [104,117,123])

            # 配置resize参数 / Configure resize parameters
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)

            # 构建预处理Pipeline / Build preprocessing pipeline
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],
                           [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        """后处理方法 / Post-processing method

        处理模型的原始输出，得到最终的检测结果
        Process the model's raw output to get final detection results
        """
        with ScopedTiming("postprocess", self.debug_mode > 0):
            # 调用aidemo库进行人脸检测后处理 / Call aidemo library for face detection post-processing
            res = aidemo.face_det_post_process(self.confidence_threshold,
                                             self.nms_threshold,
                                             self.model_input_size[0],
                                             self.anchors,
                                             self.image_size,
                                             results)
            if len(res) == 0:
                return res
            else:
                return res[0], res[1]

    def get_pad_param(self, image_input_size):
        """计算padding参数 / Calculate padding parameters

        计算等比例缩放后需要的padding参数
        Calculate the padding parameters needed after proportional scaling
        """
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]

        # 计算缩放比例 / Calculate scaling ratio
        ratio_w = dst_w / image_input_size[0]
        ratio_h = dst_h / image_input_size[1]
        ratio = min(ratio_w, ratio_h)

        # 计算新的尺寸 / Calculate new dimensions
        new_w = int(ratio * image_input_size[0])
        new_h = int(ratio * image_input_size[1])

        # 计算padding值 / Calculate padding values
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2

        top = int(round(0))
        bottom = int(round(dh * 2 + 0.1))
        left = int(round(0))
        right = int(round(dw * 2 - 0.1))

        return [0,0,0,0,top, bottom, left, right]

class FaceRegistrationApp(AIBase):
    """人脸注册应用类 / Face Registration Application Class

    处理人脸注册相关的功能
    Handle face registration related functions
    """

    def __init__(self, kmodel_path, model_input_size,
                 rgb888p_size=[1920,1080], display_size=[1920,1080],
                 debug_mode=0):
        """初始化函数 / Initialization function"""
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        self.kmodel_path = kmodel_path                # 模型路径 / Model path
        self.model_input_size = model_input_size      # 模型输入尺寸 / Model input size
        # RGB尺寸，确保16字节对齐 / RGB size, ensure 16-byte aligned
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        # 显示尺寸，确保16字节对齐 / Display size, ensure 16-byte aligned
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode                  # 调试模式 / Debug mode

        # 标准5个关键点坐标 / Standard 5 keypoint coordinates
        self.umeyama_args_112 = [
            38.2946 , 51.6963,
            73.5318 , 51.5014,
            56.0252 , 71.7366,
            41.5493 , 92.3655,
            70.7299 , 92.2041
        ]

        # 初始化AI2D / Initialize AI2D
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                nn.ai2d_format.NCHW_FMT,
                                np.uint8, np.uint8)

    def config_preprocess(self, landm, input_image_size=None):
        """配置预处理参数 / Configure preprocessing parameters"""
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size

            # 计算仿射变换矩阵并配置 / Calculate and configure affine transformation matrix
            affine_matrix = self.get_affine_matrix(landm)
            self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, affine_matrix)

            # 构建预处理Pipeline / Build preprocessing pipeline
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],
                           [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        """后处理方法 / Post-processing method"""
        with ScopedTiming("postprocess", self.debug_mode > 0):
            return results[0][0]

    def svd22(self, a):
        """2x2矩阵的奇异值分解 / Singular Value Decomposition for 2x2 matrix"""
        s = [0.0, 0.0]
        u = [0.0, 0.0, 0.0, 0.0]
        v = [0.0, 0.0, 0.0, 0.0]

        # 计算奇异值 / Calculate singular values
        s[0] = (math.sqrt((a[0] - a[3]) ** 2 + (a[1] + a[2]) ** 2) +
                math.sqrt((a[0] + a[3]) ** 2 + (a[1] - a[2]) ** 2)) / 2
        s[1] = abs(s[0] - math.sqrt((a[0] - a[3]) ** 2 + (a[1] + a[2]) ** 2))

        # 计算右奇异向量 / Calculate right singular vectors
        v[2] = math.sin((math.atan2(2 * (a[0] * a[1] + a[2] * a[3]),
                                   a[0] ** 2 - a[1] ** 2 + a[2] ** 2 - a[3] ** 2)) / 2) if s[0] > s[1] else 0
        v[0] = math.sqrt(1 - v[2] ** 2)
        v[1] = -v[2]
        v[3] = v[0]

        # 计算左奇异向量 / Calculate left singular vectors
        u[0] = -(a[0] * v[0] + a[1] * v[2]) / s[0] if s[0] != 0 else 1
        u[2] = -(a[2] * v[0] + a[3] * v[2]) / s[0] if s[0] != 0 else 0
        u[1] = (a[0] * v[1] + a[1] * v[3]) / s[1] if s[1] != 0 else -u[2]
        u[3] = (a[2] * v[1] + a[3] * v[3]) / s[1] if s[1] != 0 else u[0]

        v[0] = -v[0]
        v[2] = -v[2]

        return u, s, v

    def image_umeyama_112(self, src):
        """使用Umeyama算法计算仿射变换矩阵 / Calculate affine transformation matrix using Umeyama algorithm"""
        SRC_NUM = 5
        SRC_DIM = 2

        # 计算源点和目标点的均值 / Calculate mean of source and target points
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

        # 去中心化 / De-mean
        src_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        dst_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]

        for i in range(SRC_NUM):
            src_demean[i][0] = src[2 * i] - src_mean[0]
            src_demean[i][1] = src[2 * i + 1] - src_mean[1]
            dst_demean[i][0] = self.umeyama_args_112[2 * i] - dst_mean[0]
            dst_demean[i][1] = self.umeyama_args_112[2 * i + 1] - dst_mean[1]

        # 计算协方差矩阵 / Calculate covariance matrix
        A = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(SRC_DIM):
            for k in range(SRC_DIM):
                for j in range(SRC_NUM):
                    A[i][k] += dst_demean[j][i] * src_demean[j][k]
                A[i][k] /= SRC_NUM

        # SVD分解和旋转矩阵计算 / SVD decomposition and rotation matrix calculation
        T = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        U, S, V = self.svd22([A[0][0], A[0][1], A[1][0], A[1][1]])

        T[0][0] = U[0] * V[0] + U[1] * V[2]
        T[0][1] = U[0] * V[1] + U[1] * V[3]
        T[1][0] = U[2] * V[0] + U[3] * V[2]
        T[1][1] = U[2] * V[1] + U[3] * V[3]

        # 计算缩放因子 / Calculate scaling factor
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

        # 计算平移向量 / Calculate translation vector
        T[0][2] = dst_mean[0] - scale * (T[0][0] * src_mean[0] + T[0][1] * src_mean[1])
        T[1][2] = dst_mean[1] - scale * (T[1][0] * src_mean[0] + T[1][1] * src_mean[1])

        # 应用缩放 / Apply scaling
        T[0][0] *= scale
        T[0][1] *= scale
        T[1][0] *= scale
        T[1][1] *= scale

        return T

    def get_affine_matrix(self, sparse_points):
        """获取仿射变换矩阵 / Get affine transformation matrix"""
        with ScopedTiming("get_affine_matrix", self.debug_mode > 1):
            matrix_dst = self.image_umeyama_112(sparse_points)
            matrix_dst = [matrix_dst[0][0], matrix_dst[0][1], matrix_dst[0][2],
                         matrix_dst[1][0], matrix_dst[1][1], matrix_dst[1][2]]
            return matrix_dst

class FaceRegistration:
    """人脸注册主类 / Main Face Registration Class

    整合人脸检测和注册功能的主类
    Main class that integrates face detection and registration functions
    """

    def __init__(self, face_det_kmodel, face_reg_kmodel, det_input_size,
                 reg_input_size, database_dir, anchors,
                 confidence_threshold=0.25, nms_threshold=0.3,
                 rgb888p_size=[1280,720], display_size=[1920,1080],
                 debug_mode=0):
        """初始化函数 / Initialization function"""
        # 人脸检测模型路径 / Face detection model path
        self.face_det_kmodel = face_det_kmodel
        # 人脸注册模型路径 / Face registration model path
        self.face_reg_kmodel = face_reg_kmodel
        # 人脸检测模型输入尺寸 / Face detection model input size
        self.det_input_size = det_input_size
        # 人脸注册模型输入尺寸 / Face registration model input size
        self.reg_input_size = reg_input_size
        self.database_dir = database_dir
        self.anchors = anchors
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        # RGB尺寸，确保16字节对齐 / RGB size, ensure 16-byte aligned
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        # 显示尺寸，确保16字节对齐 / Display size, ensure 16-byte aligned
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode

        # 初始化人脸检测和注册模型 / Initialize face detection and registration models
        self.face_det = FaceDetApp(self.face_det_kmodel,
                                 model_input_size=self.det_input_size,
                                 anchors=self.anchors,
                                 confidence_threshold=self.confidence_threshold,
                                 nms_threshold=self.nms_threshold,
                                 debug_mode=0)
        self.face_reg = FaceRegistrationApp(self.face_reg_kmodel,
                                          model_input_size=self.reg_input_size,
                                          rgb888p_size=self.rgb888p_size)

    def run(self, input_np, img_file):
        """运行人脸注册流程 / Run face registration process"""
        # 配置人脸检测预处理 / Configure face detection preprocessing
        self.face_det.config_preprocess(input_image_size=[input_np.shape[3],input_np.shape[2]])
        # 执行人脸检测 / Perform face detection
        det_boxes, landms = self.face_det.run(input_np)

        try:
            if det_boxes:
                if det_boxes.shape[0] == 1:
                    # 若只检测到一张人脸，进行注册 / If only one face is detected, proceed with registration
                    db_i_name = img_file.split('.')[0]
                    for landm in landms:
                        # 配置人脸注册预处理 / Configure face registration preprocessing
                        self.face_reg.config_preprocess(landm, input_image_size=[input_np.shape[3],input_np.shape[2]])
                        # 执行人脸特征提取 / Perform face feature extraction
                        reg_result = self.face_reg.run(input_np)
                        # 保存特征到数据库 / Save features to database
                        with open(self.database_dir+'{}.bin'.format(db_i_name), "wb") as file:
                            file.write(reg_result.tobytes())
                            print('Success!')
                else:
                    print('Only one person in a picture when you sign up')
            else:
                print('No person detected')
        except:
            print("Register failed")

    def image2rgb888array(self, img):
        """将图像转换为RGB888数组 / Convert image to RGB888 array"""
        with ScopedTiming("fr_kpu_deinit", self.debug_mode > 0):
            # 转换为RGB888格式 / Convert to RGB888 format
            img_data_rgb888 = img.to_rgb888()
            # 转换为numpy数组 / Convert to numpy array
            img_hwc = img_data_rgb888.to_numpy_ref()
            shape = img_hwc.shape
            # 重塑并转置数组 / Reshape and transpose array
            img_tmp = img_hwc.reshape((shape[0] * shape[1], shape[2]))
            img_tmp_trans = img_tmp.transpose()
            img_res = img_tmp_trans.copy()
            # 返回NCHW格式的数组 / Return array in NCHW format
            img_return = img_res.reshape((1, shape[2], shape[0], shape[1]))
        return img_return

def ensure_dir(directory):
    """
    递归创建目录
    (Recursively create directory)
    """
    # 如果目录为空字符串或根目录，直接返回
    # (If directory is empty string or root directory, return directly)
    if not directory or directory == '/':
        return

    # 处理路径分隔符，确保使用标准格式
    # (Process path separators to ensure standard format)
    directory = directory.rstrip('/')

    try:
        # 尝试获取目录状态，如果目录存在就直接返回
        # (Try to get directory status, if directory exists then return directly)
        os.stat(directory)
        print(f'目录已存在: {directory}')
        # (Directory already exists: {directory})
        return
    except OSError:
        # 目录不存在，需要创建
        # (Directory does not exist, need to create)

        # 分割路径以获取父目录
        # (Split path to get parent directory)
        if '/' in directory:
            parent = directory[:directory.rindex('/')]
            if parent and parent != directory:  # 避免无限递归
                                                # (Avoid infinite recursion)
                ensure_dir(parent)

        try:
            # 创建目录
            # (Create directory)
            os.mkdir(directory)
            print(f'已创建目录: {directory}')
            # (Directory created: {directory})
        except OSError as e:
            # 可能是并发创建导致的冲突，再次检查目录是否存在
            # (Possible conflict due to concurrent creation, check again if directory exists)
            try:
                os.stat(directory)
                print(f'目录已被其他进程创建: {directory}')
                # (Directory has been created by another process: {directory})
            except:
                # 如果仍然不存在，则确实出错了
                # (If it still doesn't exist, there is definitely an error)
                print(f'创建目录时出错: {e}')
                # (Error creating directory: {e})
    except Exception as e:
        # 捕获其他可能的异常
        # (Catch other possible exceptions)
        print(f'处理目录时出错: {e}')
        # (Error processing directory: {e})

def get_directory_name(path):
    """获取路径中的目录名 / Get directory name from path"""
    parts = path.split('/')
    for part in reversed(parts):
        if part:
            return part
    return ''

def exce_demo(pl=None):
    """执行演示的主函数 / Main function to execute demonstration"""
    global eg

    # 配置模型和参数路径 / Configure model and parameter paths
    face_det_kmodel_path = "/sdcard/kmodel/face_detection_320.kmodel"
    face_reg_kmodel_path = "/sdcard/kmodel/face_recognition.kmodel"
    anchors_path = "/sdcard/utils/prior_data_320.bin"

    # 此处需要修改为你的人脸照片所在的目录
    # change this path to where your face photo in
    database_img_dir = "/data/photo/931783/"
    dir_name = get_directory_name(database_img_dir)
    face_det_input_size = [320,320]
    face_reg_input_size = [112,112]
    confidence_threshold = 0.5
    nms_threshold = 0.2
    anchor_len = 4200
    det_dim = 4

    # 加载anchors数据 / Load anchors data
    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((anchor_len, det_dim))

    # 设置最大注册人脸数和特征维度 / Set maximum number of registered faces and feature dimensions
    max_register_face = 100
    feature_num = 128

    print("Start ...")
    database_dir = "/data/face_database/" + dir_name + "/"
    ensure_dir(database_dir)

    # 初始化人脸注册对象 / Initialize face registration object
    fr = FaceRegistration(face_det_kmodel_path, face_reg_kmodel_path,
                         det_input_size=face_det_input_size,
                         reg_input_size=face_reg_input_size,
                         database_dir=database_dir,
                         anchors=anchors,
                         confidence_threshold=confidence_threshold,
                         nms_threshold=nms_threshold)

    # 获取图像列表并处理 / Get image list and process
    img_list = os.listdir(database_img_dir)
    try:
        for img_file in img_list:
            # 读取图像 / Read image
            full_img_file = database_img_dir + img_file
            print(full_img_file)
            img = image.Image(full_img_file)
            img.compress_for_ide()
            # 转换图像格式并处理 / Convert image format and process
            rgb888p_img_ndarry = fr.image2rgb888array(img)
            fr.run(rgb888p_img_ndarry, img_file)
            gc.collect()
    except Exception as e:
        print("人脸注册功能异常退出")
    finally:
        fr.face_det.deinit()
        fr.face_reg.deinit()
        print("人脸注册功能退出")

def exit_demo():
    """退出函数 / Exit function"""
    global fr
    # 清理资源 / Clean up resources
    fr.face_det.deinit()
    fr.face_reg.deinit()

if __name__ == "__main__":
    """程序入口 / Program entry"""
    exce_demo(None)

```

### 6.云API调用

- 例如调用讯飞的文字识别API 识别到文字进行相关录用
- 先不忙实现
- 注册时 可以根据识别的文字 进行相关录入

### 7.控制步进电机模块 采用PUL/DIR/EN 输出脉冲信号控制电机

- 控制电机时 需要生成对应的日志文件
- 日志需要包含 开门时间 开门方法 开门者等一系列消息
- 提供对应API使能够 对不同的指令做出不同响应（与后面的云功能进行配合）

### 8.程序需要多线程支持 以确保异步

```python
# 导入线程模块 Import thread module
import _thread
# 导入时间模块用于实现延时 Import time module for delay functionality
import time

# 定义线程执行的函数 Define the function to be executed in threads
# name: 线程名称参数 Thread name parameter
def func(name):
    while True:
        # 每隔一秒输出一次信息 Print message every second
        print("This is thread {}".format(name))
        # 休眠1秒 Sleep for 1 second
        time.sleep(1)

# 创建并启动第一个线程 Create and start the first thread
# func: 线程函数 Thread function
# ("THREAD_1",): 传递给线程函数的参数(必须是元组格式)
# Arguments passed to thread function (must be tuple format)
_thread.start_new_thread(func,("THREAD_1",))

# 延时500毫秒
# Delay 500ms to give the first thread a chance to start
time.sleep_ms(500)

# 创建并启动第二个线程 Create and start the second thread
# 参数与第一个线程类似 Similar parameters as the first thread
_thread.start_new_thread(func,("THREAD_2",))

# 主线程死循环,防止程序退出
# Main thread infinite loop to prevent program exit
# 延时1毫秒,避免占用过多CPU资源
# Delay 1ms to avoid consuming too much CPU
while True:
    time.sleep_ms(1)

```

---

## 一些重要的API手册

官方提供多种API 下面提及一些可能用到的API
Micropython特有库：
uctypes 模块 
network 模块 
socket 模块 
ADC 模块 
FFT 模块 
Pin 模块 
I2C 模块 
I2C_Slave 模块 
FPIOA 模块 
PWM 模块 
SPI 模块 
Timer 模块 
WDT 模块 
UART 模块 
machine 模块 
RTC 模块 
TOUCH 模块 
neopixel 模块 
SPI_LCD 模块 
USB Serial 模块 
Python 标准库和 Micropython 标准微库：
Ucryptolib 模块 
uhashlib 模块 
utime 时间相关功能 
gc – 内存管理 
uos – 基本操作系统服务

### Pin 模块 API 手册
概述
K230 芯片内部包含 64 个 GPIO（通用输入输出）引脚，每个引脚均可配置为输入或输出模式，并支持上下拉电阻配置和驱动能力设置。这些引脚能够灵活用于各种数字输入输出场景。

API 介绍
Pin 类位于 machine 模块中，用于控制 K230 芯片的 GPIO 引脚。

示例

from machine import Pin

# 将引脚 2 配置为输出模式，无上下拉，驱动能力为 7
pin = Pin(2, Pin.OUT, pull=Pin.PULL_NONE, drive=7)

# 设置引脚 2 输出高电平
pin.value(1)

# 设置引脚 2 输出低电平
pin.value(0)
构造函数
pin = Pin(index, mode, pull=Pin.PULL_NONE, value = -1, drive=7, alt = -1)
参数

index: 引脚编号，范围为 [0, 63]。

mode: 引脚的模式，支持输入模式或输出模式。

pull: 上下拉配置（可选），默认为 Pin.PULL_NONE。

drive: 驱动能力配置（可选），默认值为 7。

value: 设置引脚默认输出值

alt: 目前未使用

init 方法
Pin.init(mode, pull=Pin.PULL_NONE, drive=7)
用于初始化引脚的模式、上下拉配置及驱动能力。

参数

mode: 引脚的模式（输入或输出）。

pull: 上下拉配置（可选），默认值为 Pin.PULL_NONE。

drive: 驱动能力（可选），默认值为 7。

返回值

无

value 方法
Pin.value([value])
获取引脚的输入电平值或设置引脚的输出电平。

参数

value: 输出值（可选），如果传递该参数则设置引脚输出为指定值。如果不传参则返回引脚的当前输入电平值。

返回值

返回空或当前引脚的输入电平值。

mode 方法
Pin.mode([mode])
获取或设置引脚的模式。

参数

mode: 引脚模式（输入或输出），如果不传参则返回当前引脚的模式。

返回值

返回空或当前引脚模式。

pull 方法
Pin.pull([pull])
获取或设置引脚的上下拉配置。

参数

pull: 上下拉配置（可选），如果不传参则返回当前上下拉配置。

返回值

返回空或当前引脚的上下拉配置。

drive 方法
Pin.drive([drive])
获取或设置引脚的驱动能力。

参数

drive: 驱动能力（可选），如果不传参则返回当前驱动能力。

返回值

返回空或当前引脚的驱动能力。

on 方法
Pin.on()
将引脚输出设置为高电平。

参数

无

返回值

无

off 方法
Pin.off()
将引脚输出设置为低电平。

参数

无

返回值

无

high 方法
Pin.high()
将引脚输出设置为高电平。

参数

无

返回值

无

low 方法
Pin.low()
将引脚输出设置为低电平。

参数

无

返回值

无

irq 方法
Pin.irq(handler=None, trigger=Pin.IRQ_FALLING | Pin.IRQ_RISING, *, priority=1, wake=None, hard=False, debounce = 10)
使能 IO 中断功能

handler: 回调函数，必须设置

trigger: 触发模式

priority: 不支持

wake: 不支持

hard: 不支持

debounce: 高电平和低电平触发时，最小触发间隔，单位为 ms，最小值为 5

返回值

mq_irq 对象

常量定义
模式
Pin.IN: 输入模式

Pin.OUT: 输出模式

上下拉模式
PULL_NONE: 关掉上下拉

PULL_UP: 使能上拉

PULL_DOWN: 使能下拉

中断触发模式
IRQ_FALLING: 下降沿触发

IRQ_RISING: 上升沿触发

IRQ_LOW_LEVEL: 低电平触发

IRQ_HIGH_LEVEL: 高电平触发

IRQ_BOTH: 边沿触发

驱动能力
具体配置对应的电流输出能力参见fpioa

DRIVE_0

DRIVE_1

DRIVE_2

DRIVE_3

DRIVE_4

DRIVE_5

DRIVE_6

DRIVE_7

DRIVE_8

DRIVE_9

DRIVE_10

DRIVE_11

DRIVE_12

DRIVE_13

DRIVE_14

DRIVE_15

### UART模块

UART 模块 API 手册
概述
K230 内部集成了五个 UART（通用异步收发传输器）硬件模块，其中 UART0 被小核 SH 占用，UART3 被大核 SH 占用，剩余的 UART1、UART2 和 UART4 可供用户使用。UART 的 I/O 配置可参考 IOMUX 模块。

API 介绍
UART 类位于 machine 模块中。

示例代码
from machine import UART

# 配置 UART1: 波特率 115200, 8 位数据位, 无奇偶校验, 1 个停止位
u1 = UART(UART.UART1, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)

# 写入数据到 UART
u1.write("UART1 test")

# 从 UART 读取数据
r = u1.read()

# 读取一行数据
r = u1.readline()

# 将数据读入字节缓冲区
b = bytearray(8)
r = u1.readinto(b)

# 释放 UART 资源
u1.deinit()
构造函数
uart = UART(id, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE, timeout = 0)
参数

id: UART 模块编号，有效值为 UART1、UART2、UART4。

baudrate: UART 波特率，可选参数，默认值为 115200。

bits: 每个字符的数据位数，有效值为 FIVEBITS、SIXBITS、SEVENBITS、EIGHTBITS，可选参数，默认值为 EIGHTBITS。

parity: 奇偶校验，有效值为 PARITY_NONE、PARITY_ODD、PARITY_EVEN，可选参数，默认值为 PARITY_NONE。

stop: 停止位数，有效值为 STOPBITS_ONE、STOPBITS_TWO，可选参数，默认值为 STOPBITS_ONE。

timeout: 读数据超时，单位为 ms

init 方法
UART.init(baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)
配置 UART 参数。

参数

参考构造函数。

返回值

无

read 方法
UART.read([nbytes])
读取字符。如果指定了 nbytes，则最多读取该数量的字节；否则，将尽可能多地读取数据。

参数

nbytes: 最多读取的字节数，可选参数。

返回值

返回一个包含读取字节的字节对象。

readline 方法
UART.readline()
读取一行数据，并以换行符结束。

参数

无

返回值

返回一个包含读取字节的字节对象。

readinto 方法
UART.readinto(buf[, nbytes])
将字节读取到 buf 中。如果指定了 nbytes，则最多读取该数量的字节；否则，最多读取 len(buf) 数量的字节。

参数

buf: 一个缓冲区对象。

nbytes: 最多读取的字节数，可选参数。

返回值

返回读取并存入 buf 的字节数。

write 方法
UART.write(buf)
将字节缓冲区写入 UART。

参数

buf: 一个缓冲区对象。

返回值

返回写入的字节数。

deinit 方法
UART.deinit()
释放 UART 资源。

参数

无

返回值

无

### Timer模块

Timer 模块 API 手册
概述
K230 内部集成了 6 个 Timer 硬件模块，最小定时周期为 1 毫秒（ms）。

API 介绍
Timer 类位于 machine 模块中。

示例代码
from machine import Timer
import time

# 实例化一个软定时器
tim = Timer(-1)

# 配置定时器，单次模式，周期 100 毫秒，回调函数打印 1
tim.init(period=100, mode=Timer.ONE_SHOT, callback=lambda t: print(1))
time.sleep(0.2)

# 配置定时器，周期模式，周期 1000 毫秒，回调函数打印 2
tim.init(freq=1, mode=Timer.PERIODIC, callback=lambda t: print(2))
time.sleep(2)

# 释放定时器资源
tim.deinit()
构造函数
timer = Timer(index, mode=Timer.PERIODIC, freq=-1, period=-1, callback=None)
参数

index: Timer 模块编号，取值范围为 [-1, 5]，其中 -1 表示软件定时器。

mode: 定时器运行模式，可以是单次或周期模式（可选参数）。

freq: 定时器运行频率，支持浮点数，单位为赫兹（Hz），此参数优先级高于 period（可选参数）。

period: 定时器运行周期，单位为毫秒（ms）（可选参数）。

callback: 超时回调函数，必须设置并应带有一个参数。

init 方法
Timer.init(mode=Timer.PERIODIC, freq=-1, period=-1, callback=None)
初始化定时器参数。

参数

mode: 定时器运行模式，可以是单次或周期模式（可选参数）。

freq: 定时器运行频率，支持浮点数，单位为赫兹（Hz），此参数优先级高于 period（可选参数）。

period: 定时器运行周期，单位为毫秒（ms）（可选参数）。

callback: 超时回调函数，必须设置并应带有一个参数。

返回值

无

deinit 方法
Timer.deinit()
释放定时器资源。

参数

无

返回值

无

### NETWORK 模块

network 模块 API 手册
概述
本模块主要用于配置和查看网络参数，配置完成后，方可使用 socket 模块进行网络通信。

LAN 类
参考文档: Micropython LAN

此类为有线网络的配置接口。示例代码如下：

import network
nic = network.LAN()
print(nic.ifconfig())

# 配置完成后，即可像往常一样使用 socket
...
构造函数
class network.LAN() ¶

创建一个有线以太网对象。

方法
LAN.active([state]) ¶

激活或停用网络接口。传递布尔参数 True 表示激活，False 表示停用。如果不传参数，则返回当前状态。

LAN.isconnected() ¶

返回 True 表示已连接到网络，返回 False 表示未连接。

LAN.ifconfig([(ip, subnet, gateway, dns)]) ¶

获取或设置 IP 级别的网络接口参数，包括 IP 地址、子网掩码、网关和 DNS 服务器。无参数调用时，返回一个包含上述信息的四元组；如需设置参数，传入包含 IP 地址、子网掩码、网关和 DNS 的四元组。例如：

nic.ifconfig(('192.168.0.4', '255.255.255.0', '192.168.0.1', '8.8.8.8'))
LAN.config(config_parameters) ¶

获取或设置网络接口参数。当前仅支持设置或获取 MAC 地址。例如：

import network
lan = network.LAN()
# 设置 MAC 地址
lan.config(mac="42:EA:D0:C2:0D:83")
# 获取 MAC 地址
print(lan.config("mac"))
WLAN 类
参考文档: Micropython WLAN

此类为 WiFi 网络配置接口。示例代码如下：

import network
import time

SSID = "TEST"
PASSWORD = "12345678"

sta = network.WLAN(network.STA_IF)

sta.connect(SSID, PASSWORD)

timeout = 10  # 单位：秒
start_time = time.time()

while not sta.isconnected():
    if time.time() - start_time > timeout:
        print("连接超时")
        break
    time.sleep(1)  # 请稍等片刻再连接

print(sta.ifconfig())

print(sta.status())

# 这里的断开网络，只是一个测试。实际应用可不断开
sta.disconnect()
print("断开网络")
print(sta.status())

构造函数
class network.WLAN(*interface_id*)

创建 WLAN 网络接口对象。支持的接口类型包括 network.STA_IF（即站模式，连接到上游 WiFi 接入点）和 network.AP_IF（即接入点模式，允许其他设备连接）。不同接口类型的方法有所不同，例如，只有 STA 模式支持通过 WLAN.connect() 连接到接入点。

方法
WLAN.active()

查询当前接口是否激活

WLAN.connect(ssid=None, key=None, [info = None])

连接到指定 ssid 或者 info，info 是通过 scan 返回的结果。

仅 Sta 模式可用

WLAN.disconnect()

Sta 模式时断开当前的 WiFi 网络连接。 Ap 模式时，可传入指定 mac 来断开设备的连接。

WLAN.scan()

扫描可用的 WiFi 网络。此方法仅在 STA 模式下有效，返回的列表包含每个网络的信息，例如：

# print(sta.scan())
[{"ssid":"XCTech", "bssid":xxxxxxxxx, "channel":3, "rssi":-76, "security":"SECURITY_WPA_WPA2_MIXED_PSK", "band":"2.4G", "hidden":0},...]
WLAN.status([param])

返回当前网络连接的信息。当不传参数时，返回当前的连接状态。例如：

# 查看连接状态 等同与 sta.isconnected()
print(sta.status())

# 查看连接的信号质量
print(sta.status("rssi"))
支持的配置参数包括：

Sta 模式时

rssi: 连接信号质量

ap: 连接的热点名称

Ap 模式时

stations: 返回连接的设备信息

WLAN.isconnected()

返回是否连接到热点

仅 Sta 模式可用

WLAN.ifconfig([(ip, subnet, gateway, dns)])

获取或设置 IP 级别的网络接口参数。无参数调用时，返回包含 IP 地址、子网掩码、网关和 DNS 服务器的四元组；传入参数则设置这些值。例如：

sta.ifconfig(('192.168.0.4', '255.255.255.0', '192.168.0.1', '8.8.8.8'))
WLAN.config(param)

获取或设置网络接口的配置参数。支持的参数包括 MAC 地址、SSID、WiFi 通道、是否隐藏 SSID、密码等。设置参数时使用关键字参数语法；查询参数时，传递参数名即可。例如：

# 查看 auto_reconnect 配置
print(sta.config('auto_reconnect'))

# 设置自动重连
sta.config(auto_reconnect = True)
支持的配置参数包括：

Sta 模式时

mac: mac 地址

auto_reconnect: 是否自动重连

Ap 模式时

info: 当前热点信息，仅可查询

country: 国家代码

WLAN.stop()

停止开启热点

仅 Ap 模式可用

WLAN.info()

查询当前热点信息

仅 Ap 模式可用

### SOCKET 模块

socket 模块 API 手册
概述
该模块封装了 socket 库，用户可以通过调用 socket 库进行网络应用程序开发。

示例
# 配置 tcp/udp socket 调试工具
import socket
import time

PORT = 60000

def client():
    # 获取 IP 地址及端口号
    ai = socket.getaddrinfo("10.100.228.5", PORT)
    # ai = socket.getaddrinfo("10.10.1.94", PORT)
    print("地址信息:", ai)
    addr = ai[0][-1]

    print("连接地址:", addr)
    # 创建 socket 对象
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    # 连接到指定地址
    s.connect(addr)

    for i in range(10):
        msg = "K230 TCP 客户端发送测试 {0} \r\n".format(i)
        print(msg)
        # 发送字符串数据
        print(s.write(msg))
        time.sleep(0.2)

    # 延时 1 秒后关闭 socket
    time.sleep(1)
    s.close()
    print("结束")

# 运行客户端程序
client()
API 定义
详见 Micropython socket module

构造函数
class socket.socket(af=AF_INET, type=SOCK_STREAM, proto=IPPROTO_TCP)

创建一个新的套接字对象，使用指定的地址族（af）、套接字类型（type）和协议（proto）。通常，无需显式指定 proto 参数，MicroPython 会根据 type 自动选择相应的协议类型。示例：

创建一个 TCP 流套接字：socket(AF_INET, SOCK_STREAM)

创建一个 UDP 数据报套接字：socket(AF_INET, SOCK_DGRAM)

方法
socket.close()

关闭套接字并释放其相关资源。关闭后，所有对该套接字对象的操作将会失败。协议支持时，远程端会收到 EOF 指示。尽管套接字在被垃圾回收时会自动关闭，但建议在使用完后显式调用 close() 方法。

socket.bind(address)

将套接字绑定到指定的 IP 地址和端口。调用前确保套接字未被绑定。

socket.listen([backlog])

使服务器套接字开始监听连接请求。backlog 指定等待连接的最大数目，最小为 0（小于 0 的值将视为 0），如果未指定，则使用系统默认值。

socket.accept()

接受客户端连接。此方法返回 (conn, address)，其中 conn 是一个新套接字对象，可用于在该连接上发送和接收数据，address 是客户端的地址。

socket.connect(address)

连接到指定的服务端套接字地址。

socket.send(bytes)

向套接字发送数据，套接字必须已连接。返回发送的字节数，这可能小于数据的总长度（即“短写”情况）。

socket.sendall(bytes)

向套接字发送完整数据，套接字必须已连接。与 send() 不同，此方法会尝试连续发送所有数据，直至传输完成。该方法在非阻塞套接字上行为未定义，因此建议在 MicroPython 中使用 write() 方法。

socket.recv(bufsize)

从套接字接收数据，返回接收到的数据字节对象。bufsize 指定单次接收的最大字节数。

socket.sendto(bytes, address)

向未连接的套接字发送数据，目标地址由 address 指定。

socket.recvfrom(bufsize)

从未连接的套接字接收数据，返回 (bytes, address)，其中 bytes 是接收到的数据，address 是发送数据的源地址。

socket.setsockopt(level, optname, value)

设置套接字选项。value 可以是整数或字节类对象。

socket.settimeout(value)

设置套接字操作的超时时间（以秒为单位）。value 可以是一个正数、零或 None。若超时，操作将引发 OSError。在非阻塞模式下，设置 value 为零；在阻塞模式下，设置为 None。

socket.setblocking(flag)

设置套接字的阻塞模式。flag 为 False 时为非阻塞模式，为 True 时为阻塞模式。

socket.makefile(mode=’rb’, buffering=0)

返回与套接字关联的文件对象。仅支持二进制模式（如 rb、wb 和 rwb），buffering 参数在 MicroPython 中被忽略。

socket.read([size])

从套接字读取数据，返回字节对象。若未指定 size，则读取所有可用数据，直至 EOF。

socket.readinto(buf[, nbytes])

将数据读取到 buf 中，若指定 nbytes，则读取最多 nbytes 字节，否则读取 len(buf) 字节。

socket.readline()

从套接字读取一行数据，返回字节对象。

socket.write(buf)

将 buf 中的数据写入套接字。尽量写入所有数据，但对于非阻塞套接字，可能只写入部分数据。

socket.error

在 MicroPython 中未定义 socket.error 异常，直接使用 OSError。

### UOS 模块

uos – 基本操作系统服务
该模块提供了部分 CPython 操作系统模块的功能子集。有关详细信息，请参阅原始 CPython 文档：os。

uos 模块包含用于文件系统的访问与挂载、终端重定向与复制，以及诸如 uname 和 urandom 等系统信息和随机数生成函数。

函数
基础功能
uname
uos.uname()
返回一个包含有关底层机器及其操作系统信息的元组。该元组包含以下五个字段，每个字段均为字符串：

sysname – 底层操作系统名称。

nodename – 节点名称（可能与 sysname 相同）。

release – 操作系统的版本号。

version – MicroPython 版本及构建日期。

machine – 硬件标识符（如主板型号、CPU 类型等）。

urandom
uos.urandom(n)
生成并返回包含 n 个随机字节的字节对象。随机字节尽可能由硬件随机数生成器提供。

cpu_usage
uos.cpu_usage()
返回当前系统 CPU 的使用率，范围为 0 到 100。

文件系统操作
chdir
uos.chdir(path)
更改当前工作目录。

getcwd
uos.getcwd()
获取当前工作目录的路径。

ilistdir
uos.ilistdir([dir])
返回指定目录（或当前目录）的条目信息。该函数生成一个元组迭代器，其中每个元组的形式为 (name, type, inode [, size])。

name：条目名称，字符串类型。

type：条目类型，目录为 0x4000，普通文件为 0x8000。

inode：文件系统的 inode 值，对于不支持 inode 的文件系统则为 0。

size（可选）：文件大小，若无法获取则为 -1。

listdir
uos.listdir([dir])
列出指定目录中的所有条目。如果未指定目录，则列出当前目录。

mkdir
uos.mkdir(path)
在指定路径创建一个新目录。

remove
uos.remove(path)
删除指定路径的文件。

rmdir
uos.rmdir(path)
删除指定路径的目录。

rename
uos.rename(old_path, new_path)
重命名指定路径的文件或目录。

stat
uos.stat(path)
返回指定路径的文件或目录的状态信息。

statvfs
uos.statvfs(path)
获取指定路径文件系统的状态，返回一个元组，包含以下字段：

f_bsize – 文件系统块大小。

f_frsize – 片段大小。

f_blocks – 文件系统总大小，以 f_frsize 为单位。

f_bfree – 空闲块数。

f_bavail – 非特权用户可用的空闲块数。

f_files – inode 总数。

f_ffree – 可用 inode 数量。

f_favail – 非特权用户可用的 inode 数量。

f_flag – 挂载标志。

f_namemax – 最大文件名长度。

目前仅 f_bsize， f_blocks 和 f_bfree 有效

sync
uos.sync()
同步所有文件系统，将挂起的写操作写入存储设备。

dupterm
uos.dupterm(stream_object, index = 0)
在指定的 stream 对象上复制或切换 MicroPython 终端（REPL）。stream_object 必须实现 readinto() 和 write() 方法。stream 应以非阻塞模式运行，readinto() 在没有可读数据时应返回 None。

调用后，所有终端输出将被复制到该流对象，同时该流上提供的任何输入也将被传递到终端。index 参数应为非负整数，指定要设置的复制槽。

如果 stream_object 为 None，则取消指定槽的终端复制。

返回先前在指定槽中的流对象。

### UTIME 模块

utime 时间相关功能 API 手册
该模块实现了部分 CPython 模块的功能子集，具体如下所述。更多详细信息，请参考 CPython 原始文档：time。

utime 模块提供获取当前时间与日期、测量时间间隔以及延迟操作的相关功能。

纪元时间：Unix 系统移植版本使用 1970-01-01 00:00:00 UTC 作为 POSIX 系统的标准纪元时间。

维护实际日历日期/时间：这需要使用实时时钟（RTC）。在运行底层操作系统（包括部分实时操作系统，RTOS）的系统上，RTC 可能默认启用。设置和维护实际日历时间的工作由操作系统或 RTOS 负责，并且是在 MicroPython 之外完成的，MicroPython 只通过操作系统的 API 查询日期和时间。

函数
ntp_sync
utime.ntp_sync()
当系统联网后，调用该函数可以从互联网同步当前时间。函数返回 True 或 False，表示同步是否成功。某些开发板不支持 RTC 模块，因此该函数在这些板上总是返回 False。

localtime
utime.localtime([secs])
将自纪元以来以秒为单位的时间转换为 8 元组，包含以下信息：(年，月，日，小时，分钟，秒，工作日，yearday)。如果未提供秒数，则返回来自 RTC 的当前时间。

年份包含世纪（如 2014 年）

月份范围为 1-12

日（mday）范围为 1-31

小时范围为 0-23

分钟范围为 0-59

秒范围为 0-59

工作日范围为 0（周一）至 6（周日）

yearday 范围为 1-366

mktime
utime.mktime(tuple)
该函数是 localtime() 的逆函数。它接受一个 8 元组，表示本地时间，并返回自 1970-01-01 00:00:00 以来的秒数。

sleep
utime.sleep(seconds)
延迟执行指定的秒数。部分开发板支持以浮点数传入秒数，以实现亚秒级的延迟。不过，为确保兼容性，推荐使用 sleep_ms() 和 sleep_us() 函数来处理毫秒和微秒级的延迟。

sleep_ms
utime.sleep_ms(ms)
延迟指定的毫秒数。

sleep_us
utime.sleep_us(us)
延迟指定的微秒数。

ticks_ms
utime.ticks_ms()
返回一个递增的毫秒计数器，参考点为系统内部的任意时间点，该计数器会在某个值后回绕。

ticks_us
utime.ticks_us()
与 ticks_ms() 类似，但返回的是微秒级计数。

ticks_cpu
utime.ticks_cpu()
提供最高分辨率的计数器，通常与 CPU 时钟相关，用于高精度基准测试或紧凑的实时循环。

ticks_add
utime.ticks_add(ticks, delta)
根据指定的时间增量（delta，可以为正或负数）计算新的 ticks 值，用于设定任务的截止时间等。

ticks_diff
utime.ticks_diff(ticks1, ticks2)
计算两个 ticks 值之间的差异，支持处理计数器回绕。

time
utime.time()
返回自纪元以来的秒数，前提是已设置 RTC。如果未设置 RTC，则返回自系统上电或复位以来的秒数。

ticks
utime.ticks()
等同于 utime.ticks_ms()。

clock
utime.clock()
返回一个 clock 对象，用于时间测量和 FPS 计算。

clock 类
构造函数
utime.clock()
方法
tick
clock.tick()
记录当前时间（毫秒），可用于 FPS 计算。

fps
clock.fps()
根据上一次 clock.tick() 调用后的时间间隔，计算帧率（FPS）。

示例：

import utime
clock = utime.clock()
while True:
    clock.tick()
    utime.sleep(0.1)
    print("fps = ", clock.fps())
reset
clock.reset()
重置所有计时标记。

avg
clock.avg()
计算每帧的平均时间消耗。

### GC 模块

gc – 内存管理 API 手册
该模块实现了部分 CPython 内存管理模块的功能子集。有关详细信息，请参阅 CPython 原始文档：gc。

在 K230 上，新增了以下接口来获取 RT-Smart 系统的内存信息：

sys_totoal: 系统内存大小

sys_heap: 用于内核应用的内存管理。

sys_page: 用于用户应用的内存管理。

sys_mmz: 用于多媒体驱动内存管理，适用于 Sensor、Display 等模块。

函数
enable
gc.enable()
启用自动垃圾回收机制。

disable
gc.disable()
禁用自动垃圾回收机制。在禁用状态下，仍可进行堆内存的分配，并且可以通过手动调用 gc.collect() 来执行垃圾回收。

collect
gc.collect()
手动运行垃圾回收过程，回收不再使用的内存。

mem_alloc
gc.mem_alloc()
返回当前已分配的堆内存字节数。

与 CPython 的差异

此功能为 MicroPython 的扩展功能，CPython 并不包含此方法。

mem_free
gc.mem_free()
返回当前可用的堆内存字节数。如果堆剩余的内存数量无法确定，则返回 -1。

与 CPython 的差异

此功能为 MicroPython 的扩展功能，CPython 并不包含此方法。

threshold
gc.threshold([amount])
设置或查询垃圾回收的分配阈值。当堆内存不足时，通常会触发垃圾回收。如果设置了阈值，则在总共分配了超过设定值的字节后，也会触发垃圾回收。amount 参数通常小于整个堆的大小，目的是在堆耗尽之前提前触发回收，减少内存碎片。此值的效果因应用而异，最佳值需要根据应用场景调整。

不传入参数时，此函数将返回当前的阈值设置。返回值为 -1 表示分配阈值已禁用。

与 CPython 的差异

此函数为 MicroPython 的扩展。CPython 中有类似的 set_threshold() 函数，但由于垃圾回收机制的不同，其签名和语义有所差异。

sys_total
gc.sys_total()
查询系统内存大小，单位为字节（Bytes）

sys_heap
gc.sys_heap()
查询系统 heap 内存的使用情况，返回一个包含 3 个元素的元组，分别表示 total（总内存）、free（可用内存）和 used（已用内存），单位为字节（Byte）。

sys_page
gc.sys_page()
查询系统 page 内存的使用情况，返回一个包含 3 个元素的元组，分别表示 total（总内存）、free（可用内存）和 used（已用内存），单位为字节（Byte）。

sys_mmz
gc.sys_mmz()
查询系统 mmz 内存的使用情况，返回一个包含 3 个元素的元组，分别表示 total（总内存）、free（可用内存）和 used（已用内存），单位为字节（Byte）。

### WDT 模块

WDT 模块 API 手册
概述
K230 内部集成了两个 WDT（看门狗定时器）硬件模块，旨在确保系统在应用程序崩溃并进入不可恢复状态时能够重启。WDT 在启动后，如果硬件运行期间未定期进行“喂狗”操作，将会在超时后自动复位系统。

API 介绍
WDT 类位于 machine 模块中。

示例代码
from machine import WDT

# 实例化 WDT1，超时时间为 3 秒
wdt1 = WDT(1, 3)

# 执行喂狗操作
wdt1.feed()
构造函数
wdt = WDT(id=1, timeout=5, auto_close = True)
参数

id: WDT 模块编号，取值范围为 [0, 1]，默认为 1。

timeout: 超时值，单位为秒（s），默认为 5。

auto_close: 在 python 解释器停止运行的时候自动停止看门狗，防止系统被重启

注意： WDT0 暂不可用。

feed 方法
WDT.feed()
执行喂狗操作。

参数

无

返回值

无

### PWM 模块

PWM 模块 API 手册
概述
K230 内部包含两个 PWM 硬件模块，每个模块具有三个输出通道。每个模块的输出频率可调，但三个通道共享同一时钟，而占空比则可独立调整。因此，通道 0、1 和 2 输出频率相同，通道 3、4 和 5 输出频率也相同。通道输出的 I/O 配置请参考 IOMUX 模块。

API 介绍
PWM 类位于 machine 模块中。

示例代码
import time
from machine import PWM, FPIOA

CONSTRUCT_WITH_FPIOA = False

PWM_CHANNEL = 0

PWM_PIN = 42
TEST_FREQ = 1000  # Hz


# Initialize PWM with 50% duty
try:
    if CONSTRUCT_WITH_FPIOA:
        # 使用FPIOA 配置引脚复用为PWM, 构造时传入 channel
        fpioa = FPIOA()
        fpioa.set_function(PWM_PIN, fpioa.PWM0 + PWM_CHANNEL)
        pwm = PWM(PWM_CHANNEL, freq=TEST_FREQ, duty=50)
    else:
        # 直接传入引脚
        pwm = PWM(PWM_PIN, freq=TEST_FREQ, duty=50)
except Exception:
    print("FPIOA setup skipped or failed")

print("[INIT] freq: {}Hz, duty: {}%".format(pwm.freq(), pwm.duty()))
time.sleep(0.5)

# duty() getter and setter
print("[TEST] duty()")
pwm.duty(25)
print("Set duty to 25%, got:", pwm.duty(), "→ duty_u16:", pwm.duty_u16(), "→ duty_ns:", pwm.duty_ns())
time.sleep(0.2)

# duty_u16()
print("[TEST] duty_u16()")
pwm.duty_u16(32768)  # 50%
print("Set duty_u16 to 32768, got:", pwm.duty_u16(), "→ duty():", pwm.duty(), "→ duty_ns():", pwm.duty_ns())
time.sleep(0.2)

# duty_ns()
print("[TEST] duty_ns()")
period_ns = 1000000000 // pwm.freq()
duty_ns_val = (period_ns * 75) // 100  # 75%
pwm.duty_ns(duty_ns_val)
print("Set duty_ns to", duty_ns_val, "ns (≈75%), got:", pwm.duty_ns(), "→ duty():", pwm.duty(), "→ duty_u16():", pwm.duty_u16())
time.sleep(0.2)

# Change frequency and re-check duty values
print("[TEST] Change frequency to 500Hz")
pwm.freq(500)
print("New freq:", pwm.freq())
print("Duty after freq change → duty():", pwm.duty(), "→ duty_u16():", pwm.duty_u16(), "→ duty_ns():", pwm.duty_ns())
time.sleep(0.2)

# Clean up
pwm.deinit()
print("[DONE] PWM test completed")
构造函数
pwm = PWM(pin, freq = -1, duty = -1, duty_u16 = -1, duty_ns = -1)
参数

pin: Pin 对象，或者引脚号，驱动自动设置引脚为 PWM 复用功能，具体引脚对应 PWM 请参考 引脚复用

freq: PWM 通道输出频率

duty: PWM 通道输出占空比，表示高电平在整个周期中的百分比，取值范围为 [0, 100]

duty_ns: PWM 通道输出高电平的时间，单位为 ns

duty_u16: PWM通道输出高电平的时间，取值范围为 [0,65535]

duty 和 duty_ns 以及 duty_u16 只能设置其中的一个。

init 方法
PWM.init(freq = -1, duty = -1, duty_u16 = -1, duty_ns = -1)
参数

参考 构造函数

deinit 方法
PWM.deinit()
释放 PWM 通道的资源。

参数

无

返回值

无

freq 方法
PWM.freq([freq])
获取或设置 PWM 通道的输出频率。

参数

freq: PWM 通道输出频率，可选参数。如果不传入参数，则返回当前频率。

返回值

返回当前 PWM 通道的输出频率或空。

duty 方法
PWM.duty([duty])
获取或设置 PWM 通道的输出占空比。

参数

duty: PWM 通道输出占空比，可选参数。如果不传入参数，则返回当前占空比。

返回值

返回当前 PWM 通道的输出占空比或空。

返回值

无

duty_u16 方法
PWM.duty_u16([duty_u16])
获取或设置 PWM 通道的输出占空比。

参数

duty_u16: PWM 通道输出占空比，可选参数。如果不传入参数，则返回当前占空比。

返回值

返回当前 PWM 通道的输出占空比或空。

返回值

无

duty_ns 方法
PWM.duty_ns([duty_ns])
获取或设置 PWM 通道的输出占空比。

参数

duty_ns: PWM 通道输出占空比，可选参数。如果不传入参数，则返回当前占空比。

返回值

返回当前 PWM 通道的输出占空比或空。

返回值

无

PWM 引脚复用
PWM

可选引脚

PWM0

GPIO42, GPIO54, GPIO60

PWM1

GPIO43, GPIO55, GPIO61

PWM2

GPIO7, GPIO46, GPIO56

PWM3

GPIO8, GPIO47, GPIO57

PWM4

GPIO9, GPIO52, GPIO58

PWM5

GPIO25, GPIO53, GPIO59

### RTC模块

RTC 模块 API 手册
概述
当前 CanMV K230 提供一个类 RTC（实时时钟）模块，用户可以用其设置和获取当前系统时间。

API 介绍
RTC 类位于 machine 模块下。

示例
from machine import RTC

# 实例化 RTC
rtc = RTC()
# 获取当前时间
print(rtc.datetime())
# 设置当前时间
rtc.init((2024, 2, 28, 2, 23, 59, 0, 0))
构造函数
rtc = RTC()
参数

无

init 方法
rtc.init(year, month, day, hour, minute, second, microsecond, tzinfo)
参数

year: 年

month: 月

day: 日

hour: 时

minute: 分

second: 秒

microsecond: 微秒，忽略该参数

tzinfo: 时区，取值范围[-12 ~ 12]

返回值

无

datetime 方法
print(rtc.datetime())
参数

无

返回值

返回当前日期和时间信息，包括：

year: 年

mon: 月

day: 日

wday: 星期几

hour: 时

min: 分

sec: 秒

microsec: 微秒

### 文件读写
# 文件写入
with open('/sdcard/yahboom.txt', 'w') as f:
    f.write("Hello Yahboom")

# 文件读取
with open('/sdcard/yahboom.txt', 'r') as f:
    print(f.read())


### 没有ujson模块
不过我在相关文档找到了如何读取json
read_json
描述

根据json文件路径读取json中的各字段，主要用于读取部署在线训练平台脚本的配置文件。

语法

import libs.Utils import read_json

json_path="/sdcard/examples/test.json"
json_data=read_json(json_path)
参数
json_path

json文件在开发板中的路径
返回
dict

json字段内容



### 休眠策略

休眠时 每一秒调用一次摄像头 若检测到有人才正常开启摄像头

### 音频策略

音频是通过扬声器实现 我自购的还没到 扬声器为3525型号

### 关于日志

既然我都可以TCP和云端通信了 那肯定可能一个月就本地删除 云端实时接收
主要是云端有时候会需要查询开门日志什么的

### 开门细节

目前只提供了一个电机 因此可以做一个小demo 电机旋转一定时间后 回转确保关门

### 多线程相关

目前只提供了这个文档
多线程模块 (_thread)
_thread是MicroPython的一个基本线程模块,用于实现多线程编程

这里有几点是在使用过程中需要着重注意的

_thread 模块的多线程实现的是系统级别的多线程

这意味着程序从硬件的角度上来看其实还是单线程进行的，只是Micropython通过内部的线程调度机制

模拟的实现多线程的效果。

_thread 模块的线程调度机制是非抢占式的

这意味着你必须手动的去避免某一个线程一直占用处理器资源，通常的做法是在每个线程（如果有循环的话）

在循环的结尾处添加一个主动的延迟函数time.sleep_us(1)。只有当执行到sleep的时候，系统才会进行一次线程的调度

python的_thread模块经过了多次的更新，在3.8以前都未被正式的确定使用，

K230固件中的Micropython对_thread的使用方法并不一定完全符合最新版python中的文档

类型
LockType
线程锁类型，用于线程同步。

方法:
acquire(): 获取锁。如果锁已被其他线程持有，将阻塞直到锁被释放。
locked(): 返回锁的状态。如果锁被某个线程持有返回 True，否则返回 False。
release(): 释放锁。只能由持有锁的线程调用。
函数
allocate_lock()
创建并返回一个新的锁对象。

exit()
终止调用它的线程。

get_ident()
返回当前线程的标识符。

stack_size([size])
设置或获取新创建线程的栈大小。

start_new_thread(function, args)
启动一个新线程，执行给定的函数，并传入指定的参数元组。


###其他提示

要通过扬声器输出相关文字 肯定要PWM控制
步进电机 输入电压3.3V-28V 步进信号频率2-400Khz 步进脉冲宽度250ns 方向信号宽度62.5us
GPIO 步进电机PUL接42 DIR接33 EN接32
扬声器接35
电机参数 建议设置为可变量 可以通过以后的云相关方式热修改
音频反馈 直接读取文件输出语音
def play_audio(wav_file):
    global p
    spk.enable()  # 启用扬声器 Enable speaker
    # 有关音频流的宏变量
    SAMPLE_RATE = 24000         # 采样率24000Hz,即每秒采样24000次
    CHANNELS = 1                # 通道数 1为单声道，2为立体声
    FORMAT = paInt16            # 音频输入输出格式 paInt16
    CHUNK = int(24000*0.3)    # 每次读取音频数据的帧数，设置为0.3s的帧数24000*0.3=7200
​
    # 用于播放音频
    output_stream = p.open(format=FORMAT,channels=CHANNELS,rate=SAMPLE_RATE,output=True,frames_per_buffer=CHUNK)
    wf = wave.open(wav_file, "rb")
    wav_data = wf.read_frames(CHUNK)
    while wav_data:
        output_stream.write(wav_data)
        wav_data = wf.read_frames(CHUNK)
    time.sleep(2) # 时间缓冲，用于播放声音
    wf.close()
    output_stream.stop_stream()
    output_stream.close()
    spk.disable()  # 禁用扬声器 Disable speaker


### 架构
主程序入口：需要一个main.py作为系统启动入口，管理所有模块
模块划分：建议按功能划分成独立的.py文件
例如：core unitls mouduls（等等 你自己看着办）

### 其他可能的信息
WiFi配置：WiFi的SSID和密码需要配置文件编码
一般很多东西都需要配置文件的吧…
需要config.json统一管理所有配置
CanMV/sdcard/ 目录对应在代码里面就是 /sdcard/

### 文件储存在 C:\Python_Project\K230 Controller

请按此格式输出：

FILE: 路径
SEARCH:
<<<
原代码
>>>
REPLACE:
<<<
新代码
>>>

⚠️ 重要：
1. 创建新文件时，SEARCH的<<<和>>>之间必须有空行
2. SEARCH内容要包含足够上下文（5-10行）确保唯一
3. 如果代码重复出现，加ANCHOR_BEFORE定位
4. 如果需要删除较长的代码，可以尝试注释掉开头和结尾来实现

