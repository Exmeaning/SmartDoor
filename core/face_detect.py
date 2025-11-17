from libs.PipeLine import ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import aidemo
import gc
from utils.logger import get_logger
from utils.config_loader import ConfigLoader

class FaceDetector(AIBase):
    """人脸检测器 / Face Detector
    
    优化版本，支持休眠模式和低功耗检测
    """
    
    def __init__(self, sleep_manager=None):
        self.config = ConfigLoader()
        self.logger = get_logger()
        self.sleep_manager = sleep_manager
        
        # 加载配置
        kmodel_path = self.config.get('face_recognition.det_model_path')
        anchors_path = self.config.get('face_recognition.anchors_path')
        model_input_size = [320, 320]
        confidence_threshold = self.config.get('face_recognition.confidence_threshold', 0.5)
        nms_threshold = self.config.get('face_recognition.nms_threshold', 0.2)
        rgb888p_size = self.config.get('display.rgb888p_size', [640, 480])
        display_size = self.config.get('display.display_size', [640, 480])
        
        # 加载锚框数据
        anchors = np.fromfile(anchors_path, dtype=np.float)
        anchors = anchors.reshape((4200, 4))
        
        # 调用父类构造函数
        super().__init__(kmodel_path, model_input_size, rgb888p_size, 0)
        
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        
        # 初始化AI2D
        self.ai2d = Ai2d(0)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                nn.ai2d_format.NCHW_FMT,
                                np.uint8, np.uint8)
        
        # 检测统计
        self.detection_count = 0
        self.no_face_count = 0
        self.max_no_face_count = 30  # 连续30次无人脸则进入休眠
        
        self.config_preprocess()
    
    def config_preprocess(self, input_image_size=None):
        """配置预处理 / Configure preprocessing"""
        with ScopedTiming("set preprocess config", False):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            
            # 配置padding和resize
            self.ai2d.pad(self.get_pad_param(), 0, [104, 117, 123])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            
            # 构建预处理pipeline
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                           [1, 3, self.model_input_size[1], self.model_input_size[0]])
    
    def postprocess(self, results):
        """后处理 / Post-processing"""
        with ScopedTiming("postprocess", False):
            res = aidemo.face_det_post_process(
                self.confidence_threshold,
                self.nms_threshold,
                self.model_input_size[0],
                self.anchors,
                self.rgb888p_size,
                results
            )
            
            # 更新检测统计
            if res and len(res) > 0:
                self.detection_count += 1
                self.no_face_count = 0
                
                # 更新活动时间
                if self.sleep_manager:
                    self.sleep_manager.update_activity()
                
                return res[0] if len(res) == 1 else (res[0], res[1])
            else:
                self.no_face_count += 1
                
                # 检查是否需要进入休眠
                if self.no_face_count >= self.max_no_face_count:
                    if self.sleep_manager:
                        self.sleep_manager.check_sleep_condition()
                
                return res
    
    def get_pad_param(self):
        """计算padding参数 / Calculate padding parameters"""
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        ratio = min(ratio_w, ratio_h)
        
        new_w = int(ratio * self.rgb888p_size[0])
        new_h = int(ratio * self.rgb888p_size[1])
        
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        
        return [0, 0, 0, 0,
                int(round(0)),
                int(round(dh * 2 + 0.1)),
                int(round(0)),
                int(round(dw * 2 - 0.1))]
    
    def detect_faces(self, image):
        """检测人脸 / Detect faces"""
        try:
            result = self.run(image)
            
            if result:
                self.logger.debug(f"检测到 {len(result)} 张人脸")
            
            return result
            
        except Exception as e:
            self.logger.error(f"人脸检测失败: {e}")
            return None
    
    def draw_result(self, pl, dets):
        """绘制检测结果 / Draw detection results"""
        if dets:
            pl.osd_img.clear()
            for det in dets:
                x, y, w, h = map(lambda x: int(round(x, 0)), det[:4])
                x = x * self.display_size[0] // self.rgb888p_size[0]
                y = y * self.display_size[1] // self.rgb888p_size[1]
                w = w * self.display_size[0] // self.rgb888p_size[0]
                h = h * self.display_size[1] // self.rgb888p_size[1]
                
                pl.osd_img.draw_rectangle(x, y, w, h, color=(255, 255, 0, 255), thickness=2)
        else:
            pl.osd_img.clear()
    
    def get_statistics(self):
        """获取检测统计 / Get detection statistics"""
        return {
            "total_detections": self.detection_count,
            "no_face_count": self.no_face_count,
            "is_face_present": self.no_face_count == 0
        }