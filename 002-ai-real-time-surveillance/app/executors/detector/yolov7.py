from typing import Tuple, Union

import numpy as np
import torch

from .models import attempt_load
from .utils import check_img_size, letterbox, non_max_suppression, scale_coords


class ObjectDetector:
    """
    YOLOv7 detector wrapper for loading a model and performing object detection.
    """

    def __init__(self, config: dict) -> None:
        self.conf_th = config["CONF_TH"]
        self.iou_th = config["IOU_TH"]
        self.device = config["DEVICE"]
        self.config = config
        self.model = self.load()
        self.stride = None
        self.img_size = None

    def load(self, img_size: int = 640) -> None:
        """Load the YOLOv7 model from a checkpoint file."""
        self.model = attempt_load(self.config["CKPT_PATH"], map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, self.stride)
        self.model.eval()

    def preprocess(self, img: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocess the input image for detection."""
        resized_img = letterbox(img, self.img_size, stride=self.stride)[0]
        resized_img = resized_img[:, :, ::-1].transpose(
            2, 0, 1
        )  # BGR to RGB, to 3x416x416
        resized_img = np.ascontiguousarray(resized_img)
        tensor_img = torch.from_numpy(resized_img).to(self.device).float() / 255.0
        return tensor_img, img

    @staticmethod
    def xyxy2xywh(
        x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]."""
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @torch.inference_mode()
    def detect(
        self, image: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform object detection on the input image."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() method first.")

        img, orig = self.preprocess(image)
        img = img.unsqueeze(0)

        pred = self.model(img)
        dets = non_max_suppression(pred[0], self.conf_th, self.iou_th, classes=[0])[0]
        dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], orig.shape).round()

        xywhs = self.xyxy2xywh(dets[:, 0:4])
        confs = dets[:, 4]
        clss = dets[:, 5]

        return xywhs.cpu(), confs.cpu(), clss.cpu()

    def __del__(self):
        if self.model is not None:
            self.model.cpu()
