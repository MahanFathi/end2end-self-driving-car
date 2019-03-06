import torch
import cv2
from model import build_model
from config import get_cfg_defaults


class Drive(object):
    def __init__(self):
        self.cfg = get_cfg_defaults()
        self.model = build_model(self.cfg)
        self.device = torch.device(self.cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.model.train(False)
        self.model.load_state_dict(torch.load(self.cfg.MODEL.WEIGHTS))

    def _preprocess_img(self, img):
        h, w = self.cfg.IMAGE.TARGET_HEIGHT, self.cfg.IMAGE.TARGET_WIDTH
        img_cropped = img[range(*self.cfg.DRIVE.IMAGE.CROP_HEIGHT), :, :]
        img_resized = cv2.resize(img_cropped, dsize=(w, h))
        return img_resized.astype('float32')

    def forward(self, image):
        # *** NOTE: Assuming input image of numpy array is in RGB ***
        image = cv2.cvtColor(image, code=cv2.COLOR_RGB2BGR)
        image = torch.from_numpy(self._preprocess_img(image))
        image = image.to(self.device)
        prediction = self.model(image.unsqueeze(0))
        return prediction.item()
