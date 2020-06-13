import torch
import numpy as np
import cv2

from pytorchDL.networks.resnet import ResNet


class Predictor:

    def __init__(self, ckpt_path, device, num_proc=1):

        self.device = torch.device('cuda' if device == 'gpu' else 'cpu')
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        self.cfg = checkpoint['trainer_cfg']

        self.model = ResNet(**self.cfg)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(self.device)
        self.model.eval()
        self.num_proc = num_proc

    def preprocess_single_image(self, img):

        h, w, ch = self.cfg['input_size']
        if ch == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ih, iw = img.shape[0:2]
        interp_method = cv2.INTER_AREA if (w < iw and h < ih) else cv2.INTER_CUBIC
        img = cv2.resize(img, (w, h), interpolation=interp_method)  # resize to network training input size
        img = img.astype(np.float32) / 255.0

        if ch == 1:
            img = img[..., None]

        return img

    def preprocess_input(self, images_bgr):

        proc_images = [self.preprocess_single_image(x) for x in images_bgr]
        img_tensor = torch.tensor(proc_images).permute(0, 3, 1, 2).to(self.device)
        return img_tensor

    def predict_on_batch(self, batch):

        pred_logits = self.model(batch)
        label_probs = torch.nn.functional.softmax(pred_logits, dim=1)
        pred_label_prob, pred_labels = torch.max(label_probs, dim=1)

        pred_labels = pred_labels.cpu().numpy()
        pred_label_prob = pred_label_prob.cpu().numpy()
        return pred_labels, pred_label_prob

    def run(self, input):
        """Runs class inference over input images

        :param input: list of BGR image arrays
        :return: a N x 2 numpy array containing the predicted label and probability for each input in each row
        """

        with torch.no_grad():
            input_tensor = self.preprocess_input(input)
            p_labels, p_probs = self.predict_on_batch(input_tensor)
        predictions = np.stack((p_labels, p_probs), axis=1)

        return predictions
