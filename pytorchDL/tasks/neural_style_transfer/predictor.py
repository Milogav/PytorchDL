import cv2
import torch
import numpy as np

from pytorchDL.tasks.neural_style_transfer.feature_extractor import FeatureExtractor
from pytorchDL.utils.color import rgb_to_yuv, yuv_to_rgb
from pytorchDL.loggers import ProgressLogger


def compute_gram_matrix(input_tensor):
    a, b, c, d = input_tensor.size()

    features = input_tensor.view(a * b, c * d)
    gram_mat = torch.mm(features, features.t())  # compute the gram product
    return gram_mat / (a * b * c * d)


class NeuralStyleTransfer:
    """ Implementation of neural style transfer described in the paper by Gatys et al.
    'A Neural Algorithm of Artistic Style' https://arxiv.org/abs/1508.06576
    """

    def __init__(self, device=None, initialization_mode='content_image', color_mode='content_image', lr=0.05, content_weight=1.0, style_weight=1e05, tv_weight=1e-06):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if device == 'gpu' else 'cpu')

        self.feature_extractor = FeatureExtractor(device=device)
        if initialization_mode not in ['content_image', 'random']:
            raise Exception('Input initialization mode not valid. Options: "content_image", "random"')

        if color_mode not in ['content_image', 'free']:
            raise Exception('Input color mode not valid. Options: "preserve", "free"')

        self.cfg = {'init_mode': initialization_mode,
                    'color_mode': color_mode,
                    'lr': lr,
                    'content_weight': content_weight,
                    'style_weight': style_weight,
                    'total_variation_weight': tv_weight}

        self.output_result = None

    def _preproc_inputs(self, content_img, style_img, output_shape=None):
        if output_shape is None:
            out_h, out_w = content_img.shape[0:2]
        else:
            out_h, out_w = output_shape[0:2]
            content_img = cv2.resize(content_img, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        content_img = content_img[..., ::-1].astype(np.float32) / 255.0
        content_yuv = rgb_to_yuv(content_img)

        style_img = cv2.resize(style_img, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        style_img = style_img[..., ::-1].astype(np.float32) / 255.0

        content_tensor = torch.from_numpy(content_img[None]).to(self.device).permute(0, 3, 1, 2)
        style_tensor = torch.from_numpy(style_img[None]).to(self.device).permute(0, 3, 1, 2)

        return content_tensor, style_tensor, content_yuv

    def compute_losses(self, output_tensor, target_content_features, target_style_features):
        content_loss = torch.zeros(1, device=self.device)
        style_loss = torch.zeros(1, device=self.device)

        out_content_features, out_style_features = self.feature_extractor(output_tensor)

        n_content = len(out_content_features)
        n_style = len(out_style_features)

        # content loss
        for out_cf, target_cf in zip(out_content_features, target_content_features):
            content_loss += torch.nn.functional.mse_loss(out_cf, target_cf)
        content_loss /= n_content

        # style_loss
        for out_sf, target_sf in zip(out_style_features, target_style_features):
            out_gram = compute_gram_matrix(out_sf)
            target_gram = compute_gram_matrix(target_sf)
            style_loss += torch.nn.functional.mse_loss(out_gram, target_gram)
        style_loss /= n_style

        # total variation regularization
        tv_loss = torch.sum(torch.abs(output_tensor[..., :-1] - output_tensor[..., 1:])) + \
                  torch.sum(torch.abs(output_tensor[:, :, :-1, :] - output_tensor[:, :, 1:, :]))

        content_loss *= self.cfg['content_weight']
        style_loss *= self.cfg['style_weight']
        tv_loss *= self.cfg['total_variation_weight']
        total_loss = content_loss + style_loss + tv_loss
        return total_loss, content_loss, style_loss, tv_loss

    def run(self, content_img, style_img, steps, output_shape=None):

        cv2.imshow('Content image', content_img)
        cv2.imshow('Style image', style_img)

        content_tensor, style_tensor, content_yuv = self._preproc_inputs(content_img, style_img, output_shape)

        target_content_features, _ = self.feature_extractor(content_tensor)
        _, target_style_features = self.feature_extractor(style_tensor)

        target_content_features = [x.detach() for x in target_content_features]
        target_style_features = [x.detach() for x in target_style_features]

        if self.cfg['init_mode'] == 'content_image':
            output_tensor = content_tensor.clone()
        else:
            output_tensor = torch.randn(content_tensor.size(), device=self.device)

        output_tensor.requires_grad_()
        optimizer = torch.optim.Adam([output_tensor], lr=self.cfg['lr'])

        prog_bar = ProgressLogger(total_steps=steps, description='Neural style transfer')
        prog_bar.prepend_text('\nConfig:\n%s\n\n' % self.cfg)
        for step in range(steps):

            optimizer.zero_grad()

            total_loss, c_loss, s_loss, tv_loss = self.compute_losses(output_tensor,
                                                                      target_content_features,
                                                                      target_style_features)

            total_loss.backward()
            optimizer.step()

            prog_bar.log(total_loss=total_loss.item(), content_loss=c_loss.item(),
                         style_loss=s_loss.item(), tv_loss=tv_loss.item())

            # display current results
            out_img = output_tensor.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
            out_img = np.clip(out_img, 0.0, 1.0)

            if self.cfg['color_mode'] == 'content_image':
                out_img_yuv = rgb_to_yuv(out_img)
                out_img = np.stack((out_img_yuv[..., 0], content_yuv[..., 1], content_yuv[..., 2]), axis=2)
                out_img = yuv_to_rgb(out_img)

            out_img = out_img[..., ::-1] * 255.0
            self.output_result = out_img.astype(np.uint8)
            cv2.imshow('Output image', self.output_result)
            cv2.waitKey(20)

        cv2.destroyAllWindows()
