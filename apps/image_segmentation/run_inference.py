import os
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from pytorchDL.tasks.image_segmentation.predictor import Predictor
from pytorchDL.utils.io import batch_split
from pytorchDL.utils.misc import generate_random_colors
from pytorchDL.loggers import ProgressLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckptPath', type=str, required=True, help='Path to checkpoint to be used for inference')
    parser.add_argument('--device', type=str, required=False, default='cpu', choices=['cpu', 'gpu'],
                        help='Device in which to run inference')
    parser.add_argument('--batchSize', type=int, required=False, default=1, help='Batch size')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Accepted options: txt file with image paths, a directory containing images, an image file or "live" for real time inference on camera stream')
    parser.add_argument('-o', '--outDir', type=str, required=False,
                        help='Output directory to save prediction results. Not applicable of "live" input')
    parser.add_argument('-n', '--numProc', type=int, required=False, default=4,
                        help='Number of parallel processes for image loading')

    return parser.parse_args()


def main():

    args = parse_args()
    predictor = Predictor(args.ckptPath, device=args.device, num_proc=args.numProc)
    cl_colors = np.uint8(255 * generate_random_colors(n_colors=predictor.cfg['num_out_classes'], random_seed=42))

    if args.input.lower() == 'live':  # real time predictions from camera
        v_cap = cv2.VideoCapture(0)
        while v_cap.isOpened():
            ret, frame = v_cap.read()
            pred_seg = predictor.run(input=[frame])
            pred_seg_rgb = cl_colors[pred_seg]

            cv2.imshow('Inference', pred_seg_rgb)
            key = cv2.waitKey(25)

            if key == 27:  # press esc to stop program execution
                print('Execution manually terminated by user!!')
                v_cap.release()

    else:
        assert args.outDir, 'Output directory must be specified when input is not "live"'
        os.makedirs(args.outDir, exist_ok=True)

        if os.path.isfile(args.input):
            if '.txt' in args.input:  # input is a list of image paths
                with open(args.input, 'r') as fp:
                    input_img_paths = fp.read().splitlines()
            else:  # input is an image path
                input_img_paths = [args.input]

        elif os.path.isdir(args.input):  # input is a directory containing images
            input_img_paths = [os.path.join(args.input, x) for x in os.listdir(args.input)]

        input_batches = batch_split(input_img_paths, args.batchSize)
        prog_logger = ProgressLogger(total_steps=len(input_batches), description='Inference -- batch progress')

        with ProcessPoolExecutor(max_workers=args.numProc) as executor:
            for batch_paths in input_batches:
                batch_imgs = list(executor.map(cv2.imread, batch_paths))
                pred_seg_masks = predictor.run(input=batch_imgs)
                for pred_seg, in_path in zip(pred_seg_masks, batch_paths):
                    img_name = os.path.basename(in_path)
                    ext = '.' + img_name.split('.')[-1]

                    out_seg_path = os.path.join(args.outDir, img_name.replace(ext, '_infer' + ext))
                    cv2.imwrite(out_seg_path, pred_seg)

                    pred_seg_rgb = cl_colors[pred_seg]
                    out_seg_rgb_path = os.path.join(args.outDir, img_name.replace(ext, '_infer_rgb' + ext))
                    cv2.imwrite(out_seg_rgb_path, pred_seg_rgb)

                prog_logger.log()

        prog_logger.close()


if __name__ == '__main__':
    main()
