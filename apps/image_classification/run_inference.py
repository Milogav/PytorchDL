import os
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from pytorchDL.tasks.image_classification.predictor import Predictor
from pytorchDL.utils.io import batch_split
from pytorchDL.loggers import ProgressLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckptPath', type=str, required=True, help='Path to checkpoint to be used for inference')
    parser.add_argument('--device', type=str, required=False, default='cpu', choices=['cpu', 'gpu'],
                        help='Device in which to run inference')
    parser.add_argument('--batchSize', type=int, required=False, default=1, help='Batch size for class inference')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Accepted options: txt file with image paths, a directory containing images, an image file or "live" for real time inference on camera stream')
    parser.add_argument('-o', '--output', type=str, required=False,
                        help='Output CSV file to save prediction results. Not applicable of "live" input')
    parser.add_argument('-n', '--numProc', type=int, required=False, default=4,
                        help='Number of parallel processes for image loading')

    return parser.parse_args()


def main():

    args = parse_args()
    predictor = Predictor(args.ckptPath, device=args.device, num_proc=args.numProc)

    if args.input.lower() == 'live':  # real time predictions from camera
        v_cap = cv2.VideoCapture(0)
        while v_cap.isOpened():
            ret, frame = v_cap.read()
            fh, fw = frame.shape[0:2]

            preds = predictor.run(input=[frame])
            label, prob = preds[0]

            pred_str = 'Pred label: %d -- Prob: %.3f' % (label, prob)
            cv2.putText(frame, pred_str, (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Inference', frame)
            key = cv2.waitKey(25)

            if key == 27:  # press esc to stop program execution
                print('Execution manually terminated by user!!')
                v_cap.release()

    else:
        assert args.output, 'Output file must be specified when input is not "live"'

        predictions = list()
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
                preds = predictor.run(input=batch_imgs)
                predictions.append(preds)
                prog_logger.log()

        prog_logger.close()
        predictions = np.concatenate(predictions, axis=0)
        output_data = {'IMAGE': input_img_paths,
                       'PRED_CLASS_ID': predictions[:, 0],
                       'PRED_CLASS_PROB': predictions[:, 1]}

        # save output predictions to a csv file
        df = pd.DataFrame(data=output_data)
        df = df.astype({'PRED_CLASS_ID': int})

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
