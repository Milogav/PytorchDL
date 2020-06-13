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

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help='Input image file to classify')
    input_group.add_argument('-iI', '--inputImages', type=str, help='Input text file containing a list of image paths')
    input_group.add_argument('-iV', '--inputVideo', type=str,
                             help='Input video file. If "live", compute predictions on camera input')

    parser.add_argument('--batchSize', type=int, required=False, default=1, help='Batch size for class inference')
    parser.add_argument('-o', '--output', type=str, required=False,
                        help='Output CSV file to save prediction results')

    parser.add_argument('-n', '--numProc', type=int, required=False, default=4,
                        help='Number of parallel processes for image loading')

    return parser.parse_args()


def main():
    args = parse_args()

    predictor = Predictor(args.ckptPath, device=args.device, num_proc=args.numProc)
    predictions = list()

    if args.inputVideo:

        if args.inputVideo.lower() == 'live':
            v_cap = cv2.VideoCapture(0)
        else:
            v_cap = cv2.VideoCapture(args.inputVideo)

        frame_count = 0
        while v_cap.isOpened():
            ret, frame = v_cap.read()
            fh, fw = frame.shape[0:2]

            preds = predictor.run(input=[frame])
            predictions.append(preds)
            label, prob = preds[0]

            pred_str = 'Pred label: %d -- Prob: %.3f' % (label, prob)
            cv2.putText(frame, pred_str, (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Inference', frame)
            key = cv2.waitKey(25)

            if key == 27:  # press esc to stop program execution
                print('Execution manually terminated by user!!')
                v_cap.release()
                exit(0)

            frame_count += 1

        v_cap.release()

        predictions = np.concantenate(predictions, axis=0)
        output_data = {'FRAME': np.arange(0, frame_count),
                       'PRED_CLASS_ID': predictions[:, 0],
                       'PRED_CLASS_PROB': predictions[:, 1]}

    else:
        if args.inputImages:
            with open(args.inputImages, 'r') as fp:
                input_img_paths = fp.read().splitlines()
        else:
            input_img_paths = [args.input]

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

    if args.output is not None:
        df = pd.DataFrame(data=output_data)
        df = df.astype({'PRED_CLASS_ID': int})
        df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
