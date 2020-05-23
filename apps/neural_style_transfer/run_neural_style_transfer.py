import argparse
import traceback

import cv2
from pytorchDL.tasks.neural_style_transfer.predictor import NeuralStyleTransfer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--contentImg', type=str, required=True, help='Content image path')
    parser.add_argument('-s', '--styleImg', type=str, required=True, help='Style image path')
    parser.add_argument('-o', '--outputPath', type=str, required=True, help='Output image path')
    parser.add_argument('--maxIter',  type=int, default=1500, required=False, help='Maximum number of iterations')

    parser.add_argument('--device', type=str, default=None, required=False, choices=['gpu', 'cpu'], help='Device selection')
    parser.add_argument('--outputSize', type=str, required=False, default=None, help='Output image size -> H,W  e.g "512,512". Default is content image size')
    parser.add_argument('--initMode', type=str, default='content_image', required=False, choices=['content_image', 'random'], help='Output image initialization')
    parser.add_argument('--colorMode', type=str, default='free', required=False, choices=['content_image', 'free'], help='Keep content image color or allow free color transfer')
    parser.add_argument('-lr', '--learningRate', type=float, default=0.05, required=False, help='Learning rate')
    parser.add_argument('--contentWeight', type=float, default=1.0, required=False, help='Content loss weight')
    parser.add_argument('--styleWeight', type=float, default=1e05, required=False, help='Style loss weight')
    parser.add_argument('--totalVariationWeight', type=float, default=1e-05, required=False, help='Total variation regularization weight')

    return parser.parse_args()


def main():

    args = parse_args()
    nst = NeuralStyleTransfer(device=args.device,
                              initialization_mode=args.initMode,
                              color_mode=args.colorMode,
                              lr=args.learningRate,
                              content_weight=args.contentWeight,
                              style_weight=args.styleWeight,
                              tv_weight=args.totalVariationWeight)

    content_img = cv2.imread(args.contentImg)
    style_img = cv2.imread(args.styleImg)

    output_shape = list(map(int, args.outputSize.split(','))) if args.outputSize is not None else None

    try:
        nst.run(content_img, style_img, steps=args.maxIter, output_shape=output_shape)
        print('\n\nMaximum steps reached. Saving output to: %s\n\n' % args.outputPath)
        cv2.imwrite(args.outputPath, nst.output_result)

    except KeyboardInterrupt:
        print('\n\nExecution terminated manually by the user!!! Saving current output to: %s\n\n' % args.outputPath)
        cv2.imwrite(args.outputPath, nst.output_result)

    except Exception as error:
        print('\n\n')
        traceback.print_exc()


if __name__ == '__main__':
    main()
