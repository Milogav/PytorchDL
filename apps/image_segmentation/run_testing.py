import argparse

from pytorchDL.tasks.image_segmentation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testDataDir', type=str, required=True, help='Directory containing the test data')
    parser.add_argument('--ckptPath', type=str, required=True, help='Path to checkpoint to evaluate')
    parser.add_argument('--device', type=str, required=False, default='cpu', choices=['cpu', 'gpu'],
                        help='Device in which to run evaluation')

    parser.add_argument('--outDir', type=str, required=True, help='Directory to store the test results')

    parser.add_argument('--batchSize', type=int, required=False, default=1, help='Batch size')
    parser.add_argument('-n', '--numProc', type=int, required=False, default=4,
                        help='Number of parallel processes for data loading')

    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = Evaluator(test_data_dir=args.testDataDir,
                          out_dir=args.outDir,
                          ckpt_path=args.ckptPath,
                          batch_size=args.batchSize,
                          device=args.device,
                          num_proc=args.numProc)
    evaluator.run_testing()


if __name__ == '__main__':
    main()
