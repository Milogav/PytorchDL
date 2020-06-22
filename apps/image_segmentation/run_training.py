import argparse
import traceback

from pytorchDL.tasks.image_segmentation.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['start', 'resume', 'test', 'debug'], help='Trainer mode')
    parser.add_argument('--trainDir', type=str, required=True, help='Folder containing train images and labels')
    parser.add_argument('--valDir', type=str, required=True, help='Folder containing val images and labels')
    parser.add_argument('--outDir', type=str, required=True, help='Output folder')

    parser.add_argument('--batchSize', type=int, required=True, help='Training batch size')
    parser.add_argument('--numClasses', type=int, required=True, help='Number of classes in data')
    parser.add_argument('--imgSize', type=str, required=True, help='Input training image size -> H,W,Ch  e.g "256,256,1" ')

    parser.add_argument('--maxEpochs', type=int, required=False, default=100, help='Maximum number of epochs to train')
    parser.add_argument('--testDir', type=str, required=False, help='Folder containing test images and labels')

    parser.add_argument('--trainStepsPerEpoch', type=int, required=False, default=-1, help='Number of steps that define one epoch. Default = -1 -->  all dataset examples')
    parser.add_argument('--logInterval', type=int, required=False, default=15, help='Log step interval. Default = log every 15 steps')
    parser.add_argument('--initLR', type=float, required=False, default=0.001)
    parser.add_argument('--classWeights', type=str, required=False, help='Weight for each class, as a comma separated strin of floats e.g "1,2,0.5,7"')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    input_size = list(map(int, args.imgSize.split(',')))

    if args.classWeights is not None:
        class_weights = tuple(map(float, args.imgSize.split(',')))
        assert len(class_weights) == args.numClasses
    else:
        class_weights = tuple([1.0] * args.numClasses)

    cfg = {'train_data_dir': args.trainDir,
           'val_data_dir': args.valDir,
           'input_size': input_size,
           'num_out_classes': args.numClasses,
           'class_weights': class_weights}

    trainer = Trainer(trainer_mode=args.mode,
                      out_dir=args.outDir,
                      batch_size=args.batchSize,
                      max_epochs=args.maxEpochs,
                      train_steps_per_epoch=args.trainStepsPerEpoch,
                      val_steps_per_epoch=-1,
                      log_interval=args.logInterval,
                      init_lr=args.initLR,
                      **cfg)

    try:
        trainer.run()
    except KeyboardInterrupt:
        print('\n\nExecution terminated manually by the user!!! Saving checkpoint...\n\n')
        trainer.save_checkpoint('checkpoint-step-%d-interrupt' % trainer.state['train_step'])
    except Exception as error:
        print('\n\n')
        traceback.print_exc()


if __name__ == '__main__':
    main()
