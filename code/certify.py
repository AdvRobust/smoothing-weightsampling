# coding=utf-8
from smoothing import SmoothedDetector, ImpSmooth
from time import time
import datetime


def run_certify(model, args, dataset):
    f = open(args.save_path, 'a')

    for i in range(args.start_idx, len(dataset)):
        if i % args.skip != 0:
            continue
        if i == args.nb_max:
            break
        print(i)
        (x, label) = dataset[i]
        before_time = time()
        x = x.cuda()
        pred, radius = model.certify(x, n0=args.N0, n=args.N, alpha=args.alpha, batch_size=args.batch_size)
        after_time = time()
        correct = int(pred == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(i, label, pred, radius, correct, time_elapsed), file=f, flush=True)

    f.close()


def eval_certify(detector, base, dataset, args):
    smoothed_detector = SmoothedDetector(detector, 2, args.detector_noise)

    smoothed_classifier = ImpSmooth(base, smoothed_detector, args.detector_nd, 10, args.base_noise)

    run_certify(smoothed_classifier, args, dataset)
