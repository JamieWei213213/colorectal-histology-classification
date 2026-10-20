import argparse
from datetime import datetime, timezone
from histology.experiment import Config, Experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['today', 'full'], default='today')
    parser.add_argument('--output', default=None)
    parser.add_argument('--data', default='data')
    args = parser.parse_args()
    output = args.output or ('outputs/' + args.profile + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    experiment = Experiment(Config.for_profile(args.profile), args.data, output)
    print(experiment.prepare())
    print(experiment.preflight())
    for name in ['cnn', 'resnet18']:
        experiment.train_one(name)
    print(experiment.freeze_selection())
    print(experiment.evaluate_final().to_string(index=False))
