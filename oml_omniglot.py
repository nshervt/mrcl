import argparse
import logging

import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter

import datasets.datasetfactory as df
import datasets.task_sampler as ts
import configs.classification.class_parser as class_parser
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification

logger = logging.getLogger('experiment')


def main():
    p = class_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)
    verbos = True

    utils.set_seed(args['seed'])

    my_experiment = experiment(args['name'], args, "../results/", commit_changes=False, rank=0, seed=1)
    # writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    # -- Using first 963 classes of the omniglot as the meta-training set
    args['classes'] = list(range(963))
    args['traj_classes'] = list(range(int(963/2), 963))

    dataset = df.DatasetFactory.get_dataset(args['dataset'], background=True, train=True, path=args["path"], all=True)
    dataset_test = df.DatasetFactory.get_dataset(args['dataset'], background=True, train=False, path=args["path"], all=True)

    # Iterators used for evaluation todo: does this guarantee no overlap b/w train and test sets?
    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, num_workers=1)
    iterator_test = torch.utils.data.DataLoader(dataset_test, batch_size=5, shuffle=True, num_workers=1)

    sampler = ts.SamplerFactory.get_sampler(args['dataset'], args['classes'], dataset, dataset_test)

    config = mf.ModelFactory.get_model("na", args['dataset'], output_dimension=1000)

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config).to(device)

    for step in range(args['steps']):

        d_traj_iterators = []
        t1 = np.random.choice(args['traj_classes'], args['tasks'], replace=False)
        for t in t1:
            d_traj_iterators.append(sampler.sample_task([t]))  # task t from t1~[481-962] in batches of 1

        d_rand_iterator = sampler.get_complete_iterator()  # tasks [0-480] in batches of 15

        x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(d_traj_iterators, d_rand_iterator,
                                                               steps=args['update_step'], reset=not args['no_reset'])
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        accs, loss = maml(x_spt, y_spt, x_qry, y_qry, verbos=verbos)

        # -- Evaluation during training for sanity checks
        if verbos:
            if step % 40 == 5:
                print(accs[-1], step)
                logger.info('\nstep: %d \t training acc %s', step, str(accs))
            if step % 300 == 3:
                # todo: my_experiment? , iterator_test?
                utils.log_accuracy(maml, my_experiment, iterator_test, device)
                utils.log_accuracy(maml, my_experiment, iterator_train, device)
                pass


if __name__ == '__main__':

    main()
