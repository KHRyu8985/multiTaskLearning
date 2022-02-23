"""Docstring for mtl.py

This is the user-facing module for MTL training.
Run this module from the command line, and pass in the
appropriate arguments to define the model and data.
"""
# Run like this example: python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare --epochs 10 --experimentname MHUSHARE_ALL --device cuda:0 --scarcities 1 1 0 --savefreq 1
# ---- Jan 21 : Added Poisson 1D random sampling
# Run like this example: python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare --epochs 10 --experimentname MHUSHARE_ALL --device cuda:0 --scarcities 1 1 0 --savefreq 1 --mask_type poisson

import argparse
import json
import os
import numpy as np

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from dloader import genDataLoader
from wrappers import multi_task_trainer
from utils import label_blockstructures

"""
=========== Model ============
"""
from models_v2 import MTL_MoDL
from models import MTL_VarNet

"""
=========== command line parser ============
"""
parser = argparse.ArgumentParser(
    description = 'define parameters and roots for STL training'
)

# hyperparameters
parser.add_argument(
    '--epochs', default=100, type=int,
    help='number of epochs to run'
)
parser.add_argument(
    '--lr', default=0.0002, type=float,
    help='learning rate'
)
parser.add_argument(
    '--gradaccumulation', default=1, type=int,
    help='how many iterations per gradient accumulation; cannot be less than 1'
)
parser.add_argument(
    '--gradaverage', default=0, type=int,
    help="""if true, will average accumulated grads before optimizer.step
    if false, gradaccumulation will occur without averaging (i.e. no hooks)
    value does not matter if gradaccumulation is equal to 1"""
)


# model training

parser.add_argument(
    '--blockstructures', nargs='+',
    default=[
        'trueshare', 'mhushare', 'attenshare', 'mhushare',
        'split', 'attenshare', 'split', 'split',
        'mhushare', 'mhushare', 'attenshare', 'attenshare',
    ],
    help="""explicit list of what each block will be;
    defines total number of blocks;
    possible options = [
        trueshare, mhushare, attenshare, split
    ]
    trueshare block shares encoder and decoder;
    mhushare block shares encoder but not decoder;
    attenshare block has global shared unet, atten at all levels
    split does not share anything
    """
)

parser.add_argument(
    '--network', default='modl', type=str,
    help="""Defines types of network (varnet or modl)
    """
    )

parser.add_argument(
    '--weightsdir', default=None, type=str,
    help="""for transfer learning, give directory for loading weights;
    i.e. 'models/STL_SPLIT_modlVVVVVVVV_ankle_elbow/N=578_l1.pt'
    default to None because we're not usually doing transfer learning
    """
)

parser.add_argument(
    '--shareetas', default=1, type=int,
    help="""if false, there will be len(opt.datasets) number of etas for each unrolled block
    if true, there will be 1 eta for each unrolled block
    """
    )

parser.add_argument(
    '--temp', default=2.0, type=float, 
    help='temperature for DWA (must be positive)'
)

parser.add_argument(
    '--weighting', default='naive',
    help='naive, uncert, dwa'
)

parser.add_argument(
    '--device', nargs='+', default=['cuda:0'],
    help='cuda:0 device default'
)


# dataset properties
parser.add_argument(
    '--datasets', nargs='+',
    help='names of relevant tasks',
    required = True,
)

parser.add_argument(
    '--datadir', default='/scratch/users/calkan/datasets/MTLData',
    help='data root directory; where are datasets contained'
)

parser.add_argument(
    '--scarcities', default=[1, 2, 3], type=int, nargs='+',
    help='number of samples in second task will be decreased by 1/2^N; i.e. 1 2 3'
    )

parser.add_argument(
    '--accelerations', default=[6], type=int, nargs='+',
    help='list of undersampling factor of k-space for training; validation is average acceleration '
    )

parser.add_argument(
    '--centerfracs', default=[0.05], type=int, nargs='+',
    help='list of center fractions sampled of k-space for training; val is average centerfracs'
    )

# data loader properties
parser.add_argument(
    '--stratified', default=0, type=int,
    help="""if true, stratifies the dataloader"""
)

parser.add_argument(
    '--stratifymethod', default='upsample',
    help="""
    one of [upsample, downsample] for
    scarce, abundant dataset, respectively
    does not matter if --stratified is false"""
)

parser.add_argument(
    '--numworkers', default=16, type=int,
    help='number of workers for PyTorch dataloader'
)

# save / display data
parser.add_argument(
    '--experimentname', default='unnamed_experiment',
    help='experiment name i.e. STL or MTAN_pareto or MHU_naive_etc'
)
parser.add_argument(
    '--verbose', default=1, type=int,
    help="""if true, prints to console average loss / metrics"""
)
parser.add_argument(
    '--tensorboard', default=1, type=int,
    help='if true, creates TensorBoard'
)
parser.add_argument(
    '--savefreq', default=10, type=int,
    help='how many epochs per saved recon image'
)
parser.add_argument(
    '--mask_type', default='poisson', type=str,
    help='type of mask used for undersampling (poisson or equi)'
)

opt = parser.parse_args()

# validation of user inputs
for structure in opt.blockstructures:
    assert structure in ['trueshare', 'mhushare', 'split', 'attenshare'], \
           f'unet structure is not yet a supported block structure'
assert opt.gradaccumulation > 0; 'opt.gradaccumulation must be greater than 0'

assert opt.weighting in ['naive', 'uncert', 'dwa', 'same'], f'weighting method not yet supported'



"""
=========== Runs ============
"""

def main(opt):
    """Calls wrappers.py for training

    Creates data loaders, initializes model, and defines learning parameters.
    Trains MTL from command line.

    Parameters
    ----------
    opt : argparse.ArgumentParser
        Refer to help documentation abov
    Returns
    -------
    None

    See Also
    --------
    multi_task_trainer from wrappers.
    """
    basedirs = [
        os.path.join(opt.datadir, dataset)
        for dataset in opt.datasets]

    for scarcity in opt.scarcities:
        print(f'experiment w scarcity {scarcity}')

        if len(opt.datasets) == 1:
            scarcity_list = [scarcity] # downsample
        else:
            scarcity_list = [1, scarcity, 0] # downsample

        train_dloader = genDataLoader(
            [f'{basedir}/Train' for basedir in basedirs],
            scarcity_list, # downsample
            center_fractions = opt.centerfracs,
            accelerations = opt.accelerations,
            shuffle = True,
            num_workers= opt.numworkers,
            stratified = opt.stratified, method = opt.stratifymethod, mask_type = opt.mask_type
        )

        val_dloader = genDataLoader(
            [f'{basedir}/Val' for basedir in basedirs],
            [0 for _ in opt.datasets], # no downsampling
            center_fractions = [np.mean(opt.centerfracs)],
            accelerations = [int(np.mean(opt.accelerations))],
            shuffle = False, # no shuffling to allow visualization
            num_workers= opt.numworkers, mask_type = opt.mask_type 
        )
        print('generated dataloaders')

        # other inputs to MTL wrapper
        if opt.network == 'modl':
            network = MTL_MoDL(
                opt.datasets,
                opt.blockstructures,
                opt.shareetas,
                opt.device,
                )
        elif opt.network == 'varnet':
            network = MTL_VarNet(
                opt.datasets,
                opt.blockstructures,
                opt.shareetas,
                opt.device,
                )
        
        # load weights if doing transfer learning
        if opt.weightsdir:
            network.load_state_dict(torch.load(
                opt.weightsdir, 
                map_location = opt.device[0],
                )
            )
            print('loaded model from %s' % opt.weightsdir)

        optimizer = torch.optim.Adam(network.parameters(), lr = opt.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        print('start training')
        multi_task_trainer(
            train_dloader[0], val_dloader[0],
            train_dloader[1], val_dloader[1], # ratios dicts
            network, writer_tensorboard,
            optimizer, scheduler,
            opt,
        ) 

# <Update things to do>
# TODO: update loss after zero-padding k-space to right orientation
# TODO: calibration size fix to 0.05
# TODO: Check images (if some is wrong)

if __name__ == '__main__':
    # run / model name
    run_name = f"runs/{opt.experimentname}_" + \
        f"{'strat_' if opt.stratified else ''}" + \
        f"{opt.network}{label_blockstructures(opt.blockstructures)}_{'_'.join(opt.datasets)}/"
    model_name = f"models/{opt.experimentname}_" + \
        f"{'strat_' if opt.stratified else ''}" + \
        f"{opt.network}{label_blockstructures(opt.blockstructures)}_{'_'.join(opt.datasets)}/"
    if not os.path.isdir(model_name):
        os.makedirs(model_name)
    writer_tensorboard = SummaryWriter(log_dir = run_name)


    # write json files to models and runs directories; for future reference
    with open(
        os.path.join(run_name,'parameters.json'), 'w'
        ) as parameter_file:
        json.dump(vars(opt), parameter_file)

    with open(
        os.path.join(model_name,'parameters.json'), 'w'
        ) as parameter_file:
        json.dump(vars(opt), parameter_file)

    main(opt)
    writer_tensorboard.flush()
    writer_tensorboard.close()
