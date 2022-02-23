"""Docstring for evaluate.py

Evaluates models on val or test dataset. 
Produces quantitative metrics and qualitative images.

Notes
-----
Every time a new model is developed in models.py,
one must manually add it to the imports

"""
import os
import argparse
from pathlib import Path
import glob

import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
import fastmri
from fastmri.data import transforms


from dloader import genDataLoader
# from evaluate import df_single_task_all_models
from utils import criterion, metrics
# from utils import plot_quadrant
# from utils import interpret_blockstructures

import metrics_v2

### add to this every time new model is trained ###
from models import MTL_VarNet
from models_v2 import MTL_MoDL

"""
=========== command line parser ============
"""

parser = argparse.ArgumentParser(
    description = 'to load the correct model and data for inference'
)

############## required ##############
# parser.add_argument(
#     '--blockstructures',
#     nargs='+',
#     help="""explicit string of what each block is in MTL network;
#     i.e. IYYIVVIYV
#     I : trueshare
#     Y : mhushare
#     V : split
#     trueshare block shares encoder and decoder;
#     mhushare block shares encoder but not decoder;
#     split does not share anything.
#     """,
#     required = True,
# )
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

############## required ##############
# parser.add_argument(
#     '--shareetas', type=int, nargs = '+',
#     help="""for each --experimentnames, did we have one or more etas?
#     1 for one eta; 0 for more than one etas
#     """,
#     required = True,
#     )
parser.add_argument(
    '--shareetas', default=1, type=int,
    help="""if false, there will be len(opt.datasets) number of etas for each unrolled block
    if true, there will be 1 eta for each unrolled block
    """
    )

############## required ##############
# parser.add_argument(
#     '--stratified', type=int, nargs='+',
#     help="""used to find the model name; give as list
#     0 for STL experiments""",
#     required = True,
# )

############## required ##############
# parser.add_argument(
#     '--scarcemax', type=int,
#     help="""497 for div_coronal_pd_fs""",
#     required = True,
# )

# parser.add_argument(
#     '--modeldir', default='/scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models',
#     help='models root directory; where are pretrained models are'
# )
############## required ##############
parser.add_argument(
    '--network', default='modl', type=str,
    help="""Defines types of network (varnet or modl)
    """
    )
parser.add_argument(
    '--weightsdir', default=None, type=str,
    help="""give directory for loading weights;
    i.e. 'models/STL_SPLIT_modlVVVVVVVV_ankle_elbow'
    """
)
############## required ##############
# parser.add_argument(
#     '--mixeddata', type=int, nargs = '+',
#     help="""If true, the model trained on mixed data;
#         almost always true except for STL trained on single task
#         give a list to match experimentnames;
#         0 for False; 1 for True""",
#     required = True,
# )

parser.add_argument(
    '--device', nargs='+', default=['cuda:0'],
    help='cuda:0 device default'
)

# dataset properties
############## required ##############
parser.add_argument(
    '--datasets', nargs='+',
    help="""names of datasets """,
    required = True
)

parser.add_argument(
    '--datadir', default='/scratch/users/calkan/datasets/MTLData',
    help='data root directory; where are datasets contained'
)

parser.add_argument(
    '--datasplit', default='Val',
    help='data split to use. One of `Train`, `Val` and `Test'
)

parser.add_argument(
    '--accelerations', default=[6], type=int, nargs='+',
    help='list of undersampling factor of k-space for training; validation is average acceleration '
    )

parser.add_argument(
    '--centerfracs', default=[0.05], type=int, nargs='+',
    help='list of center fractions sampled of k-space for training; val is average centerfracs'
    )

parser.add_argument(
    '--numworkers', default=16, type=int,
    help='number of workers for PyTorch dataloader'
)

parser.add_argument(
    '--mask_type', default='poisson', type=str,
    help='type of mask used for undersampling (poisson or equi)'
)

# plot properties
############## required ##############
parser.add_argument(
    '--resultdir',
    help='name of results directory',
    required = True,
)

# parser.add_argument(
#     '--colors', default = ['Category20c_20'], nargs = '+',
#     help="""Category20_10, Category20c_20, Paired10, Set1_8, Set2_8, Colorblind8, Category10_10
#     https://docs.bokeh.org/en/latest/docs/reference/palettes.html#bokeh-palette
#     OR
#     give a list of custom colors"""
# )

# parser.add_argument(
#     '--plotnames', nargs='+', default = ['loss', 'ssim', 'psnr', 'nrmse'],
#     help='a list of plot names for image title; DO NOT CHANGE',
# )

# parser.add_argument(
#     '--createplots', default=1, type=int,
#     help='if true, creates plots of metrics for different ratios of MRI; 0 1'
# )

# parser.add_argument(
#     '--showbaselines', default=0, type=int,
#     help="""if true, shows baselines for STL non-joint learning;
#         line at N=20 for abundant task and N=2,5,10,20 for scarce task"""
# )

# parser.add_argument(
#     '--showbest', default=0, type=int,
#     help="""if true, shows best metric / loss out of all runs; manually created"""
# )

# parser.add_argument(
#     '--bestdir', default='plots/best',
#     help="""directory of where best run summary plots are; csv's must be manually created"""
# )

# parser.add_argument(
#     '--bestruncolor', default = '#FF6C0C',
#     help='color for best run; only matters if --showbest is true; default Caltech orange',
# )

# parser.add_argument(
#     '--baselinenetwork', default = 'varnet',
#     help='varnet or unet for baselines',
# )

# parser.add_argument(
#     '--tensorboard', default=1, type=int,
#     help="""if true, creates TensorBoard of MR; 0 1
#         note: even if 1, but already has summary.csv, won't give tensorboard"""
# )

# parser.add_argument(
#     '--savefreq', default=2, type=int,
#     help='how many slices per saved image'
# )

# save / display data
############## required ##############
# parser.add_argument(
#     '--experimentnames', nargs='+',
#     help="""list of experiment names i.e. STL or MTAN_pareto etc.""",
#     required = True
    
# )

opt = parser.parse_args()

def create_df(
    network, dloader, model_filedir, resultdir, task
):
    """Calculates pandas dataframe ready

    - Loads model weights evaluates val/test data of a single task
    - Plots reconstructed images to TensorBoard (not yet)
    - Sorts df by dataset scarcity and save it to csv (not yet)
      so time-intensive calculation is only done once

    Parameters
    ----------
    the_model : model-like object
        Weights are not loaded. See the imports from models.py
    dloader : genDataLoader
        Contains slices from the specified dataset. Identical undersampling masks
    model_filedir : path object
        Directory of .pt weights for a specified network
    resultdir : path object or str
        Top directory where results are saved
    task : str
        The task name (i.e. 'wrist')
   idx_experimentname : int
        When analyzing multiple experiments, the index of the experiment
        from the list of experiments. This is for correct column naming.
    writer : TensorBoard SummaryWriter
        Contains directory to save logs.

    Returns
    -------
    df : pandas dataframe
        columns are loss, ssim, psnr, nrmse, task1, (task2)
        rows are various dataset scarcities
        The task1/task2 columns denote how many slices are in each task.
    
    Notes
    -----
    This data structure is currently only suitable for two tasks.

    """

    modelpaths = glob.glob(f"{model_filedir}/*.pt")

    column_names = [
        'l1_loss', 'ssim', 'psnr', 'nrmse', 'ssim_v2_norm', 
        'ssim_v2_unnorm', 'psnr_v2', 'psnr_v2_mag', 'nrmse_v2', 
        'model_name'
    ]

    # column dicts
    mean_column_dicts = {cn: [] for cn in column_names}
    std_column_dicts = {cn: [] for cn in column_names}

    with torch.no_grad():
        for idx_model, model_filepath in enumerate(modelpaths):

            print('\t eval on %s'%model_filepath)
            
            # create neccessary folders if needed
            parsed_filepath = model_filepath.split('models/')[1][:-6].split('/')
            result_subdir = parsed_filepath[0]
            result_fullpath = os.path.join(resultdir, result_subdir)
            exp_name = parsed_filepath[1]
            os.makedirs(result_fullpath, exist_ok=True)

            summary_file_mean = Path(
                os.path.join(
                    result_fullpath, f'summary_{task}_mean.csv'
                )
            )
            summary_file_std = Path(
                os.path.join(
                    result_fullpath, f'summary_{task}_std.csv'
                )
            )

            # load model
            network.load_state_dict(torch.load(
                model_filepath, 
                map_location = opt.device[0],
                )
            )
            network.eval()

            # iterate thru test set
            # data_batch = len(dloader[0])
            dataset = iter(dloader[0])

            metrics_dict = {cn: [] for cn in column_names[:-1]}

            for idx, eval_data in enumerate(dataset):
                kspace, mask, esp_maps, im_fs, task = eval_data
                task = task[0] # torch dataset loader returns as tuple

                kspace, mask = kspace.to(opt.device[0]), mask.to(opt.device[0])
                esp_maps, im_fs = esp_maps.to(opt.device[0]), im_fs.to(opt.device[0])

                _, im_us, _ = network(kspace, mask, esp_maps, task)

                # crop so im_us has same size as im_fs
                im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))

                # L1 loss
                loss = criterion(im_fs, im_us)
                metrics_dict[column_names[0]].append(loss.item())

                # ssim, psnr, nrmse
                for j in range(3):
                    metrics_dict[column_names[j+ 1]].append(metrics(im_fs, im_us)[j])
                
                # metrics from dl-ss-recon
                im_us = im_us.cpu()
                im_fs = im_fs.cpu()
                metrics_dict[column_names[4]].append(metrics_v2.compute_ssim(im_us, im_fs, data_range=1.,normalize=True).item())
                metrics_dict[column_names[5]].append(metrics_v2.compute_ssim(im_us, im_fs).item())
                metrics_dict[column_names[6]].append(metrics_v2.compute_psnr(im_us, im_fs).item())
                metrics_dict[column_names[7]].append(metrics_v2.compute_psnr(im_us, im_fs, magnitude=True).item())
                metrics_dict[column_names[8]].append(metrics_v2.compute_nrmse(im_us, im_fs).item())
            
            for field in column_names[:-1]:
                mean_column_dicts[field].append(np.mean(metrics_dict[field]))
                std_column_dicts[field].append(np.std(metrics_dict[field]))
            mean_column_dicts['model_name'].append(exp_name)
            std_column_dicts['model_name'].append(exp_name)
            # df_row[idx_model, 9] = exp_name

    ### create csv file  
    # df = pd.DataFrame(
    #     df_row,
    #     columns=column_names
    # )
    df_mean = pd.DataFrame(
        mean_column_dicts
    )
    df_std = pd.DataFrame(
        std_column_dicts
    )
    # df = df.sort_values(by=[sort_by])
    df_mean.to_csv(summary_file_mean)
    df_std.to_csv(summary_file_std)

    return df_mean, df_std


def save_eval_summaries(opt):
    """Iterates through datasets and experiments for evaluation.

    Calculates metrics and saves the results in a .csv file.

    Parameters
    ----------
    opt : argparse ArgumentParser
        Contains user-defined parameters. See help documentation above.

    Returns
    -------
    None
    
    """

    # do one task at a time
    for idx_dataset, dataset in enumerate(opt.datasets):
        print(f'working on {dataset}, {opt.weightsdir}')

        basedir = os.path.join(opt.datadir, dataset)

        # data loader for this one task
        dloader = genDataLoader(
            [f'{basedir}/{opt.datasplit}'],
            [0], # no downsampling
            center_fractions = [np.mean(opt.centerfracs)],
            accelerations = [int(np.mean(opt.accelerations))],
            shuffle = False,
            num_workers = opt.numworkers,
            mask_type = opt.mask_type,
            # use same mask so aliasing patterns are comparable
            use_same_mask = True, 
        )

        # other inputs to MTL wrapper
        if opt.network == 'modl':
            network = MTL_MoDL(
                opt.datasets,
                opt.blockstructures,
                opt.shareetas,
                opt.device,
                training=False
                )
        elif opt.network == 'varnet':
            network = MTL_VarNet(
                opt.datasets,
                opt.blockstructures,
                opt.shareetas,
                opt.device,
                training=False
                )

        df_mean, df_std = create_df(network, dloader, opt.weightsdir, opt.resultdir, dataset)

    return 0











"""
=========== main function ============
"""
if __name__ == '__main__':
    save_eval_summaries(opt)