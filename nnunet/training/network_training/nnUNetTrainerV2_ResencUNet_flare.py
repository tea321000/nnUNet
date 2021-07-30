#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import Tuple

import numpy as np
import torch
from torch import nn
# from nnunet.network_architecture.generic_modular_residual_UNet_flare_deploy import FabiansUNet_flare
from nnunet.network_architecture.generic_modular_residual_UNet_flare import FabiansUNet_flare
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.loss_functions.dice_loss import DC_and_topk_loss


class CropMultipleOutputLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(CropMultipleOutputLoss, self).__init__()
        self.weight_factors = [0.7, 0.2, 0.1]
        # print("weight_factors", self.weight_factors)
        self.loss = loss

    def forward(self, x, y):
        # print("len of output", len(x))
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l


def get_default_network_config(dim=2, dropout_p=None, nonlin="LeakyReLU", norm_type="bn"):
    """
    returns a dictionary that contains pointers to conv, nonlin and norm ops and the default kwargs I like to use
    :return:
    """
    props = {}
    if dim == 2:
        props['conv_op'] = nn.Conv2d
        props['dropout_op'] = nn.Dropout2d
    elif dim == 3:
        props['conv_op'] = nn.Conv3d
        props['dropout_op'] = nn.Dropout3d
    else:
        raise NotImplementedError

    if norm_type == "bn":
        if dim == 2:
            props['norm_op'] = nn.BatchNorm2d
        elif dim == 3:
            props['norm_op'] = nn.BatchNorm3d
        props['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}
    elif norm_type == "in":
        if dim == 2:
            props['norm_op'] = nn.InstanceNorm2d
        elif dim == 3:
            props['norm_op'] = nn.InstanceNorm3d
        props['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True, 'track_running_stats': True}
    else:
        raise NotImplementedError

    if dropout_p is None:
        props['dropout_op'] = None
        props['dropout_op_kwargs'] = {'p': 0, 'inplace': True}
    else:
        props['dropout_op_kwargs'] = {'p': dropout_p, 'inplace': True}

    props['conv_op_kwargs'] = {'stride': 1, 'dilation': 1, 'bias': True}  # kernel size will be set by network!

    if nonlin == "LeakyReLU":
        props['nonlin'] = nn.LeakyReLU
        props['nonlin_kwargs'] = {'negative_slope': 1e-2, 'inplace': True}
    elif nonlin == "ReLU":
        props['nonlin'] = nn.ReLU
        props['nonlin_kwargs'] = {'inplace': True}
    else:
        raise ValueError

    return props


class nnUNetTrainerV2_ResencUNet_flare(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = DC_and_topk_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False},
                                     {'k': 10})

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = CropMultipleOutputLoss(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        if self.threeD:
            cfg = get_default_network_config(3, None, norm_type="in")

        else:
            cfg = get_default_network_config(1, None, norm_type="in")

        stage_plans = self.plans['plans_per_stage'][self.stage]
        conv_kernel_sizes = stage_plans['conv_kernel_sizes']
        blocks_per_stage_encoder = stage_plans['num_blocks_encoder']
        blocks_per_stage_decoder = stage_plans['num_blocks_decoder']
        pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        self.network = FabiansUNet_flare(self.num_input_channels, self.base_num_features, blocks_per_stage_encoder, 2,
                                         pool_op_kernel_sizes, conv_kernel_sizes, cfg, self.num_classes,
                                         blocks_per_stage_decoder, True, False, 320, InitWeights_He(1e-2))

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        """
        net_num_pool_op_kernel_sizes is different in resunet
        """
        super().setup_DA_params()
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0))[:-1]

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.validate(self, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
                                     step_size=step_size, save_softmax=save_softmax, use_gaussian=use_gaussian,
                                     overwrite=overwrite, validation_folder_name=validation_folder_name,
                                     debug=debug, all_in_gpu=all_in_gpu,
                                     segmentation_export_kwargs=segmentation_export_kwargs,
                                     run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.decoder.deep_supervision = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring=do_mirroring,
                                                                             mirror_axes=mirror_axes,
                                                                             use_sliding_window=use_sliding_window,
                                                                             step_size=step_size,
                                                                             use_gaussian=use_gaussian,
                                                                             pad_border_mode=pad_border_mode,
                                                                             pad_kwargs=pad_kwargs,
                                                                             all_in_gpu=all_in_gpu,
                                                                             verbose=verbose,
                                                                             mixed_precision=mixed_precision)
        self.network.decoder.deep_supervision = ds
        return ret

    def run_training(self):
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = True
        ret = nnUNetTrainer.run_training(self)
        self.network.decoder.deep_supervision = ds
        return ret
