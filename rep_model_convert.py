import numpy as np
import torch
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='path to nnunet trained_models fold_x dir', required=True,
                        type=source_path)
    parser.add_argument('-o', '--output', help='checkpoint output directory', required=True, type=output_path)
    parser.add_argument('-n', '--name', help='checkpoint name', default="model_final_checkpoint.model", type=str)
    parser.add_argument('-on', '--out_name', help='output checkpoint name', default="rep_model.model", type=str)
    return parser.parse_args()


def output_path(path):
    if os.path.isdir(path):
        return path
    else:
        os.mkdir(path)
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"checkpoint output directory:{path} is not a valid path")


def source_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"nnunet trained_models fold_x dir:{path} is not a valid path")


if __name__ == '__main__':
    parse_args = parse_arguments()
    model = torch.load(os.path.join(parse_args.checkpoint, parse_args.name))
    state_dict = model['state_dict']
    for key, val in model['state_dict'].items():
        print(key, val.shape)
    # encode stage:
    for i in range(6):
        for j in range(2):
            if i < 5:
                conv_weight_str = 'conv_blocks_context.' + str(i) + '.blocks.' + str(j) + '.conv.weight'
                conv_bias_str = 'conv_blocks_context.' + str(i) + '.blocks.' + str(j) + '.conv.bias'
                running_mean_str = 'conv_blocks_context.' + str(i) + '.blocks.' + str(j) + '.instnorm.running_mean'
                running_var_str = 'conv_blocks_context.' + str(i) + '.blocks.' + str(j) + '.instnorm.running_var'
                gamma_str = 'conv_blocks_context.' + str(i) + '.blocks.' + str(j) + '.instnorm.weight'
                beta_str = 'conv_blocks_context.' + str(i) + '.blocks.' + str(j) + '.instnorm.bias'
                num_batches_tracked_str = 'conv_blocks_context.' + str(i) + '.blocks.' + str(j) + '.instnorm.num_batches_tracked'
            else:
                conv_weight_str = 'conv_blocks_context.' + str(i) + '.' + str(j) + '.blocks.0.conv.weight'
                conv_bias_str = 'conv_blocks_context.' + str(i) + '.' + str(j) + '.blocks.0.conv.bias'
                running_mean_str = 'conv_blocks_context.' + str(i) + '.' + str(j) + '.blocks.0.instnorm.running_mean'
                running_var_str = 'conv_blocks_context.' + str(i) + '.' + str(j) + '.blocks.0.instnorm.running_var'
                gamma_str = 'conv_blocks_context.' + str(i) + '.' + str(j) + '.blocks.0.instnorm.weight'
                beta_str = 'conv_blocks_context.' + str(i) + '.' + str(j) + '.blocks.0.instnorm.bias'
                num_batches_tracked_str = 'conv_blocks_context.' + str(i) + '.' + str(j) + '.blocks.0.instnorm.num_batches_tracked'

            conv_weight = state_dict[conv_weight_str].detach().cpu().numpy()
            conv_bias = state_dict[conv_bias_str].detach().cpu().numpy()
            running_mean = state_dict[running_mean_str].detach().cpu().numpy()
            running_var = state_dict[running_var_str].detach().cpu().numpy()
            gamma = state_dict[gamma_str].detach().cpu().numpy()
            beta = state_dict[beta_str].detach().cpu().numpy()
            eps = 1e-5
            std = np.sqrt(running_var + eps)
            t = gamma / std
            t = np.reshape(t, (-1, 1, 1, 1, 1))
            t = np.tile(t, (1, 1, conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]))
            state_dict[conv_weight_str] = torch.tensor(conv_weight * t)
            state_dict[conv_bias_str] = torch.tensor(conv_bias + beta - running_mean * gamma / std)
            del state_dict[gamma_str]
            del state_dict[beta_str]
            del state_dict[running_var_str]
            del state_dict[running_mean_str]
            del state_dict[num_batches_tracked_str]


    # decode stage
    for i in range(5):
        for j in range(2):
            conv_weight_str = 'conv_blocks_localization.' + str(i) + '.' + str(j) + '.blocks.0.conv.weight'
            conv_bias_str = 'conv_blocks_localization.' + str(i) + '.' + str(j) + '.blocks.0.conv.bias'
            running_mean_str = 'conv_blocks_localization.' + str(i) + '.' + str(j) + '.blocks.0.instnorm.running_mean'
            running_var_str = 'conv_blocks_localization.' + str(i) + '.' + str(j) + '.blocks.0.instnorm.running_var'
            gamma_str = 'conv_blocks_localization.' + str(i) + '.' + str(j) + '.blocks.0.instnorm.weight'
            beta_str = 'conv_blocks_localization.' + str(i) + '.' + str(j) + '.blocks.0.instnorm.bias'
            num_batches_tracked_str = 'conv_blocks_localization.' + str(i) + '.' + str(j) + '.blocks.0.instnorm.num_batches_tracked'

            conv_weight = state_dict[conv_weight_str].detach().cpu().numpy()
            conv_bias = state_dict[conv_bias_str].detach().cpu().numpy()
            running_mean = state_dict[running_mean_str].detach().cpu().numpy()
            running_var = state_dict[running_var_str].detach().cpu().numpy()
            gamma = state_dict[gamma_str].detach().cpu().numpy()
            beta = state_dict[beta_str].detach().cpu().numpy()
            eps = 1e-5
            std = np.sqrt(running_var + eps)
            t = gamma / std
            t = np.reshape(t, (-1, 1, 1, 1, 1))
            t = np.tile(t, (1, 1, conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]))
            state_dict[conv_weight_str] = torch.tensor(conv_weight * t)
            state_dict[conv_bias_str] = torch.tensor(conv_bias + beta - running_mean * gamma / std)
            del state_dict[gamma_str]
            del state_dict[beta_str]
            del state_dict[running_var_str]
            del state_dict[running_mean_str]
            del state_dict[num_batches_tracked_str]

    model['state_dict'] = state_dict
    print("after modify")
    for key,val in model['state_dict'].items():
        print(key, val.shape)
    torch.save(model, os.path.join(parse_args.output, parse_args.out_name))
