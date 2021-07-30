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

    #inital_conv
    conv_weight_str = 'encoder.initial_conv.weight'
    conv_bias_str = 'encoder.initial_conv.bias'
    running_mean_str = 'encoder.initial_norm.running_mean'
    running_var_str = 'encoder.initial_norm.running_var'
    gamma_str = 'encoder.initial_norm.weight'
    beta_str = 'encoder.initial_norm.bias'
    num_batches_tracked_str = 'encoder.initial_norm.num_batches_tracked'

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

    # encode stage:
    for i in range(6):
        if i == 0:
            sub_stage = 1
        elif i==1:
            sub_stage = 2
        elif i==2:
            sub_stage = 3
        else:
            sub_stage = 4
        for j in range(sub_stage):
            for k in range(1, 3):
                conv_weight_str = 'encoder.stages.' + str(i) + '.convs.' + str(j) + '.conv'+str(k)+'.weight'
                conv_bias_str = 'encoder.stages.' + str(i) + '.convs.' + str(j) + '.conv'+str(k)+'.bias'
                running_mean_str = 'encoder.stages.' + str(i) + '.convs.' + str(j) + '.norm'+str(k)+'.running_mean'
                running_var_str = 'encoder.stages.' + str(i) + '.convs.' + str(j) + '.norm'+str(k)+'.running_var'
                gamma_str = 'encoder.stages.' + str(i) + '.convs.' + str(j) + '.norm'+str(k)+'.weight'
                beta_str = 'encoder.stages.' + str(i) + '.convs.' + str(j) + '.norm'+str(k)+'.bias'
                num_batches_tracked_str = 'encoder.stages.' + str(i) + '.convs.' + str(j) + '.norm'+str(k)+'.num_batches_tracked'

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
        if i != 0:
            conv_weight_str = 'encoder.stages.' + str(i) + '.convs.0.downsample_skip.0.weight'
            running_mean_str = 'encoder.stages.' + str(i) + '.convs.0.downsample_skip.1.running_mean'
            running_var_str = 'encoder.stages.' + str(i) + '.convs.0.downsample_skip.1.running_var'
            gamma_str = 'encoder.stages.' + str(i) + '.convs.0.downsample_skip.1.weight'
            beta_str = 'encoder.stages.' + str(i) + '.convs.0.downsample_skip.1.bias'
            num_batches_tracked_str = 'encoder.stages.' + str(i) + '.convs.0.downsample_skip.1.num_batches_tracked'

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
        # if i == 0:
        #     sub_stage = 1
        # elif i == 1:
        #     sub_stage = 2
        # elif i == 2:
        #     sub_stage = 3
        # else:
        #     sub_stage = 4
        for j in range(1):
            conv_weight_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.conv.weight'
            conv_bias_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.conv.bias'
            running_mean_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.norm.running_mean'
            running_var_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.norm.running_var'
            gamma_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.norm.weight'
            beta_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.norm.bias'
            num_batches_tracked_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.norm.num_batches_tracked'

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

            #all
            conv_weight_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.all.0.weight'
            conv_bias_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.all.0.bias'
            running_mean_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.all.2.running_mean'
            running_var_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.all.2.running_var'
            gamma_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.all.2.weight'
            beta_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.all.2.bias'
            num_batches_tracked_str = 'decoder.stages.' + str(i) + '.convs.' + str(j) + '.all.2.num_batches_tracked'

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
