# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
import numpy as np
import pandas as pd
import torch
import arguments
import models.data_utils.data_utils as data_utils
import models.model_utils as model_utils
from models.vrpModel import vrpModel
# import torch.nn as nn


def create_model(args):
    model = vrpModel(args)
    model = model.apply(weight_init)  # initialization the model
    # for name, param in model.named_parameters():
    # 	print(name, param.size())
    # 	print(name, param.data)

    if model.cuda_flag:
        model = model.cuda()
    model.share_memory()

    model_supervisor = model_utils.vrpSupervisor(model, args)
    print(type(model_supervisor))
    if args.load_model:
        model_supervisor.load_pretrained(args.load_model)
    elif args.resume:
        pretrained = 'ckpt-' + str(args.resume).zfill(8)
        print('Resume from {} iterations.'.format(args.resume))
        model_supervisor.load_pretrained(args.model_dir + '/' + pretrained)
    else:
        print('Created model with fresh parameters.')
        model_supervisor.model.init_weights(args.param_init)
    #

    return model_supervisor


def train(args):
    print('Training:')

    train_data = data_utils.load_dataset(args.train_dataset, args)
    train_data_size = len(train_data)
    if args.train_proportion < 1.0:
        random.shuffle(train_data)
        train_data_size = int(train_data_size * args.train_proportion)
        train_data = train_data[:train_data_size]

    eval_data = data_utils.load_dataset(args.val_dataset, args)

    DataProcessor = data_utils.vrpDataProcessor()
    model_supervisor = create_model(args)

    if args.resume:
        resume_step = True
    else:
        resume_step = False
    resume_idx = args.resume * args.batch_size

    logger = model_utils.Logger(args)
    if args.resume:
        logs = pd.read_csv("../logs/" + args.log_name)
        for index, log in logs.iterrows():
            val_summary = {'avg_reward': log['avg_reward'], 'global_step': log['global_step']}
            logger.write_summary(val_summary)

    for epoch in range(resume_idx // train_data_size, args.num_epochs):
        random.shuffle(train_data)
        for batch_idx in range(0 + resume_step * resume_idx % train_data_size, train_data_size, args.batch_size):
            resume_step = False
            print(epoch, batch_idx)
            batch_data = DataProcessor.get_batch(train_data, args.batch_size, batch_idx)
            train_loss, train_reward = model_supervisor.train(batch_data)
            print('train loss: %.4f train reward: %.4f' % (train_loss, train_reward))

            if model_supervisor.global_step % args.eval_every_n == 0:
                eval_loss, eval_reward = model_supervisor.eval(eval_data, args.output_trace_flag, args.max_eval_size)
                val_summary = {'avg_reward': eval_reward, 'global_step': model_supervisor.global_step}
                logger.write_summary(val_summary)
                model_supervisor.save_model()

            if args.lr_decay_steps and model_supervisor.global_step % args.lr_decay_steps == 0:
                model_supervisor.model.lr_decay(args.lr_decay_rate)
                if model_supervisor.model.cont_prob > 0.01:
                    model_supervisor.model.cont_prob *= 0.5


def evaluate(args):
    print('Evaluation:')

    test_data = data_utils.load_dataset(args.test_dataset, args)
    test_data_size = len(test_data)
    args.dropout_rate = 0.0

    dataProcessor = data_utils.vrpDataProcessor()
    model_supervisor = create_model(args)
    test_loss, test_reward = model_supervisor.eval(test_data, args.output_trace_flag)

    print('test loss: %.4f test reward: %.4f' % (test_loss, test_reward))


def chaos(n):
    np_chaos = np.load('lvjinhu2560.npy')
    np_chaos = np_chaos[1:n + 1]
    return np_chaos


def weight_init(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            # print('weight')
            init_weight_Linear(param)
        elif 'bias' in name:
            # print('bias')
            init_weight_bias(param)

    # if isinstance(m, nn.Conv2d):
    #     init_weight_conv2d(m.weight.data)
    #     if m.bias == None:
    #         pass
    #     else:
    #         init_weight_bias(m.bias.data)
    # elif isinstance(m, nn.BatchNorm2d):
    #     init_weight_bn(m.weight.data)
    #     init_weight_bn(m.bias.data)
    #     #print("nn.BatchNorm2d", m.weight.data)
    # elif isinstance(m, nn.Linear):
    #     #print("m.weight",m.weight.shape,"m.bias",m.bias.shape)
    #     init_weight_Linear(m.weight.data)
    #     init_weight_bias(m.bias.data)
    #     #print("nn.Linear", m.weight.data)
    # elif isinstance(m, nn.Conv1d):
    #     init_weight_conv1d(m.weight.data)
    #     init_weight_bias(m.bias.data)


def init_weight_conv1d(tensor):
    a, b, c = tensor.shape
    T = a * b * c
    data = chaos(T)
    data = data.reshape(a, b, c)
    with torch.no_grad():
        tensor_clone = tensor_data = torch.tensor(data)
        tensor.uniform_(0, 0)
        tensor += tensor + tensor_clone


def init_weight_conv2d(tensor):
    a, b, c, d = tensor.shape
    T = a * b * c * d
    data = chaos(T)
    data = data.reshape(a, b, c, d)
    with torch.no_grad():
        tensor_clone = tensor_data = torch.tensor(data)
        tensor.uniform_(0, 0)
        tensor += tensor + tensor_clone


def init_weight_Linear(tensor):
    a, b = tensor.shape
    T = a * b
    data = chaos(T)
    data = data.reshape(a, b)
    with torch.no_grad():
        tensor_clone = tensor_data = torch.tensor(data)
        tensor.uniform_(0, 0)
        tensor += tensor + tensor_clone


def init_weight_bias(tensor):
    a = tensor.tolist()
    T = len(a)
    data = chaos(T)
    data = data.reshape(T)
    with torch.no_grad():
        tensor_clone = tensor_data = torch.tensor(data)
        tensor.uniform_(0, 0)
        tensor += tensor + tensor_clone


def init_weight_bn(tensor):
    a = tensor.tolist()
    T = len(a)
    data = chaos(T)
    data = data.reshape(T)
    with torch.no_grad():
        tensor_clone = tensor_data = torch.tensor(data).cuda()
        tensor.uniform_(0, 0).cuda()
        tensor += tensor + tensor_clone.cuda()


def init_weight_(tensor):
    print("tensor shape", tensor.shape)
    with torch.no_grad():
        tensor_clone = torch.ones(tensor.size()).cuda()
        tensor.uniform_(0, 0).cuda()


if __name__ == "__main__":
    argParser = arguments.get_arg_parser("vrp")
    args = argParser.parse_args()
    args.cuda = not args.cpu and torch.cuda.is_available()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.eval:
        evaluate(args)
    else:
        train(args)
