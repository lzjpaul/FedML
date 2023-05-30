import fedml
from .runner import FedMLRunner

from .constants import FEDML_TRAINING_PLATFORM_CROSS_SILO

import numpy as np
import torch

def run_cross_silo_server():
    """FedML Octopus"""
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

    args = fedml.init()
    args.role = "server"

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    print ("server device: ", device)
    model = model.to(device)

    ### check model norm
    flatten_tensor = None
    for param_name, f in model.named_parameters():
        if flatten_tensor is None:
            flatten_tensor = torch.flatten(f.data.cpu())
        else:
            flatten_tensor = torch.cat((flatten_tensor, torch.flatten(f.data.cpu())))
    print ("23-5-29 print server init test flatten_tensor size: ", flatten_tensor.size())
    print ("23-5-29 print server init test flatten_tensor norm: ", torch.norm(flatten_tensor))
    ### check model norm


    # print ("server model: ", model)
    print ("server model param address")
    print(list(map(id,model.parameters())))
    for param_name, f in model.named_parameters():
        if 'weight' in param_name and 'conv' in param_name:
            print ('server param name: ', param_name)
            print ('server param norm: ', np.linalg.norm(f.data.cpu().numpy()))
    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()


def run_cross_silo_client():
    """FedML Octopus"""
    global _global_training_type
    _global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

    args = fedml.init()
    args.role = "client"

    # init device
    device = fedml.device.get_device(args)
    print ("client device: ", device)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    ### check model norm
    flatten_tensor = None
    for param_name, f in model.named_parameters():
        if flatten_tensor is None:
            flatten_tensor = torch.flatten(f.data.cpu())
        else:
            flatten_tensor = torch.cat((flatten_tensor, torch.flatten(f.data.cpu())))
    print ("23-5-29 print client init test flatten_tensor size: ", flatten_tensor.size())
    print ("23-5-29 print client init test flatten_tensor norm: ", torch.norm(flatten_tensor))
    ### check model norm

    # print ("client model: ", model)
    print ("client model param address")
    print(list(map(id,model.parameters())))
    for param_name, f in model.named_parameters():
        if 'weight' in param_name and 'conv' in param_name:
            print ('client param name: ', param_name)
            print ('client param norm: ', np.linalg.norm(f.data.cpu().numpy()))
    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
