import json
import os

import numpy as np
import wget
from ...ml.engine import ml_engine_adapter

cwd = os.getcwd()

import zipfile

from ...constants import FEDML_DATA_MNIST_URL
import logging


def download_mnist(data_cache_dir):
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)

    file_path = os.path.join(data_cache_dir, "MNIST.zip")
    logging.info(file_path)

    # Download the file (if we haven't already)
    print ('file_path: ', file_path)
    print ("os.path.exists(file_path): ", os.path.exists(file_path))
    if not os.path.exists(file_path):
        wget.download(FEDML_DATA_MNIST_URL, out=file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_cache_dir)


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    print ("train_files: ", train_files)
    train_files = [f for f in train_files if f.endswith(".json")]
    print ("train_files: ", train_files)
    for f in train_files:
        print ("train_data_dir: ", train_data_dir)
        print ("f: ", f)
        file_path = os.path.join(train_data_dir, f)
        print ("file_path: ", file_path)
        print ("begin cdata = json.load(inf)")
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        print ("after cdata = json.load(inf)")
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        train_data.update(cdata["user_data"])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        test_data.update(cdata["user_data"])

    clients = sorted(cdata["users"])

    return clients, groups, train_data, test_data


def batch_data(args, data, batch_size):

    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data["x"]
    data_y = data["y"]

    # print ("len(data_x): ", len(data_x))
    # print ("len(data_x[0]): ", len(data_x[0]))
    # print ("len(data_y): ", len(data_y))
    # print ("len(data_y[0]): ", len(data_y[0]))

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i : i + batch_size]
        batched_y = data_y[i : i + batch_size]
        batched_x, batched_y = ml_engine_adapter.convert_numpy_to_ml_engine_data_format(args, batched_x, batched_y)
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_mnist_by_device_id(batch_size, device_id, train_path="MNIST_mobile", test_path="MNIST_mobile"):
    train_path += os.path.join("/", device_id, "train")
    test_path += os.path.join("/", device_id, "test")
    return load_partition_data_mnist(batch_size, train_path, test_path)

"""
def load_partition_data_mnist(
    args, batch_size, train_path=os.path.join(os.getcwd(), "MNIST", "train"),
        test_path=os.path.join(os.getcwd(), "MNIST", "test")
):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]["x"])
        user_test_data_num = len(test_data[u]["x"])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        print ("u: ", u)
        print ("g: ", g)
        print ("user_train_data_num: ", user_train_data_num)
        print ("train_data_num: ", train_data_num)
        print ("user_test_data_num: ", user_test_data_num)
        print ("test_data_num: ", test_data_num)
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(args, train_data[u], batch_size)
        test_batch = batch_data(args, test_data[u], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 10

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
"""

"""
def load_partition_data_mnist(
    args, batch_size, train_path=os.path.join(os.getcwd(), "MNIST", "train"),
        test_path=os.path.join(os.getcwd(), "MNIST", "test")
):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    client_num_in_total = args.client_num_in_total
    logging.info("loading data...")
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]["x"])
        user_test_data_num = len(test_data[u]["x"])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        # print ("u: ", u)
        # print ("g: ", g)
        # print ("user_train_data_num: ", user_train_data_num)
        # print ("train_data_num: ", train_data_num)
        # print ("user_test_data_num: ", user_test_data_num)
        # print ("test_data_num: ", test_data_num)
        if client_idx < client_num_in_total:
            train_data_local_num_dict[client_idx] = user_train_data_num
        else:
            train_data_local_num_dict[client_idx%client_num_in_total] = train_data_local_num_dict[client_idx%client_num_in_total] + user_train_data_num

        # transform to batches
        train_batch = batch_data(args, train_data[u], batch_size)
        test_batch = batch_data(args, test_data[u], batch_size)

        # index using client index
        if client_idx < client_num_in_total:
            train_data_local_dict[client_idx] = train_batch
            test_data_local_dict[client_idx] = test_batch
        else:
            train_data_local_dict[client_idx%client_num_in_total] = train_data_local_dict[client_idx%client_num_in_total] +  train_batch
            test_data_local_dict[client_idx%client_num_in_total] = test_data_local_dict[client_idx%client_num_in_total] + test_batch  
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    logging.info("finished the loading data")
    client_num = client_num_in_total
    class_num = 10

    print ("client_num: ", client_num)
    print ("train_data_num ", train_data_num)
    print ("test_data_num: ", test_data_num)
    print ("train_data_local_num_dict: \n", train_data_local_num_dict)
    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
"""

def batch_all_data(args, data_x_all, data_y_all, batch_size):

    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """

    # print ("len(data_x): ", len(data_x))
    # print ("len(data_x[0]): ", len(data_x[0]))
    # print ("len(data_y): ", len(data_y))
    # print ("len(data_y[0]): ", len(data_y[0]))

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x_all)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y_all)

    client_data_idx_list = []
    for i in range(0, len(data_x_all), int(len(data_x_all)/args.client_num_in_total)):
        client_data_idx_list.append(i)
    print ("MNIST data client_data_idx_list: ", client_data_idx_list)

    batch_data_list = []
    data_local_num_dict = {}
    for cli_idx in range(args.client_num_in_total):
        if cli_idx != (args.client_num_in_total-1):
            data_x_partial = data_x_all[client_data_idx_list[cli_idx]:client_data_idx_list[cli_idx+1]]
            data_y_partial = data_y_all[client_data_idx_list[cli_idx]:client_data_idx_list[cli_idx+1]]
            data_local_num_dict[cli_idx] = (client_data_idx_list[cli_idx+1] - client_data_idx_list[cli_idx])
        else:
            data_x_partial = data_x_all[client_data_idx_list[cli_idx]:]
            data_y_partial = data_y_all[client_data_idx_list[cli_idx]:]
            data_local_num_dict[cli_idx] = (len(data_x_all) - client_data_idx_list[cli_idx])
        batch_data = list()
        for i in range(0, len(data_x_partial), batch_size):
            batched_x = data_x_partial[i : i + batch_size]
            batched_y = data_y_partial[i : i + batch_size]
            batched_x, batched_y = ml_engine_adapter.convert_numpy_to_ml_engine_data_format(args, batched_x, batched_y)
            batch_data.append((batched_x, batched_y))
        print ("23-6-5 test data_loader cli_idx: ", cli_idx)
        print ("23-6-5 test data_loader len(batch_data): ", len(batch_data))
        batch_data_list.append(batch_data)
    return batch_data_list, data_local_num_dict


def load_partition_data_mnist(
    args, batch_size, train_path=os.path.join(os.getcwd(), "MNIST", "train"),
        test_path=os.path.join(os.getcwd(), "MNIST", "test")
):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    cli_idx = 0
    client_num_in_total = args.client_num_in_total
    logging.info("loading data...")
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]["x"])
        user_test_data_num = len(test_data[u]["x"])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        # print ("u: ", u)
        # print ("g: ", g)
        # print ("user_train_data_num: ", user_train_data_num)
        # print ("train_data_num: ", train_data_num)
        # print ("user_test_data_num: ", user_test_data_num)
        # print ("test_data_num: ", test_data_num)
        
        if cli_idx == 0:
            train_data_x_all = train_data[u]["x"]
            train_data_y_all = train_data[u]["y"]
            test_data_x_all = test_data[u]["x"]
            test_data_y_all = test_data[u]["y"]
        else:
            train_data_x_all = train_data_x_all + train_data[u]["x"]
            train_data_y_all = train_data_y_all + train_data[u]["y"]
            test_data_x_all = test_data_x_all + test_data[u]["x"]
            test_data_y_all = test_data_y_all + test_data[u]["y"]
        
        cli_idx += 1

    train_batch_list, train_data_local_num_dict = batch_all_data(args, train_data_x_all, train_data_y_all, batch_size)
    test_batch_list, test_data_local_num_dict = batch_all_data(args, test_data_x_all, test_data_y_all, batch_size)
    ### [[(batched_x, batched_y)...], [(batched_x, batched_y)...]] --> list(list((batched_x, batched_y)))

    for client_idx in range(client_num_in_total):
        # train_data_local_num_dict[client_idx] = len(train_batch_list[client_idx])
        train_data_local_dict[client_idx] = train_batch_list[client_idx]
        test_data_local_dict[client_idx] = test_batch_list[client_idx]
        print ("23-6-5 test data_loader client_idx: ", client_idx)
        print ("23-6-5 test data_loader len(train_data_local_dict[client_idx]): ", len(train_data_local_dict[client_idx]))
        if client_idx == 0:
            train_data_global = train_batch_list[client_idx].copy()
            test_data_global = test_batch_list[client_idx].copy()
        else:
            train_data_global += train_batch_list[client_idx]
            test_data_global += test_batch_list[client_idx]
    # transform to batches
    # train_batch = batch_data(args, train_data[u], batch_size)
    # test_batch = batch_data(args, test_data[u], batch_size)

    # index using client index
    # if client_idx < client_num_in_total:
    #     train_data_local_dict[client_idx] = train_batch
    #     test_data_local_dict[client_idx] = test_batch
    # else:
    #     train_data_local_dict[client_idx%client_num_in_total] = train_data_local_dict[client_idx%client_num_in_total] +  train_batch
    #     test_data_local_dict[client_idx%client_num_in_total] = test_data_local_dict[client_idx%client_num_in_total] + test_batch  
    
    # train_data_global += train_batch
    # test_data_global += test_batch
    logging.info("finished the loading data")
    
    client_num = client_num_in_total
    class_num = 10

    print ("client_num: ", client_num)
    print ("train_data_num ", train_data_num)
    print ("test_data_num: ", test_data_num)
    print ("train_data_local_num_dict: \n", train_data_local_num_dict)
    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )

