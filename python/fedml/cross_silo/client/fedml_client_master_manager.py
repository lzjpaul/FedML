import json
import logging
import platform
import time

import torch.distributed as dist

from fedml import mlops
from fedml.constants import FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL
from .message_define import MyMessage
from .utils import convert_model_params_from_ddp, convert_model_params_to_ddp
from ...core.distributed.fedml_comm_manager import FedMLCommManager
from ...core.distributed.communication.message import Message
from ...core.mlops.mlops_profiler_event import MLOpsProfilerEvent

import di_zkp_interface
import os
import base64
import torch
import numpy as np

class ClientMasterManager(FedMLCommManager):
    ONLINE_STATUS_FLAG = "ONLINE"
    RUN_FINISHED_STATUS_FLAG = "FINISHED"

    def __init__(self, args, trainer_dist_adapter, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer_dist_adapter = trainer_dist_adapter
        self.args = args

        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.rank = rank
        self.client_real_ids = json.loads(args.client_id_list)
        logging.info("self.client_real_ids = {}".format(self.client_real_ids))
        # for the client, len(self.client_real_ids)==1: we only specify its client id in the list, not including others.
        self.client_real_id = self.client_real_ids[0]

        self.has_sent_online_msg = False
        self.is_inited = False

        # zkp_prob: initialize clients + protocol_type needes category
        if args.privacy_optimizer == "zkp" and args.check_type == "zkp_prob":
            if args.proto_type == 'int':
                protocol_type = di_zkp_interface.PROTOCOL_TYPE_NON_PRIV_INT
            else:  # 'float'
                protocol_type = di_zkp_interface.PROTOCOL_TYPE_NON_PRIV_FLOAT
            args.linear_comb_bound_bits = args.weight_bits + args.random_normal_bit_shifter + 4
            args.max_bound_sq_bits = 2 * (args.weight_bits + args.random_normal_bit_shifter) + 20
            self.client_instance = di_zkp_interface.ClientInterface(args.client_num_in_total, args.max_malicious_clients, args.dim, 
                    args.num_blinds_per_weight_key, args.weight_bits, args.random_normal_bit_shifter, args.num_norm_bound_samples, 
                    args.linear_comb_bound_bits, args.max_bound_sq_bits, rank, False, protocol_type)
            print ("init client_instance rank: ", rank)
            print ("init client_instance.dim = " + str(self.client_instance.dim))
            print ("init client_instance.client_id = " + str(self.client_instance.client_id))

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_message_connection_ready
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS, self.handle_message_check_status
        )

        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init)
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_receive_model_from_server,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_FINISH, self.handle_message_finish,
        )

    def handle_message_connection_ready(self, msg_params):
        if not self.has_sent_online_msg:
            self.has_sent_online_msg = True
            self.send_client_status(0)

            mlops.log_sys_perf(self.args)

    def handle_message_check_status(self, msg_params):
        self.send_client_status(0)

    def handle_message_init(self, msg_params):
        if self.is_inited:
            return

        self.is_inited = True

        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        data_silo_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        logging.info("data_silo_index = %s" % str(data_silo_index))

        # Notify MLOps with training status.
        self.report_training_status(MyMessage.MSG_MLOPS_CLIENT_STATUS_TRAINING)

        if self.args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            global_model_params = convert_model_params_to_ddp(global_model_params)
            self.sync_process_group(0, global_model_params, data_silo_index)

        self.trainer_dist_adapter.update_dataset(int(data_silo_index))
        self.trainer_dist_adapter.update_model(global_model_params)
        self.round_idx = 0

        self.__train()
        self.round_idx += 1

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            model_params = convert_model_params_to_ddp(model_params)
            self.sync_process_group(self.round_idx, model_params, client_index)

        self.trainer_dist_adapter.update_dataset(int(client_index))
        logging.info("update_dataset client_index = %s" % str(client_index))
        logging.info("current round index {}, total rounds {}".format(self.round_idx, self.num_rounds))
        self.trainer_dist_adapter.update_model(model_params)
        if self.round_idx < self.num_rounds:
            self.__train()
            self.round_idx += 1
        else:
            self.send_client_status(0, ClientMasterManager.RUN_FINISHED_STATUS_FLAG)
            mlops.log_training_finished_status()
            self.finish()

    def handle_message_finish(self, msg_params):
        logging.info(" ====================cleanup ====================")
        self.cleanup()

    def cleanup(self):
        self.send_client_status(0, ClientMasterManager.RUN_FINISHED_STATUS_FLAG)
        mlops.log_training_finished_status()
        self.finish()

    #self.send_model_to_server_zkp_prob(0, client_message, grad_shapes, local_sample_num)
    def send_model_to_server_zkp_prob(self, receive_id, client_message, grad_shapes, local_sample_num):
        tick = time.time()
        mlops.event("comm_c2s", event_started=True, event_value=str(self.round_idx))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.client_real_id, receive_id,)
        message.add_params("clientmessage", client_message)
        message.add_params("gradshapes", grad_shapes)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

        MLOpsProfilerEvent.log_to_wandb({"Communication/Send_Total": time.time() - tick})
        mlops.log_client_model_info(
            self.round_idx+1, self.num_rounds, model_url=message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL),
        )

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        tick = time.time()
        mlops.event("comm_c2s", event_started=True, event_value=str(self.round_idx))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.client_real_id, receive_id,)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

        MLOpsProfilerEvent.log_to_wandb({"Communication/Send_Total": time.time() - tick})
        mlops.log_client_model_info(
            self.round_idx+1, self.num_rounds, model_url=message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL),
        )

    def send_client_status(self, receive_id, status=ONLINE_STATUS_FLAG):
        logging.info("send_client_status")
        logging.info("self.client_real_id = {}".format(self.client_real_id))
        message = Message(MyMessage.MSG_TYPE_C2S_CLIENT_STATUS, self.client_real_id, receive_id)
        sys_name = platform.system()
        if sys_name == "Darwin":
            sys_name = "Mac"
        # Debug for simulation mobile system
        # sys_name = MyMessage.MSG_CLIENT_OS_ANDROID

        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_STATUS, status)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, sys_name)

        if hasattr(self.args, "using_mlops") and self.args.using_mlops and \
                status == ClientMasterManager.RUN_FINISHED_STATUS_FLAG:
            mlops.log_server_payload(self.args.run_id, self.client_real_id, json.dumps(message.get_params()))
        else:
            self.send_message(message)

    def report_training_status(self, status):
        mlops.log_training_status(status)

    def sync_process_group(self, round_idx, model_params=None, client_index=None, src=0):
        logging.info("sending round number to pg")
        round_number = [round_idx, model_params, client_index]
        dist.broadcast_object_list(
            round_number, src=src, group=self.trainer_dist_adapter.process_group_manager.get_process_group(),
        )
        logging.info("round number %d broadcast to process group" % round_number[0])

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)

        mlops.event("train", event_started=True, event_value=str(self.round_idx))

        ### client 4-1: judge zkp, if yes, grads, local_sample_num = self.trainer_dist_adapter.train(self.round_idx)
        if self.args.privacy_optimizer == 'zkp':
            grads, local_sample_num = self.trainer_dist_adapter.train(self.round_idx)
        else:
            weights, local_sample_num = self.trainer_dist_adapter.train(self.round_idx)

        # logging.info("debug fedml client train round number %d " % round_number[0])

        mlops.event("train", event_started=False, event_value=str(self.round_idx))

        # the current model is still DDP-wrapped under cross-silo-hi setting
        if self.args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            weights = convert_model_params_from_ddp(weights)

        ### zkp_prob: check after scale
        flatten_tensor = None
        for k in grads.keys():  # iterated the order according to inserting order
            if flatten_tensor is None:
                flatten_tensor = torch.flatten(grads[k])
            else:
                flatten_tensor = torch.cat((flatten_tensor, torch.flatten(grads[k])))
        print ("23-6-2 test print after scale and before client encode flatten_tensor norm: ", torch.norm(flatten_tensor))
        print ("23-6-2 test print max: ", torch.max(flatten_tensor))
        print ("23-6-2 test print min: ", torch.min(flatten_tensor))
        print ("23-6-2 test print flatten_tensor shape: ", flatten_tensor.shape)
        ### zkp_prob: expand grads (after bounded)
        if self.args.privacy_optimizer == "zkp" and self.args.check_type == "zkp_prob":
            flatten_tensor = None
            grad_shapes = [] # [(name, [shape]), (name, [shape]) ...]
            grad_shapes_name_validate = []
            for k in grads.keys():  # iterated the order according to inserting order
                grad_shapes_name_validate.append(k)
                if flatten_tensor is None:
                    flatten_tensor = torch.flatten(grads[k])
                    grad_shapes.append((k, list(grads[k].shape)))
                else:
                    flatten_tensor = torch.cat((flatten_tensor, torch.flatten(grads[k])))
                    grad_shapes.append((k, list(grads[k].shape)))
            print ("23-6-2 test print grad_shapes_name_validate: ", grad_shapes_name_validate)
            print ("23-6-2 test print after scale and before client encode flatten_tensor norm: ", torch.norm(flatten_tensor))
            print ("23-6-2 test print max: ", torch.max(flatten_tensor))
            print ("23-6-2 test print min: ", torch.min(flatten_tensor))
            print ("23-6-2 test print flatten_tensor shape: ", flatten_tensor.shape)
            weights_i = flatten_tensor.cpu().numpy()  # cpu and numpy of expanded grad of client i
            print ("numpy weights_i[:10]: \n", weights_i[:10])
            # out_file = 'flatten_tensor_' + str(self.rank) + '_' + str(self.round_idx) + '.npy'
            # np.save(out_file, weights_i)
            weights_i = list(weights_i)
            print ("list weights_i[:10]: \n", weights_i[:10])
            print ("len(weights_i): ", len(weights_i))
            weights_di_zkp = di_zkp_interface.VecFloat(len(weights_i))
            print ("for loop for the weights_di_zkp[j]")
            print ("weights_i[0]:", weights_i[0])
            print ("type(weights_i): ", type(weights_i))
            print ("weights_di_zkp[0]: ", weights_di_zkp[0])
            print ("type(weights_di_zkp): ", type(weights_di_zkp))
            for j in range(len(weights_i)):
                # if j % 1000 == 0:
                #     print ("j: ", j)
                weights_di_zkp[j] = float(weights_i[j])  # swig vector ...
            # print(weights)
            # print(type(weights))
            for j in range(6):
                print ("weights_di_zkp[j]: ", weights_di_zkp[j])
            client_message = self.client_instance.send_1(self.args.norm_bound, self.args.standard_deviation_factor, weights_di_zkp)
            print ("23-6-2 test print client side client_message[:10]: \n", client_message[:10])
        ### need to send both client_message and grad_shapes --> client_message is a string ...

        ### client 4-2: judge zkp, if yes, self.send_model_to_server(0, grads, local_sample_num)
        if self.args.privacy_optimizer == 'zkp':
            if self.args.check_type == "zkp_prob":  # pass both client_message and grad_shapes
                self.send_model_to_server_zkp_prob(0, client_message, grad_shapes, local_sample_num)
            else:
                self.send_model_to_server(0, grads, local_sample_num)
        else:
            self.send_model_to_server(0, weights, local_sample_num)

    def run(self):
        super().run()
