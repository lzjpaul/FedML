import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple
from .. import Context
from ..contribution.contribution_assessor_manager import ContributionAssessorManager
from ..dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ..security.fedml_attacker import FedMLAttacker
from ..security.fedml_defender import FedMLDefender
from ...ml.aggregator.agg_operator import FedMLAggOperator
import fedml
import torch
from torch import nn
import numpy as np

class ServerAggregator(ABC):
    """Abstract base class for federated learning trainer."""

    def __init__(self, model, args):
        self.model = model
        self.id = 0
        self.args = args
        FedMLAttacker.get_instance().init(args)
        FedMLDefender.get_instance().init(args)
        FedMLDifferentialPrivacy.get_instance().init(args)
        self.contribution_assessor_mgr = ContributionAssessorManager(args)
        self.final_contribution_assigment_dict = dict()

        self.eval_data = None

        #### criterion and optimizer
        self.device = fedml.device.get_device(args)
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

    def set_id(self, aggregator_id):
        self.id = aggregator_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    def on_before_aggregation(
            self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]
    ):
        print ("23-5-23 test print enter on_before_aggregation")
        if FedMLDifferentialPrivacy.get_instance().is_global_dp_enabled() and FedMLDifferentialPrivacy.get_instance().is_clipping():
            print ("23-5-23 test print on_before_aggregation FedMLDifferentialPrivacy")
            raw_client_model_or_grad_list = FedMLDifferentialPrivacy.get_instance().global_clip(raw_client_model_or_grad_list)
        if FedMLAttacker.get_instance().is_reconstruct_data_attack():
            print ("23-5-23 test print on_before_aggregation reconstruct_data_attack")
            FedMLAttacker.get_instance().reconstruct_data(
                raw_client_grad_list=raw_client_model_or_grad_list,
                extra_auxiliary_info=self.get_model_params(),
            )
        if FedMLAttacker.get_instance().is_model_attack():
            print ("23-5-23 test print on_before_aggregation is model attack")
            raw_client_model_or_grad_list = FedMLAttacker.get_instance().attack_model(
                raw_client_grad_list=raw_client_model_or_grad_list,
                extra_auxiliary_info=self.get_model_params(),
            )
        client_idxs = [i for i in range(len(raw_client_model_or_grad_list))]
        if FedMLDefender.get_instance().is_defense_enabled():
            print ("23-5-23 test print on_before_aggregation is_defense_enabled")
            raw_client_model_or_grad_list = FedMLDefender.get_instance().defend_before_aggregation(
                raw_client_grad_list=raw_client_model_or_grad_list,
                extra_auxiliary_info=self.get_model_params(),
            )
            client_idxs = FedMLDefender.get_instance().get_benign_client_idxs(client_idxs=client_idxs)

        return raw_client_model_or_grad_list, client_idxs

    def aggregate(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        print ("23-5-23 test print enter aggregate test fedml server_aggragator.py call FedMLAggOperator.agg --> torch aggragator")
        if FedMLDefender.get_instance().is_defense_enabled():
            print ("23-5-23 test print fedml aggregation is_defense_enabled")
            return FedMLDefender.get_instance().defend_on_aggregation(
                raw_client_grad_list=raw_client_model_or_grad_list,
                base_aggregation_func=FedMLAggOperator.agg,
                extra_auxiliary_info=self.get_model_params(),
            )
        if FedMLDifferentialPrivacy.get_instance().to_compute_params_in_aggregation_enabled():
            print ("23-5-23 test print fedml aggregation to_compute_params_in_aggregation_enabled")
            FedMLDifferentialPrivacy.get_instance().set_params_for_dp(raw_client_model_or_grad_list)
        ### server 1: self.zkp + self.model=model + optimizer.step() --> returned avg_params!!!
        ### server 1-2: copy the model_aggregator and torch aggregator, etc functions .. --> refer to agg_operator.py: line 26 - 45
        ### local_sample_num, local_model_params = raw_grad_list[i]
        ### each i is a party: (num0, avg_params) = raw_grad_list[0]
        if self.args.privacy_optimizer == 'zkp':
            """ no baseline checking
            training_num = 0
            for i in range(len(raw_client_model_or_grad_list)):
                local_sample_num, local_model_grads = raw_client_model_or_grad_list[i]  # each i is a party
                training_num += local_sample_num
            (num0, avg_grads) = raw_client_model_or_grad_list[0]
            for k in avg_grads.keys():
                for i in range(0, len(raw_client_model_or_grad_list)):
                    local_sample_number, local_model_grads = raw_client_model_or_grad_list[i]
                    w = local_sample_number / training_num
                    if i == 0:
                        avg_grads[k] = local_model_grads[k] * w
                    else:
                        avg_grads[k] += local_model_grads[k] * w
            """
            ### checking
            training_num = 0
            valid_client_id_list = []
            for i in range(len(raw_client_model_or_grad_list)):  # each party --> valid party + training number
                local_sample_num, local_model_grads = raw_client_model_or_grad_list[i]  # each i is a party
                ### check party validity
                flatten_tensor = None
                for k in local_model_grads.keys():
                    if flatten_tensor is None:
                        flatten_tensor = torch.flatten(local_model_grads[k])
                    else:
                        flatten_tensor = torch.cat((flatten_tensor, torch.flatten(local_model_grads[k])))
                if self.args.check_type == 'strict':
                    flatten_tensor_norm = torch.norm(flatten_tensor)
                    if flatten_tensor_norm <= self.args.strict_bound:
                        valid_client_id_list.append(i)
                        training_num += local_sample_num
                elif self.args.check_type == 'chi_sqr':  # has the C++ implementation??
                    ### sample a gaussain the same shape as flatten_tensor
                    gaussian_vector = torch.normal(mean=torch.zeros(list(flatten_tensor.size())[0]), std=torch.ones(list(flatten_tensor.size())[0]))
                    ### dot product (flatten_tensor, gaussian_vector)
                    gaussian_product = torch.dot(flatten_tensor, gaussian_vector)
                    if gaussian_product <= self.args.chi_bound:
                        valid_client_id_list.append(i)
                        training_num += local_sample_num
                else:  # normal
                    print ("test 23-5-28 just normal no checking in server_aggragator.py")
                    valid_client_id_list.append(i)
                    training_num += local_sample_num
            print ("test 23-5-28 valid_client_id_list: ", valid_client_id_list)
            (num0, avg_grads) = raw_client_model_or_grad_list[0]
            for k in avg_grads.keys():
                for i in range(0, len(raw_client_model_or_grad_list)):
                    if i in valid_client_id_list:
                        local_sample_number, local_model_grads = raw_client_model_or_grad_list[i]
                        w = local_sample_number / training_num
                        if i == valid_client_id_list[0]:  # ???  --> lists have order?? + may have no valid parties!!!
                            avg_grads[k] = local_model_grads[k] * w
                        else:
                            avg_grads[k] += local_model_grads[k] * w 
            ### checking till here
            ### refer to fedml_aggregator.py get_dummy_input()
            if self.args.dataset == 'cifar10':
                dummy_input, dummy_label = torch.ones((1, 3, 32, 32)).to(self.device), torch.ones(1).to(self.device)
            else:
                pass
            # https://github.com/FedML-AI/FedML/blob/master/python/fedml/ml/trainer/my_model_trainer_classification.py
            # https://github.com/lzjpaul/pytorch/blob/residual-knowledge-driven/examples/residual-knowledge-driven-example-aaai-L2/train_lstm_main_hook_resreg_real_wlm_wd00001.py
            self.model.train()
            self.model.zero_grad()
            log_probs = self.model(dummy_input)
            dummy_label = dummy_label.long()
            loss = self.criterion(log_probs, dummy_label)  # pylint: disable=E1102
            loss.backward()
            # num_named_params = 0
            for param_name, f in self.model.named_parameters():
                if 'weight' in param_name and 'conv' in param_name:
                    print ('before weight update step')
                    print ('param name: ', param_name)
                    print ('param size:', f.data.size())
                    # print ('param: ', f)
                    print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
            #         print ('param grad size: ', f.grad.data.size())
            #     num_named_params = num_named_params + 1
            #     f.grad.data = avg_grads[param_name].to(self.device)
            for param_name, f in self.model.named_parameters():
                f.data = f.data + avg_grads[param_name].to(self.device)
                # f.add_(avg_grads[param_name].to(self.device))
            # print ("23-5-24 test print num_named_params: ", num_named_params)
            # print ("23-5-24 test print avg_grads keys lengths: ", len(avg_grads.keys()))
            # print ("23-5-24 test print avg_grads keys: ", avg_grads.keys())
            # print ("23-5-24 test print self.model keys lengths: ", len(self.model.cpu().state_dict().keys()))
            # print ("23-5-24 test print self.model keys: ", self.model.cpu().state_dict().keys())
            # self.optimizer.step()
            for param_name, f in self.model.named_parameters():
                # f.grad.data.add_(float(weightdecay), f.data)
                if 'weight' in param_name and 'conv' in param_name:
                    print ('after weight update step')
                    print ('param name: ', param_name)
                    print ('param size:', f.data.size())
                    # print ('param: ', f)
                    print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                    # print ('param grad size: ', f.grad.data.size())
            # model_updated_param_dict = OrderedDict()
            # for param_name, f in self.model.named_parameters():
            #     model_updated_param_dict[param_name] = f.data.cpu()
            # self.model.model_weight_update_dict = OrderedDict()
            # for param_key in model_updated_param_dict.keys():
            #     self.model.model_weight_update_dict[param_key] = model_updated_param_dict[param_key] - model_origin_param_dict[param_key]
            return self.model.cpu().state_dict()
        else:
            return FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)  # return averaged_params

    def on_after_aggregation(self, aggregated_model_or_grad: OrderedDict) -> OrderedDict:
        print ("23-5-23 test print enter on_after_aggregation")
        if FedMLDifferentialPrivacy.get_instance().is_global_dp_enabled():
            logging.info("-----add central DP noise ----")
            print ("23-5-23 test print on_after_aggregation is_global_dp_enabled")
            aggregated_model_or_grad = FedMLDifferentialPrivacy.get_instance().add_global_noise(
                aggregated_model_or_grad
            )
        if FedMLDefender.get_instance().is_defense_enabled():
            print ("23-5-23 test print on_after_aggragation is_defense-enabled")
            aggregated_model_or_grad = FedMLDefender.get_instance().defend_after_aggregation(aggregated_model_or_grad)
        return aggregated_model_or_grad

    def assess_contribution(self):
        if self.contribution_assessor_mgr is None:
            print ("23-5-23 test print assess_contribution contribution_assessor_mgr is none")
            return
        # TODO: start to run contribution assessment in an independent python process
        print ("23-5-23 test print assess_contribution contribution_assessor_mgr is not none")
        client_num_per_round = len(Context().get(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND))
        client_index_for_this_round = Context().get(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND)
        local_weights_from_clients = Context().get(Context.KEY_CLIENT_MODEL_LIST)

        metric_results_in_the_last_round = Context().get(Context.KEY_METRICS_ON_LAST_ROUND)
        (acc_on_last_round, _, _, _) = metric_results_in_the_last_round
        metric_results_on_aggregated_model = Context().get(Context.KEY_METRICS_ON_AGGREGATED_MODEL)
        (acc_on_aggregated_model, _, _, _) = metric_results_on_aggregated_model
        test_data = Context().get(Context.KEY_TEST_DATA)
        validation_func = self.test
        self.contribution_assessor_mgr.run(
            client_num_per_round,
            client_index_for_this_round,
            FedMLAggOperator.agg,
            local_weights_from_clients,
            acc_on_last_round,
            acc_on_aggregated_model,
            test_data,
            validation_func,
            self.args.device,
        )

        if self.args.round_idx == self.args.comm_round - 1:
            print ("23-5-23 test print assess_contribution self.contribution_assessor_mgr.get_final_contribution_assignment()")
            self.final_contribution_assigment_dict = self.contribution_assessor_mgr.get_final_contribution_assignment()
            logging.info(
                "self.final_contribution_assigment_dict = {}".format(self.final_contribution_assigment_dict))

    @abstractmethod
    def test(self, test_data, device, args):
        pass

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        pass
