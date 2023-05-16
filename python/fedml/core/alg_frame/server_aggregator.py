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
        if FedMLDifferentialPrivacy.get_instance().is_global_dp_enabled() and FedMLDifferentialPrivacy.get_instance().is_clipping():
            raw_client_model_or_grad_list = FedMLDifferentialPrivacy.get_instance().global_clip(raw_client_model_or_grad_list)
        if FedMLAttacker.get_instance().is_reconstruct_data_attack():
            FedMLAttacker.get_instance().reconstruct_data(
                raw_client_grad_list=raw_client_model_or_grad_list,
                extra_auxiliary_info=self.get_model_params(),
            )
        if FedMLAttacker.get_instance().is_model_attack():
            raw_client_model_or_grad_list = FedMLAttacker.get_instance().attack_model(
                raw_client_grad_list=raw_client_model_or_grad_list,
                extra_auxiliary_info=self.get_model_params(),
            )
        client_idxs = [i for i in range(len(raw_client_model_or_grad_list))]
        if FedMLDefender.get_instance().is_defense_enabled():
            raw_client_model_or_grad_list = FedMLDefender.get_instance().defend_before_aggregation(
                raw_client_grad_list=raw_client_model_or_grad_list,
                extra_auxiliary_info=self.get_model_params(),
            )
            client_idxs = FedMLDefender.get_instance().get_benign_client_idxs(client_idxs=client_idxs)

        return raw_client_model_or_grad_list, client_idxs

    def aggregate(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        print ("test fedml server_aggragator.py call FedMLAggOperator.agg --> torch aggragator")
        if FedMLDefender.get_instance().is_defense_enabled():
            print ("test fedml aggregation is_defense_enabled")
            return FedMLDefender.get_instance().defend_on_aggregation(
                raw_client_grad_list=raw_client_model_or_grad_list,
                base_aggregation_func=FedMLAggOperator.agg,
                extra_auxiliary_info=self.get_model_params(),
            )
        if FedMLDifferentialPrivacy.get_instance().to_compute_params_in_aggregation_enabled():
            print ("test fedml aggregation to_compute_params_in_aggregation_enabled")
            FedMLDifferentialPrivacy.get_instance().set_params_for_dp(raw_client_model_or_grad_list)
        ### server 1: self.zkp + self.model=model + optimizer.step() --> returned avg_params!!!
        ### server 1-2: copy the model_aggregator and torch aggregator, etc functions .. --> refer to agg_operator.py: line 26 - 45
        ### local_sample_num, local_model_params = raw_grad_list[i]
        ### each i is a party: (num0, avg_params) = raw_grad_list[0]
        if self.args.privacy_optimizer == 'zkp':
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
            ### fedml_aggregator.py get_dummy_input()
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
            for param_name, f in self.model.named_parameters():
                # f.grad.data.add_(float(weightdecay), f.data)
                print ('param name: ', param_name)
                print ('param size:', f.data.size())
                # print ('param: ', f)
                print ('param grad size: ', f.grad.data.size())
                f.grad.data = avg_grads[param_name].to(self.device)
            self.optimizer.step()
            return self.model.cpu().state_dict()
        else:
            return FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)  # return averaged_params

    def on_after_aggregation(self, aggregated_model_or_grad: OrderedDict) -> OrderedDict:
        if FedMLDifferentialPrivacy.get_instance().is_global_dp_enabled():
            logging.info("-----add central DP noise ----")
            aggregated_model_or_grad = FedMLDifferentialPrivacy.get_instance().add_global_noise(
                aggregated_model_or_grad
            )
        if FedMLDefender.get_instance().is_defense_enabled():
            aggregated_model_or_grad = FedMLDefender.get_instance().defend_after_aggregation(aggregated_model_or_grad)
        return aggregated_model_or_grad

    def assess_contribution(self):
        if self.contribution_assessor_mgr is None:
            return
        # TODO: start to run contribution assessment in an independent python process
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
            self.final_contribution_assigment_dict = self.contribution_assessor_mgr.get_final_contribution_assignment()
            logging.info(
                "self.final_contribution_assigment_dict = {}".format(self.final_contribution_assigment_dict))

    @abstractmethod
    def test(self, test_data, device, args):
        pass

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        pass
