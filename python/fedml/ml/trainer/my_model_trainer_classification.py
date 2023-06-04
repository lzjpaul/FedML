import torch
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer
from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
import logging
import copy
import logging
import numpy as np

from collections import OrderedDict
# from functorch import grad_and_value, make_functional, vmap


class ModelTrainerCLS(ClientTrainer):
    def get_model_params(self):
        print ("test fedml get_model_params")
        return self.model.cpu().state_dict()

    ### client 1: def get_model_grads(self)
    # def get_model_grads(self):
    #     model_grads_dict = OrderedDict()
    #     for param_name, f in self.model.named_parameters():
    #         model_grads_dict[param_name] = f.grad.data.cpu()
    #     return model_grads_dict

    def get_model_grads_origin(self):
        # model_grads_dict = OrderedDict()
        # for param_name, f in self.model.named_parameters():
        # print ("self.model.model_weight_update_dict: ", self.model.model_weight_update_dict)
        # for param_key in self.model.model_weight_update_dict.keys():
        #     model_grads_dict[param_key] = self.model.model_weight_update_dict[param_key]
        # return model_grads_dict
        flatten_tensor = None
        for param_key in self.model.model_weight_update_dict.keys():
            if flatten_tensor is None:
                flatten_tensor = torch.flatten(self.model.model_weight_update_dict[param_key])
            else:
                flatten_tensor = torch.cat((flatten_tensor, torch.flatten(self.model.model_weight_update_dict[param_key])))
        flatten_tensor_norm = torch.norm(flatten_tensor)
        print ("23-6-3 test print get_model_grads_origin grad flatten_tensor[:10]: ", flatten_tensor[:10])
        print ("23-6-3 test print get_model_grads_origin max: ", torch.max(flatten_tensor))
        print ("23-6-3 test print get_model_grads_origin min: ", torch.min(flatten_tensor))
        print ("23-6-3 test print get_model_grads_origin flatten_tensor shape: ", flatten_tensor.shape)
        print ("23-6-3 test print get_model_grads_origin flatten_tensor_norm: ", flatten_tensor_norm)
        return self.model.model_weight_update_dict

    def get_model_grads(self, param_bound):
        eps=1e-8
        model_grads_dict = OrderedDict()
        flatten_tensor = None
        # for param_name, f in self.model.named_parameters():
        for param_key in self.model.model_weight_update_dict.keys():
            if flatten_tensor is None:
                flatten_tensor = torch.flatten(self.model.model_weight_update_dict[param_key])
            else:
                flatten_tensor = torch.cat((flatten_tensor, torch.flatten(self.model.model_weight_update_dict[param_key])))
        flatten_tensor_norm = torch.norm(flatten_tensor)
        print ("23-6-2 test print param_bound: ", param_bound)
        print ("23-6-2 test print before scale grad flatten_tensor[:10]: ", flatten_tensor[:10])
        print ("23-6-2 test print max: ", torch.max(flatten_tensor))
        print ("23-6-2 test print min: ", torch.min(flatten_tensor))
        print ("23-6-2 test print flatten_tensor shape: ", flatten_tensor.shape)
        print ("23-6-2 test print flatten_tensor_norm: ", flatten_tensor_norm)
        # for param_name, f in self.model.named_parameters():
        for param_key in self.model.model_weight_update_dict.keys():
            model_grads_dict[param_key] = self.model.model_weight_update_dict[param_key] * (param_bound / (eps + flatten_tensor_norm))
        return model_grads_dict

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        ### in order for weight updates
        model_origin_param_dict = model.cpu().state_dict()

        model.to(device)
        print ("test fedml model.train()")
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        # model_origin_param_dict = OrderedDict()
        # for param_name, f in self.model.named_parameters():
        #     model_origin_param_dict[param_name] = f.data.cpu()

        epoch_loss = []
        ### client -1: print model before updating
        print ("before client train epochs")
        for param_name, f in model.named_parameters():
            if 'weight' in param_name and 'conv1' in param_name and 'layer1' in param_name:
                print ('param name: ', param_name)
                print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
        ### client 0: I only need one step?? not one epoch!!!
        for epoch in range(args.epochs):
            batch_loss = []

            for batch_idx, (x, labels) in enumerate(train_data):
                # print ("training batch_idx: ", batch_idx)
                if batch_idx % 10 == 0:
                    print ("training batch_idx: ", batch_idx)
                x, labels = x.to(device), labels.to(device)
                # print ("x shape: ", x.shape)
                # print ("labels: ", labels)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )

                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

        print ("after client train epochs")
        for param_name, f in model.named_parameters():
            if 'weight' in param_name and 'conv1' in param_name and 'layer1' in param_name:
                print ('param name: ', param_name)
                print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
        # weight update dictionary
        model_updated_param_dict = model.cpu().state_dict()
        # model_updated_param_dict = OrderedDict()
        # for param_name, f in self.model.named_parameters():
        #     model_updated_param_dict[param_name] = f.data.cpu()
        model.model_weight_update_dict = OrderedDict()
        for param_key in model_updated_param_dict.keys():
            model.model_weight_update_dict[param_key] = model_updated_param_dict[param_key] - model_origin_param_dict[param_key]


    def train_iterations(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        current_steps = 0
        current_epoch = 0
        while current_steps < args.local_iterations:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
                current_steps += 1
                if current_steps == args.local_iterations:
                    break
            current_epoch += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                target = target.long()
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics
