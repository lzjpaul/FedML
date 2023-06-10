import di_zkp_interface
import os
import base64
# import fedml

# initialize parameters
num_clients = 3
max_malicious_clients = 1
dim = 2
num_blinds_per_weight_key = 2
random_normal_bit_shifter = 24
linear_comb_bound_bits = 44
max_bound_sq_bits = 100
norm_bound = 2

# different combinations of (num_norm_bound_samples, standard_deviation_factor)
#   for different probabilities of malicious detection
num_norm_bound_samples = 1000
standard_deviation_factor = 15.69132269

# num_norm_bound_samples = 3000
# standard_deviation_factor = 14.5520977546

# num_norm_bound_samples = 9000
# standard_deviation_factor = 13.9108065055

weight_updates_collection = [[0, 0], [0, 1], [0.3, 0.4], [0.6, -0.9]]

# non-private protocols
protocol_type = di_zkp_interface.PROTOCOL_TYPE_NON_PRIV_INT         # weight updates rounded to integers
# protocol_type = di_zkp_interface.PROTOCOL_TYPE_NON_PRIV_FLOAT     # use float32 weight updates

# if use di_zkp_interface.PROTOCOL_TYPE_NON_PRIV_INT, rounded weight updates to integers of bit-length weight_bits
weight_bits = 16

# initialize server
server = di_zkp_interface.ServerInterface(
    num_clients, max_malicious_clients, dim, num_blinds_per_weight_key,
    weight_bits, random_normal_bit_shifter,
    num_norm_bound_samples, linear_comb_bound_bits, max_bound_sq_bits,
    False, protocol_type)

server.initialize_new_iteration(norm_bound, standard_deviation_factor)

print("server.dim = " + str(server.dim))
print("server.weight_bits = " + str(server.weight_bits))

# initialize clients
clients = []
for i in range(num_clients+1):
    client = di_zkp_interface.ClientInterface(
        num_clients, max_malicious_clients, dim, num_blinds_per_weight_key,
        weight_bits, random_normal_bit_shifter,
        num_norm_bound_samples, linear_comb_bound_bits, max_bound_sq_bits,
        i,
        False, protocol_type)
    # print("client.client_id = " + str(client.client_id))
    clients.append(client)

print("clients[0].dim = " + str(clients[0].dim))
print("clients[0].client_id = " + str(clients[0].client_id))

client_ids = range(1, num_clients+1)
print(client_ids)

# step 1: clients send messages to server
for i in client_ids:
    weights_i = weight_updates_collection[i]
    weights = di_zkp_interface.VecFloat(len(weights_i))
    # weights = weights_i
    print ("type(weights): ", type(weights))
    for j in range(len(weights_i)):
        print ("type(weights_i): ", type(weights_i))
        print ("weights_i: ", weights_i)
        weights[j] = weights_i[j]
    # print(weights)
    # print(type(weights))
    server.receive_1(clients[i].send_1(norm_bound, standard_deviation_factor,
                                       weights), i)

server.finish_iteration()
print("***** iteration finished *****")
# finish one iteration, aggregate sum is in server.final_update_float
#   aggregate average is in server.final_update_float_avg

print ("server.final_update_float_avg: \n", server.final_update_float_avg)
print ("server.final_update_float_avg[0:2]: \n", server.final_update_float_avg[0:2])
print ("server.final_update_float_avg[0]: \n", server.final_update_float_avg[0])
col_sums = [sum(x) for x in zip(*weight_updates_collection)]
for j in range(dim):
    assert (abs(num_clients * server.final_update_float_avg[j] - col_sums[j]) < 1e-4)

print("test python interface success")
