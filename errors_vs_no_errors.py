import tensorflow as tf
from typing import List

# Federated averaging algorithm
def federated_averaging(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    '''Return the federated average of the listed scaled weights. This is equivalent to the scaled average of the weights.'''
    avg_grad = []
    num_clients = len(scaled_weight_list)

    # Get the average gradient across all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        layer_sum = tf.math.reduce_sum(grad_list_tuple, axis=0)
        layer_mean = layer_sum / num_clients
        avg_grad.append(layer_mean)

    return avg_grad

def fed_avg_with_errors(variance:float):
    
    def federated_averaging(scaled_weight_list: List[List[tf.Tensor]]) -> List[tf.Tensor]:
        '''Return the federated average of the listed scaled weights. This is equivalent to the scaled average of the weights.'''
        
        avg_grad: List[tf.Tensor] = []
        num_clients = len(scaled_weight_list)

        # Get the average gradient across all client gradients
        for grad_list_tuple in zip(*scaled_weight_list):
            # Sum the gradients for the current layer across all clients
            layer_sum = tf.math.reduce_sum(grad_list_tuple, axis=0) 
            layer_mean = (layer_sum / num_clients) + tf.random.normal(layer_sum.shape, mean=0.0, stddev=variance)
            # Append the averaged gradient to the list
            avg_grad.append(layer_mean)

        return avg_grad
    
    return federated_averaging


from FL_base import main
# main(federated_averaging, 'results/fedavg')
# main(fed_avg_with_errors(0), 'results/0variance')
variances_to_explore = [0.1]
for variance in variances_to_explore:
    main(fed_avg_with_errors(variance), f'results/{variance}variance')