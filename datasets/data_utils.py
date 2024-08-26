import torch


def intersecting_tensor_from_non_intersecting_vec(y):
    '''convert a non intersecting community vector to an intersecting community tensor'''
    #? TESTED
    num_nodes = y.shape[0]
    num_comms = int(torch.max(y).item() + 1)
    y_intersecting = torch.zeros([num_nodes, num_comms])
    for i in range(num_nodes):
        y_intersecting[i, int(y[i].item())] = 1
    return y_intersecting.int()


