import sys

sys.path.append("/root/ATLPINN")

import torch
from networks import NeuralNetwork_PINN
from utilities import relative_error
from pinns import PINN_Burgers1D
from dataset import Dataset1D
import numpy as np
import time

torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Use GPU: {torch.cuda.is_available()}\n")

### hypoparameters
init_weight = 100
bound_weight = 1
layers = [2] + 4 * [100] + [1]
nu = 0.01
batch_size = 20000
learning_rate = 0.001

data_path = r'/root/autodl-tmp/133178'
print_path = r'/root/ATLPINN/log4loss/diffreact_hard_50_100_increase.txt'

N_data = 100
N_init = 100
N_bound = 100
N_test = 20000
N_eqns = 10000


def run(tasks):
    create_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    log_path = "/root/tf-logs/Burgers1D-PINN-%d-%s.log4loss" % (tasks[0], create_date)
    pinn_path = "/root/ATLPINN/output/burgers_pinn_%d.npy" % (tasks[0])
    solver_path = '/root/ATLPINN/output/burgers_solver_%d.npy' % (tasks[0])

    ### get data
    dataset = Dataset1D(data_path, tasks)
    X_init, Y_init = dataset.get_init_points(N_init)
    X_l_bound, X_r_bound = dataset.get_bound_points(N_bound)
    X_eqns, _ = dataset.get_sample_points(N_eqns)
    X_test, Y_test = dataset.get_test_points(N_test)
    X, Y = dataset.get_all_points()

    ### modal and train | test
    network = NeuralNetwork_PINN(X_eqns, layers, device).to(device)
    model = PINN_Burgers1D(X_init, Y_init, X_l_bound, X_r_bound, X_test, Y_test, X_eqns, network, batch_size, nu,
                           init_weight, bound_weight, log_path, learning_rate, device)

    model.logging("number init: %d" % X_init.shape[0])
    model.logging("number bound: %d" % X_l_bound.shape[0])
    model.logging("number sample: %d" % X_eqns.shape[0])

    # train
    model.train(adam_it=30000, lbfgs_it=0, clip_grad=False, decay_it=10000, decay_rate=0.5, print_it=10,
                evaluate_it=100)

    # test
    Y_pred = model.predict(X.to(device))
    data_output = Y.to(device)
    error = relative_error(Y_pred, data_output)
    model.logging(f'final l2 related error: {error:.3e}')

    with open(print_path, 'a+') as f:
        f.write(f'task {tasks[0]}, l2 related error: {error:.3e}\n')

    # save dataset
    solver_data = {'x': X[:, 0], 't': X[:, 1], 'u': data_output}
    np.save(solver_path, solver_data)

    # save prediction
    pinn_data = {'u': Y_pred}
    np.save(pinn_path, pinn_data)


for i in range(0, 10):
    task = [i]
    run(task)
