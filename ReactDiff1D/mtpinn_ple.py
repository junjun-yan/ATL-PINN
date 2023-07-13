import torch
from utilities import relative_error
from atlpinns import ATLPINN_ReactDiff1D
from networks import NeuralNetwork_Soft_PLE
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
nu = 0.5
rho = 10.0
batch_size = 20000
learning_rate = 0.001

task_number = 2
expert_layers = [2] + 3 * [100]
shared_layers = [2] + 4 * [100]
tower_layers = 2 * [100] + [1]

N_init = 100
N_bound = 100
N_test = 20000
N_eqns = 10000

data_path = r'/root/ATLPINN/data/ReactDiff.h5'
result_path = r'/root/ATLPINN/result/diffreact_ple.txt'


def run(tasks):
    create_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    log_path = r"/root/ATLPINN/log/ReacDiff-ple-%d-%d-%s.log" % (tasks[0], tasks[1], create_date)

    ### get data
    dataset = Dataset1D(data_path, tasks)
    X_init, Y_init = dataset.get_init_points(N_init)
    X_l_bound, X_r_bound = dataset.get_bound_points(N_bound)
    X_test, Y_test = dataset.get_test_points(N_test)
    X_eqns, _ = dataset.get_sample_points(N_eqns)
    X, Y = dataset.get_all_points()

    ### modal and train | test
    network = NeuralNetwork_Soft_PLE(X, shared_layers, expert_layers, tower_layers, task_number, device)
    model = ATLPINN_ReactDiff1D(X_init, Y_init, X_l_bound, X_r_bound, X_test, Y_test, X_eqns, network,
                               batch_size, nu, rho, init_weight, bound_weight, log_path, learning_rate, task_number,
                               device)

    model.logging("number init: %d" % X_init.shape[0])
    model.logging("number bound: %d" % X_l_bound.shape[0])
    model.logging("number sample: %d" % X_eqns.shape[0])
    model.logging("number test: %d" % X_test.shape[0])

    # train
    model.train(adam_it=30000, lbfgs_it=0, clip_grad=False, decay_it=10000, decay_rate=0.5, print_it=10,
                evaluate_it=100, cosine_sim=False)

    # test
    Y_pred = model.predict(X.to(device))
    for task in range(task_number):
        error = relative_error(Y_pred[task], Y[task].to(device))
        model.logging(f'task {tasks[task]}, l2 related error: {error:.3e}')

        with open(result_path, 'a+') as f:
            f.write(f'task {tasks[task]}, l2 related error: {error:.3e}\n')

        # save prediction
        ple_path = "/root/ATLPINN/output/diffreact_ple_%d.npy" % (tasks[task])
        ple_data = {'u': Y_pred[task]}
        np.save(ple_path, ple_data)


for i in range(0, 100, 2):
    tasks = [i+1, i]
    run(tasks)
