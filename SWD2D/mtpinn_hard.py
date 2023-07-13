import torch
from utilities import relative_error
from mtpinns import MTPINN_SWD2D
from networks import NeuralNetwork_Hard
from dataset import Dataset2D
import numpy as np
import time

torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Use GPU: {torch.cuda.is_available()}\n")

### hypoparameters
init_weight = 70
bound_weight = 200
batch_size = 20000
learning_rate = 0.001

task_number = 2
shared_layers = [3] + 5 * [100]
special_layers = 2 * [100] + [3]

data_path = r'/root/ATLPINN/data/SWD.h5'
result_path = r'/root/ATLPINN/result/swd2d_hard.txt'

N_init = 1000
N_bound = 1000
N_test = 20000
N_eqns = 100000


def run(tasks):
    create_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    log_path = "/root/ATLPINN/log/SWD2D-hard-%s-%s.log" % (tasks[0], create_date)

    ### get data
    dataset = Dataset2D(data_path, tasks)
    X_init, Y_init = dataset.get_init_points(N_init)
    X_bound = dataset.get_bound_points(N_bound)
    X_eqns, _ = dataset.get_sample_points(N_eqns)
    X_test, Y_test = dataset.get_test_points(N_test)
    X, Y = dataset.get_all_points()

    ### modal and train | test
    network = NeuralNetwork_Hard(X, shared_layers, special_layers, task_number, device)
    model = MTPINN_SWD2D(X_init, Y_init, X_bound, X_test, Y_test, X_eqns, network, batch_size, init_weight,
                         bound_weight, log_path, learning_rate, task_number, device)

    model.logging("number init: %d" % X_init.shape[0])
    model.logging("number bound: %d" % X_bound.shape[0])
    model.logging("number sample: %d" % X_eqns.shape[0])
    model.logging("number test: %d" % X_test.shape[0])

    # train
    model.train(adam_it=30000, lbfgs_it=0, clip_grad=False, decay_it=10000, decay_rate=0.5, print_it=10,
                evaluate_it=100, cosine_sim=False)

    # test
    Y_pred = model.predict(X.to(device))

    for task in range(task_number):
        # prepare data
        h_pred = Y_pred[task, :, 2:3]
        h = Y[task, :, 0:1]
        h = h.to(device)

        error = relative_error(h_pred, h)
        model.logging(f'task {tasks[task]}, l2 related error: {error:.3e}')

        with open(result_path, 'a+') as f:
            f.write(f'task {tasks[task]}, l2 related error: {error:.3e}\n')

        # save prediction
        hard_path = "/root/ATLPINN/output/swd_hard_%d.npy" % (tasks[task])
        hard_data = {'h': h_pred[task]}
        np.save(hard_path, hard_data)


if __name__ == '__main__':
    for i in range(0, 100, 2):
        task = [str(i).rjust(4, '0'), str(i + 1).rjust(4, '0')]
        run(task)
