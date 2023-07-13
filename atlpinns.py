import torch
import torch.nn as nn
from utilities import to_device, relative_error, gradients, mean_squared_error
import numpy as np
import time

torch.manual_seed(1234)
np.random.seed(1234)


class ATLPINN():
    def __init__(self, X_test, Y_test, network, batch_size, init_weight, bound_weight,
                 log_path, learning_rate, task_number, device):
        """
        :param X_test:          test data points input
        :param Y_test:          test data points output
        :param network:         network structure, e.g. hard, soft, mmoe, ple
        :param batch_size:      batch size
        :param init_weight:     weight of initial points
        :param bound_weight:    weight of boundary points
        :param log_path:        log4loss file output path
        :param learning_rate:   learning rate
        :param task_number:     number of task
        :param device:          device for training: e.g. cpu, gpu, tpu
        """

        # dataset
        self.X_test, self.Y_test = to_device(X_test, device), to_device(Y_test, device)

        # hypo-parameters
        self.log_path = log_path
        self.batch_size = batch_size
        self.init_weight = init_weight
        self.bound_weight = bound_weight
        self.task_number = task_number

        # model
        self.network = network.to(device)

        # loss weights
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6).to(device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def loss_pde(self) -> torch.Tensor:
        # abstract method for equation loss function
        pass

    def loss_bc(self) -> torch.Tensor:
        # abstract method for boundary loss function
        pass

    def loss_ic(self) -> torch.Tensor:
        # abstract method for initial loss function
        pass

    def get_grad_cos_sim(self, grad1, grad2):
        '''
        judge the cosine similarity for two grads
        :param grad1: grad1 organized for any shape
        :param grad2: grad2 organized for any shape
        :return: cosine similarity
        '''
        grad1 = torch.concat([torch.reshape(x, shape=(-1,)) for x in grad1])
        grad2 = torch.concat([torch.reshape(x, shape=(-1,)) for x in grad2])

        # perform min(max(-1, dist),1) operation for eventual rounding errors (there's about 1 every epoch)
        dist = 1 - self.cos(grad1, grad2)
        dist = min(max(-1, dist), 1)

        return dist

    def train(self, adam_it, lbfgs_it, clip_grad, decay_it, decay_rate, print_it, evaluate_it, cosine_sim):
        '''
        :param adam_it:     iteration times for adam optimizer
        :param lbfgs_it:    iteration times for lbfgs optimizer
        :param clip_grad:   if Ture then clip gradients every step
        :param decay_it:    decay the lr rate every xx iteration, -1 for No decay
        :param decay_rate:  lr = lr * rate
        :param print_it:    print the loss every xx iteration
        :param evaluate_it: print the l2 error every xx iteration
        :param cosine_sim:  if Ture then use gradient cosine similarity
        :return:
        '''
        self.logging('\nstart training...')
        tic = time.time()
        self.it = 0

        # define the network as training method
        self.network.train()

        # once iteration
        def closure():
            # compute loss function
            loss_pde = self.loss_pde()
            loss_bc = self.loss_bc()
            loss_ic = self.loss_ic()
            loss = loss_pde + self.bound_weight * loss_bc + self.init_weight * loss_ic

            # gradient cosine similarity
            if cosine_sim == True:
                grad_main = gradients(loss[0], self.network.get_shared())
                total_loss = loss[0]

                # compute cosine similarity for every auxiliary task with the main task
                for task in range(1, self.task_number):
                    grad = gradients(loss[task], self.network.get_shared())
                    # if  the cosine similarity < 0, abandon the auxiliary task for this iteration
                    if self.get_grad_cos_sim(grad_main, grad) >= 0:
                        total_loss = total_loss + loss[task]
            # simple weighted loss
            else:
                total_loss = torch.sum(loss)

            # print
            if self.it % print_it == 0:
                for task in range(self.task_number):
                    self.logging(
                        f'it {self.it}, task {task}: loss {loss[task]:.3e}, loss_pde {loss_pde[task]:.3e}, '
                        f'loss_bc {loss_bc[task]:.3e}, loss_ic {loss_ic[task]:.3e}')

            # evaluation
            if self.it % evaluate_it == 0:
                self.evaluation()

            if clip_grad == True:
                torch.nn.utils.clip_grad_norm_(parameters=self.network.parameters(), max_norm=1, norm_type=2)

            # back propagation
            self.it += 1
            self.optimizer.zero_grad()
            total_loss.backward()
            return total_loss

        # Adam
        while self.it < adam_it:
            self.optimizer.step(closure)

            # decay learning rate
            if self.it != -1 and self.it % decay_it == 0:
                for params in self.optimizer.param_groups:
                    params['lr'] *= decay_rate

        # LBFGS
        if lbfgs_it > 0:
            self.optimizer = torch.optim.LBFGS(self.network.parameters(), max_iter=lbfgs_it)
            self.optimizer.step(closure)

        toc = time.time()
        self.logging(f'total training time: {toc - tic}')

    def predict(self, X):
        # prediction with no gradients
        with torch.no_grad():
            Y_pred = self.network(X)
        return Y_pred

    def evaluation(self):
        # for test
        Y_pred = self.network(self.X_test)
        for task in range(self.task_number):
            error = relative_error(Y_pred[task], self.Y_test[task])
            self.logging(f'task {task}, l2 related error: {error:.3e}')

    def logging(self, log_item):
        # log4loss
        with open(self.log_path, 'a+') as log:
            log.write(log_item + '\n')
        print(log_item)


class ATLPINN_ReactDiff1D(ATLPINN):
    def __init__(self, X_init, Y_init, X_l_bound, X_r_bound, X_test, Y_test, X_eqns, network,
                 batch_size, nu, rho, init_weight, bound_weight, log_path, learning_rate, task_number, device):
        '''
        :param X_init:      initial data points input
        :param Y_init:      initial data points output
        :param X_l_bound:   left boundary data points input
        :param X_r_bound:   right boundary data points input
        :param X_eqns:      equation data points input
        :param nu:          density
        :param rho:         Raylow number
        '''
        super().__init__(X_test, Y_test, network, batch_size, init_weight, bound_weight, log_path,
                         learning_rate, task_number, device)

        self.nu, self.rho = nu, rho
        self.X_init, self.Y_init = to_device(X_init, device), to_device(Y_init, device)
        self.X_l_bound, self.X_r_bound = to_device(X_l_bound, device), to_device(X_r_bound, device)
        self.X_eqns = to_device(X_eqns, device)

    def loss_pde(self):
        # mini-batch
        idx_batch = np.random.choice(self.X_eqns.shape[0], min(self.batch_size, self.X_eqns.shape[0]), replace=False)
        X_batch = self.X_eqns[idx_batch, :]

        # output shape: [tasks_number, batch_size, physical fields shape e.g. (u, v, p)]
        Y_pred = self.network(X_batch)

        # multi task loss
        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # diffusion reaction equation 1D
            u = Y_pred[task, :, 0:1]

            u_g = gradients(u, X_batch)[0]
            u_x, u_t = u_g[:, 0:1], u_g[:, 1:2]

            u_gg = gradients(u_x, X_batch)[0]
            u_xx = u_gg[:, 0:1]

            f = u_t - self.nu * u_xx - self.rho * u * (1.0 - u)
            e[task] = mean_squared_error(f, 0)
        return e

    def loss_bc(self):
        # reload the boundary condition loss function
        Y_l = self.network(self.X_l_bound)
        Y_r = self.network(self.X_r_bound)

        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # periodical boundary condition
            e[task] = mean_squared_error(Y_l[task], Y_r[task])
        return e

    def loss_ic(self):
        # reload the initial condition loss function
        Y_pred = self.network(self.X_init)

        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # points set initial condition
            e[task] = mean_squared_error(Y_pred[task], self.Y_init[task])
        return e


class ATLPINN_Burgers1D(ATLPINN):
    def __init__(self, X_init, Y_init, X_l_bound, X_r_bound, X_test, Y_test, X_eqns, network, batch_size, nu,
                 init_weight, bound_weight, log_path, learning_rate, task_number, device):
        '''
        :param X_init:      initial data points input
        :param Y_init:      initial data points output
        :param X_l_bound:   left boundary data points input
        :param X_r_bound:   right boundary data points input
        :param X_eqns:      equation data points input
        :param nu:          density
        '''
        super().__init__(X_test, Y_test, network, batch_size, init_weight, bound_weight, log_path,
                         learning_rate, task_number, device)

        self.nu = nu
        self.X_init, self.Y_init = to_device(X_init, device), to_device(Y_init, device)
        self.X_l_bound, self.X_r_bound = to_device(X_l_bound, device), to_device(X_r_bound, device)
        self.X_eqns = to_device(X_eqns, device)

    def loss_pde(self):
        # mini-batch
        idx_batch = np.random.choice(self.X_eqns.shape[0], min(self.batch_size, self.X_eqns.shape[0]), replace=False)
        X_batch = self.X_eqns[idx_batch, :]

        # ouptput shape: [tasks_number, batch_size, physical fields shape e.g. (u, v, p)]
        Y_pred = self.network(X_batch)

        # multi task loss
        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # Burgers equation 1D
            u = Y_pred[task, :, 0:1]

            u_g = gradients(u, X_batch)[0]
            u_x, u_t = u_g[:, 0:1], u_g[:, 1:2]

            u_gg = gradients(u_x, X_batch)[0]
            u_xx = u_gg[:, 0:1]

            f = u_t + u * u_x - self.nu / np.pi * u_xx
            e[task] = mean_squared_error(f, 0)
        return e

    def loss_bc(self):
        # reload the boundary condition loss function
        Y_l = self.network(self.X_l_bound)
        Y_r = self.network(self.X_r_bound)

        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # periodical boundary condition
            e[task] = mean_squared_error(Y_l[task], Y_r[task])
        return e

    def loss_ic(self):
        # reload the initial condition loss function
        Y_pred = self.network(self.X_init)

        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # points set initial condition
            e[task] = mean_squared_error(Y_pred[task], self.Y_init[task])
        return e


class ATLPINN_Advection1D(ATLPINN):
    def __init__(self, X_init, Y_init, X_l_bound, X_r_bound, X_test, Y_test, X_eqns, network,
                 batch_size, beta, init_weight, bound_weight, log_path, learning_rate, task_number, device):
        '''
        :param X_init:      initial data points input
        :param Y_init:      initial data points output
        :param X_l_bound:   left boundary data points input
        :param X_r_bound:   right boundary data points input
        :param X_eqns:      equation data points input
        :param beta:        beta number
        '''
        super().__init__(X_test, Y_test, network, batch_size, init_weight, bound_weight, log_path,
                         learning_rate, task_number, device)

        self.beta = beta
        self.X_init, self.Y_init = to_device(X_init, device), to_device(Y_init, device)
        self.X_l_bound, self.X_r_bound = to_device(X_l_bound, device), to_device(X_r_bound, device)
        self.X_eqns = to_device(X_eqns, device)

    def loss_pde(self):
        # mini-batch
        idx_batch = np.random.choice(self.X_eqns.shape[0], min(self.batch_size, self.X_eqns.shape[0]), replace=False)
        X_batch = self.X_eqns[idx_batch, :]

        # ouptput shape: [tasks_number, batch_size, physical fields shape e.g. (u, v, p)]
        Y_pred = self.network(X_batch)

        # multi task loss
        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # Advection1D
            u = Y_pred[task, :, 0:1]

            u_g = gradients(u, self.X_eqns)[0]
            u_x, u_t = u_g[:, 0:1], u_g[:, 1:2]

            f = u_t + self.beta * u_x
            e[task] = mean_squared_error(f, 0)
        return e

    def loss_bc(self):
        # reload the boundary condition loss function
        Y_l = self.network(self.X_l_bound)
        Y_r = self.network(self.X_r_bound)

        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # periodical boundary condition
            e[task] = mean_squared_error(Y_l[task], Y_r[task])
        return e

    def loss_ic(self):
        # reload the initial condition loss function
        Y_pred = self.network(self.X_init)

        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # points set initial condition
            e[task] = mean_squared_error(Y_pred[task], self.Y_init[task])
        return e


class ATLPINN_DiffSorp1D(ATLPINN):
    def __init__(self, X_init, Y_init, X_l_bound, X_r_bound, X_test, Y_test, X_eqns, network, batch_size,
                 init_weight, bound_weight, log_path, learning_rate, task_number, device):
        '''
        :param X_init:      initial data points input
        :param Y_init:      initial data points output
        :param X_l_bound:   left boundary data points input
        :param X_r_bound:   right boundary data points input
        :param X_eqns:      equation data points input
        '''
        super().__init__(X_test, Y_test, network, batch_size, init_weight, bound_weight, log_path,
                         learning_rate, task_number, device)

        self.X_init, self.Y_init = to_device(X_init, device), to_device(Y_init, device)
        self.X_l_bound, self.X_r_bound = to_device(X_l_bound, device), to_device(X_r_bound, device)
        self.X_eqns = to_device(X_eqns, device)

    def loss_pde(self):
        D: float = 5e-4
        por: float = 0.29
        rho_s: float = 2880
        k_f: float = 3.5e-4
        n_f: float = 0.874

        # mini-batch
        idx_batch = np.random.choice(self.X_eqns.shape[0], min(self.batch_size, self.X_eqns.shape[0]), replace=False)
        X_batch = self.X_eqns[idx_batch, :]
        Y_pred = self.network(X_batch)

        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # ouptput shape: [tasks_number, batch_size, physical fields shape e.g. (u, v, p)]
            u = Y_pred[task, :, 0:1]

            # du/dx du/dt
            u_g = gradients(u, self.X_eqns)[0]
            u_x, u_t = u_g[:, 0:1], u_g[:, 1:2]

            # du/dxx
            u_xg = gradients(u_x, self.X_eqns)[0]
            u_xx = u_xg[:, 0:1]

            # Diff Sorp 1D
            retardation_factor = 1 + ((1 - por) / por) * rho_s * k_f * n_f * (u + 1e-6) ** (n_f - 1)
            f = u_t - D / retardation_factor * u_xx

            e[task] = mean_squared_error(f, 0)
        return e

    def loss_bc(self):
        # reload the boundary condition loss function
        D: float = 5e-4

        Y_l_pred = self.network(self.X_l_bound)
        Y_r_pred = self.network(self.X_r_bound)

        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # ouptput shape: [tasks_number, batch_size, physical fields shape e.g. (u, v, p)]
            u = Y_l_pred[task, :, 0:1]

            # du/dx
            u_g = gradients(u, self.X_l_bound)[0]
            u_x = u_g[:, 0:1]

            # left -> Dirichlet; right -> Neumann
            Y_l = 1.0
            Y_r = D * u_x

            e[task] = mean_squared_error(Y_l_pred[task], Y_l) + mean_squared_error(Y_r_pred[task], Y_r)
        return e

    def loss_ic(self):
        # reload the initial condition loss function
        Y_pred = self.network(self.X_init)

        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # points set initial condition
            e[task] = mean_squared_error(Y_pred[task], self.Y_init[task])
        return e


class ATLPINN_SWD2D(ATLPINN):
    def __init__(self, X_init, Y_init, X_bound, X_test, Y_test, X_eqns, network, batch_size, init_weight,
                 bound_weight, log_path, learning_rate, task_number, device):
        '''
        :param X_init:      initial data points input
        :param Y_init:      initial data points output
        :param X_bound:     boundary data points input
        :param X_eqns:      equation data points input
        '''
        super().__init__(X_test, Y_test, network, batch_size, init_weight, bound_weight, log_path,
                         learning_rate, task_number, device)

        self.X_init, self.Y_init = to_device(X_init, device), to_device(Y_init, device)
        self.X_bound = to_device(X_bound, device)
        self.X_eqns = to_device(X_eqns, device)
        self.N_eqns = self.X_eqns.shape[0]

    def loss_pde(self):
        # shallow water equation 2D
        g = 1.0

        # mini-batch
        idx_batch = np.random.choice(self.N_eqns, min(self.batch_size, self.N_eqns), replace=False)
        X_batch = self.X_eqns[idx_batch, :]
        Y_pred = self.network(X_batch)

        # compute pde loss for every task
        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # shallow water equation 2D
            # ouptput shape: [tasks_number, batch_size, physical fields shape e.g. (u, v, p)]
            u, v, h = Y_pred[task, :, 0:1], Y_pred[task, :, 1:2], Y_pred[task, :, 2:3]

            u_g = gradients(u, X_batch)[0]
            u_x, u_y, u_t = u_g[:, 0:1], u_g[:, 1:2], u_g[:, 2:3]

            v_g = gradients(v, X_batch)[0]
            v_x, v_y, v_t = v_g[:, 0:1], v_g[:, 1:2], v_g[:, 2:3]

            h_g = gradients(h, X_batch)[0]
            h_x, h_y, h_t = h_g[:, 0:1], h_g[:, 1:2], h_g[:, 2:3]

            eq1 = h_t + h_x * u + h * u_x + h_y * v + h * v_y
            eq2 = u_t + u * u_x + v * u_y + g * h_x
            eq3 = v_t + u * v_x + v * v_y + g * h_y

            e[task] = mean_squared_error(eq1, 0) + mean_squared_error(eq2, 0) + mean_squared_error(eq3, 0)
        return e

    def loss_bc(self):
        # reload the boundary condition loss function
        Y_bound = self.network(self.X_bound)

        # compute boundary condition loss for every task
        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            u = Y_bound[task, :, 0:1]
            v = Y_bound[task, :, 1:2]

            u_g = gradients(u, self.X_bound)[0]
            u_x, u_y = u_g[:, 0:1], u_g[:, 1:2]

            v_g = gradients(v, self.X_bound)[0]
            v_x, v_y = v_g[:, 0:1], v_g[:, 1:2]

            # Neumann boundary condition: the two velocity in two axis, gradients = 0
            e[task] = mean_squared_error(u_x, 0) + mean_squared_error(v_x, 0) + \
                      mean_squared_error(u_y, 0) + mean_squared_error(v_y, 0)
        return e

    def loss_ic(self):
        # reload the initial condition loss function
        Y_pred = self.network(self.X_init)

        # compute initial condition loss for every task
        e = torch.zeros(size=(self.task_number,))
        for task in range(self.task_number):
            # initial condition loss function: h->h', u->0, v->0
            u_pred = Y_pred[task, :, 0:1]
            v_pred = Y_pred[task, :, 1:2]
            h_pred = Y_pred[task, :, 2:3]
            h_init = self.Y_init[task, :, 0:1]

            e[task] = mean_squared_error(h_pred, h_init) + mean_squared_error(u_pred, 0) + \
                      mean_squared_error(v_pred, 0)
        return e

    def evaluation(self):
        Y_pred = self.network(self.X_test)
        for task in range(self.task_number):
            h_pred = Y_pred[task, :, 2:3]
            h_test = self.Y_test[task, :, 0:1]
            error = relative_error(h_pred, h_test)
            self.logging(f'l2 related error: {error:.3e}')