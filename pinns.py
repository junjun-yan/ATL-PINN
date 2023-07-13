import torch
from utilities import to_device, relative_error, gradients, mean_squared_error
import numpy as np
import time

torch.manual_seed(1234)
np.random.seed(1234)


class PINN():
    def __init__(self, network, X_test, Y_test, batch_size, init_weight,
                 bound_weight, log_path, learning_rate, device):
        # hypo-parameters
        self.log_path = log_path
        self.batch_size = batch_size
        self.init_weight = init_weight
        self.bound_weight = bound_weight

        # other property from children class
        self.network = network
        self.X_test, self.Y_test = to_device(X_test, device), to_device(Y_test, device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def loss_pde(self) -> float:
        # abstract interface for pde loss
        pass

    def loss_bc(self) -> float:
        # abstract interface for boundary loss
        pass

    def loss_ic(self) -> float:
        # abstract interface for initial loss
        pass

    def train(self, adam_it, lbfgs_it, clip_grad, decay_it, decay_rate, print_it, evaluate_it):
        '''
        :param adam_it:     iteration times for adam optimizer
        :param lbfgs_it:    iteration times for lbfgs optimizer
        :param clip_grad:   if Ture then clip gradients every step
        :param decay_it:    decay the lr rate every xx iteration, -1 for No decay
        :param decay_rate:  lr = lr * rate
        :param print_it:    print the loss every xx iteration
        :param evaluate_it: print the l2 error every xx iteration
        :return:
        '''
        self.logging('\nstart training...')
        tic = time.time()
        self.it = 0
        self.network.train()

        # once iteration
        def closure():
            # compute losses
            loss_pde = self.loss_pde()
            loss_bc = self.loss_bc()
            loss_ic = self.loss_ic()
            loss = loss_pde + self.bound_weight * loss_bc + self.init_weight * loss_ic

            # print
            if self.it % print_it == 0:
                self.logging(f'it {self.it}: loss {loss:.3e}, loss_pde {loss_pde:.3e}, '
                             f'loss_bc {loss_bc:.3e}, loss_ic {loss_ic:.3e}')

            # evaluate
            if self.it % evaluate_it == 0:
                self.evaluation()

            # 梯度裁剪
            if clip_grad == True:
                torch.nn.utils.clip_grad_norm_(parameters=self.network.parameters(), max_norm=1, norm_type=2)

            # backpropagation
            self.it += 1
            self.optimizer.zero_grad()
            loss.backward()
            return loss

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
        error = relative_error(Y_pred, self.Y_test)
        self.logging(f'l2 related error: {error:.3e}')

    def logging(self, log_item):
        # write into consolo and file
        with open(self.log_path, 'a+') as log:
            log.write(log_item + '\n')
        print(log_item)


class PINN_ReactDiff1D(PINN):
    def __init__(self, X_init, Y_init, X_l_bound, X_r_bound, X_test, Y_test, X_eqns, network,
                 batch_size, nu, rho, init_weight, bound_weight, log_path, learning_rate, device):
        super().__init__(network, X_test, Y_test, batch_size, init_weight,
                         bound_weight, log_path, learning_rate, device)

        # hypo-parameters
        self.nu = nu
        self.rho = rho

        # dataset
        self.N_eqns = X_eqns.shape[0]
        self.X_init, self.Y_init = to_device(X_init, device), to_device(Y_init, device)
        self.X_l_bound, self.X_r_bound = to_device(X_l_bound, device), to_device(X_r_bound, device)
        self.X_eqns = to_device(X_eqns, device)

    def loss_pde(self):
        # pde loss function: reaction diffusion equation 1D

        # mini-batch
        idx_batch = np.random.choice(self.N_eqns, min(self.batch_size, self.N_eqns), replace=False)
        X_batch = self.X_eqns[idx_batch, :]
        Y_pred = self.network(X_batch)

        u = Y_pred[:, 0]

        u_g = gradients(u, X_batch)[0]
        u_x, u_t = u_g[:, 0], u_g[:, 1]

        u_gg = gradients(u_x, X_batch)[0]
        u_xx = u_gg[:, 0]

        e = u_t - self.nu * u_xx - self.rho * u * (1.0 - u)
        return mean_squared_error(e, 0)

    def loss_bc(self):
        # boundary loss function: periodic
        Y_l = self.network(self.X_l_bound)
        Y_r = self.network(self.X_r_bound)
        return mean_squared_error(Y_l, Y_r)

    def loss_ic(self):
        # initial loss function: points
        Y_pred = self.network(self.X_init)
        return mean_squared_error(Y_pred, self.Y_init)


class PINN_Burgers1D(PINN):
    def __init__(self, X_init, Y_init, X_l_bound, X_r_bound, X_test, Y_test, X_eqns, network, batch_size, nu,
                 init_weight, bound_weight, log_path, learning_rate, device):
        super().__init__(network, X_test, Y_test, batch_size, init_weight, bound_weight, log_path,
                         learning_rate, device)

        # hypo-parameters
        self.nu = nu

        # dataset
        self.X_init, self.Y_init = to_device(X_init, device), to_device(Y_init, device)
        self.X_l_bound, self.X_r_bound = to_device(X_l_bound, device), to_device(X_r_bound, device)
        self.X_eqns = to_device(X_eqns, device)
        self.N_eqns = X_eqns.shape[0]

    def loss_pde(self):
        # pde loss function: Burgers equation 1D

        # mini-batch
        idx_batch = np.random.choice(self.N_eqns, min(self.batch_size, self.N_eqns), replace=False)
        X_batch = self.X_eqns[idx_batch, :]
        Y_pred = self.network(X_batch)

        u = Y_pred[:, 0]
        u_g = gradients(u, X_batch)[0]

        u_x, u_t = u_g[:, 0], u_g[:, 1]
        u_gg = gradients(u_x, X_batch)[0]
        u_xx = u_gg[:, 0]

        e = u_t + u * u_x - self.nu / np.pi * u_xx
        return mean_squared_error(e, 0)

    def loss_bc(self):
        # boundary loss function: periodic
        Y_l = self.network(self.X_l_bound)
        Y_r = self.network(self.X_r_bound)
        return mean_squared_error(Y_l, Y_r)

    def loss_ic(self):
        # initial loss function: points
        Y_pred = self.network(self.X_init)
        return mean_squared_error(Y_pred, self.Y_init)


class PINN_SWD2D(PINN):
    def __init__(self, X_init, Y_init, X_bound, X_test, Y_test, X_eqns, network, batch_size, init_weight,
                 bound_weight, log_path, learning_rate, device):
        super().__init__(network, X_test, Y_test, batch_size, init_weight, bound_weight, log_path,
                         learning_rate, device)

        # dataset
        self.X_init, self.Y_init = to_device(X_init, device), to_device(Y_init, device)
        self.X_bound = to_device(X_bound, device)
        self.X_eqns = to_device(X_eqns, device)
        self.N_eqns = X_eqns.shape[0]

    def loss_pde(self):
        # shallow water equation 2D
        g = 1.0

        # mini-batch
        idx_batch = np.random.choice(self.N_eqns, min(self.batch_size, self.N_eqns), replace=False)
        X_batch = self.X_eqns[idx_batch, :]
        Y_pred = self.network(X_batch)

        u, v, h = Y_pred[:, 0:1], Y_pred[:, 1:2], Y_pred[:, 2:3]

        u_g = gradients(u, X_batch)[0]
        u_x, u_y, u_t = u_g[:, 0:1], u_g[:, 1:2], u_g[:, 2:3]

        v_g = gradients(v, X_batch)[0]
        v_x, v_y, v_t = v_g[:, 0:1], v_g[:, 1:2], v_g[:, 2:3]

        h_g = gradients(h, X_batch)[0]
        h_x, h_y, h_t = h_g[:, 0:1], h_g[:, 1:2], h_g[:, 2:3]

        eq1 = h_t + h_x * u + h * u_x + h_y * v + h * v_y
        eq2 = u_t + u * u_x + v * u_y + g * h_x
        eq3 = v_t + u * v_x + v * v_y + g * h_y

        return mean_squared_error(eq1, 0) + mean_squared_error(eq2, 0) + mean_squared_error(eq3, 0)

    def loss_bc(self):
        # Neumann boundary condition: the two velocity in two axis, gradients = 0
        Y_bound = self.network(self.X_bound)

        u = Y_bound[:, 0:1]
        v = Y_bound[:, 1:2]

        u_g = gradients(u, self.X_bound)[0]
        u_x, u_y = u_g[:, 0:1], u_g[:, 1:2]

        v_g = gradients(v, self.X_bound)[0]
        v_x, v_y = v_g[:, 0:1], v_g[:, 1:2]

        return mean_squared_error(u_x, 0) + mean_squared_error(v_x, 0) + \
            mean_squared_error(u_y, 0) + mean_squared_error(v_y, 0)

    def loss_ic(self):
        '''
        initial condition loss function: h->h', u->0, v->0
        '''
        Y_pred = self.network(self.X_init)
        u_pred = Y_pred[:, 0:1]
        v_pred = Y_pred[:, 1:2]
        h_pred = Y_pred[:, 2:3]

        h_init = self.Y_init[:, 0:1]
        return mean_squared_error(h_pred, h_init) + mean_squared_error(u_pred, 0) + mean_squared_error(v_pred, 0)

    def evaluation(self):
        Y_pred = self.network(self.X_test)
        h_pred = Y_pred[:, 2:3]
        h_test = self.Y_test[:, 0:1]

        error = relative_error(h_pred, h_test)
        self.logging(f'l2 related error: {error:.3e}')
