import torch
import h5py
import numpy as np

torch.manual_seed(1234)
np.random.seed(1234)


class Dataset1D:
    def __init__(self, data_path, tasks, method="no sorption"):
        '''
        :param data_path:   h5 file path
        :param tasks:       tasks, if len(tasks) > 1, multi tasks
        :param method:      if method = "absorption", the h5 file organization is different from others
        '''
        h5_file = h5py.File(data_path, "r")

        ### get X
        if method == "sorption":
            # dim x = [x]
            data_grid_x = torch.tensor(h5_file[tasks[0]]["grid"]["x"], dtype=torch.float)
            # dim t = [t]
            data_grid_t = torch.tensor(h5_file[tasks[0]]["grid"]["t"], dtype=torch.float)
            # dim output = [u]
            data_output = torch.tensor(np.array(h5_file[tasks[0]]["data"]), dtype=torch.float)
        else:
            # dim x = [x]
            data_grid_x = torch.tensor(h5_file["x-coordinate"], dtype=torch.float)
            # dim t = [t]
            data_grid_t = torch.tensor(h5_file["t-coordinate"], dtype=torch.float)
            # dim output = [u]
            data_output = torch.tensor(np.array(h5_file["tensor"][tasks[0]]), dtype=torch.float)

        # permute from [t, x] -> [x, t]
        data_output = data_output.T

        # boundary information
        dx = data_grid_x[1] - data_grid_x[0]
        self.xL = data_grid_x[0] - 0.5 * dx
        self.xR = data_grid_x[-1] + 0.5 * dx
        tdim = data_output.size(1)

        # generate input data from mesh
        XX, TT = torch.meshgrid([data_grid_x, data_grid_t[:tdim]], indexing="ij")
        self.X = torch.vstack([XX.ravel(), TT.ravel()]).T

        ### get Y
        Y = []
        for task in tasks:
            if method == "sorption":
                data_output = torch.tensor(h5_file[task]["data"], dtype=torch.float)
            else:
                data_output = torch.tensor(np.array(h5_file["tensor"][task]), dtype=torch.float)
            # permute from [t, x] -> [x, t]
            data_output = data_output.T
            data_output = data_output.reshape(-1, 1)
            Y.append(data_output)
        self.Y = torch.stack(Y)

        ### init
        idx_init = np.where(self.X[:, 1] == 0.0)[0]
        self.X_init = self.X[idx_init, :]
        self.Y_init = self.Y[:, idx_init, :]

        ### boundary
        idx_bound = np.where(self.X[:, 0] == self.X[0, 0])[0]
        t_bound = self.X[idx_bound, 1, None]
        x_l_bound = self.xL * torch.ones_like(t_bound)
        x_r_bound = self.xR * torch.ones_like(t_bound)
        self.X_l_bound = torch.concat((x_l_bound, t_bound), 1)
        self.X_r_bound = torch.concat((x_r_bound, t_bound), 1)

        # number information
        self.N_task = len(tasks)
        self.N = self.X.shape[0]
        self.N_init = self.X_init.shape[0]
        self.N_bound = self.X_r_bound.shape[0]

    def get_sample_points(self, N_eqns):
        # get equation points
        idx_eqns = np.random.choice(self.N, min(N_eqns, self.N), replace=False)
        X_eqns = self.X[idx_eqns, :]
        Y_eqns = self.Y[:, idx_eqns, :]

        # for multitask: only one X with multi Y
        if self.N_task == 1:
            return X_eqns, Y_eqns[0]
        else:
            return X_eqns, Y_eqns

    def get_data_points(self, N_data):
        # get intra-domain points (if any)
        idx_data = np.random.choice(self.N, min(N_data, self.N), replace=False)
        X_data = self.X[idx_data, :]
        Y_data = self.Y[:, idx_data, :]

        # for multitask: only one X with multi Y
        if self.N_task == 1:
            return X_data, Y_data[0]
        else:
            return X_data, Y_data

    def get_init_points(self, N_init):
        # get initail points
        idx_init = np.random.choice(self.N_init, min(N_init, self.N_init), replace=False)
        X_init = self.X_init[idx_init, :]
        Y_init = self.Y_init[:, idx_init, :]

        # for multitask: only one X with multi Y
        if self.N_task == 1:
            return X_init, Y_init[0]
        else:
            return X_init, Y_init

    def get_bound_points(self, N_bound):
        # get boundary points
        idx_bound = np.random.choice(self.N_bound, min(N_bound, self.N_bound), replace=False)
        X_l_bound = self.X_l_bound[idx_bound, :]
        X_r_bound = self.X_r_bound[idx_bound, :]

        return X_l_bound, X_r_bound

    def get_test_points(self, N_test):
        # get test points for evaluation
        idx_test = np.random.choice(self.N, min(N_test, self.N), replace=False)
        X_test = self.X[idx_test, :]
        Y_test = self.Y[:, idx_test, :]

        # for multitask: only one X with multi Y
        if self.N_task == 1:
            return X_test, Y_test[0]
        else:
            return X_test, Y_test

    def get_all_points(self):
        # return total points
        if self.N_task == 1:
            return self.X, self.Y[0]
        else:
            return self.X, self.Y


class Dataset2D:
    def __init__(self, data_path, tasks):
        '''
        :param data_path:   h5 file path
        :param tasks:       tasks, if len(tasks) > 1, multi tasks
        '''
        h5_file = h5py.File(data_path, "r")

        # build input data from individual dimensions
        task0_group = h5_file[tasks[0]]
        # dim x = [x]
        data_grid_x = torch.tensor(task0_group["grid"]["x"], dtype=torch.float)
        # dim y = [y]
        data_grid_y = torch.tensor(task0_group["grid"]["y"], dtype=torch.float)
        # dim t = [t]
        data_grid_t = torch.tensor(task0_group["grid"]["t"], dtype=torch.float)

        # dim output = [output]
        data_output = []
        for task in tasks:
            # read output
            task_group = h5_file[task]
            data_tensor = torch.tensor(np.array(task_group["data"]), dtype=torch.float)

            # permute from [t, x, y] -> [x, y, t]
            permute_idx = list(range(1, len(data_tensor.shape) - 1))
            permute_idx.extend(list([0, -1]))
            data_tensor = data_tensor.permute(permute_idx)

            # reshape to (x,1)
            data_tensor = data_tensor.reshape(-1, 1)
            data_output.append(data_tensor)

        # generate mesh from individual dimensions
        XX, YY, TT = torch.meshgrid([data_grid_x, data_grid_y, data_grid_t], indexing="ij")
        self.X = torch.vstack([XX.ravel(), YY.ravel(), TT.ravel()]).T
        self.Y = torch.stack(data_output)

        # boundary information
        dx = data_grid_x[1] - data_grid_x[0]
        xL = data_grid_x[0] - 0.5 * dx
        xR = data_grid_x[-1] + 0.5 * dx

        dy = data_grid_y[1] - data_grid_y[0]
        yT = data_grid_y[-1] + 0.5 * dy
        yD = data_grid_y[0] - 0.5 * dy

        # boundary in X aria
        idx_lb = np.where(self.X[:, 0] == self.X[0, 0])[0]
        y_lb = self.X[idx_lb, 1, None]
        t_lb = self.X[idx_lb, 2, None]
        x_lb = xL * torch.ones_like(y_lb)
        x_rb = xR * torch.ones_like(y_lb)

        # boundary in Y aria
        idx_ub = np.where(self.X[:, 1] == self.X[0, 1])[0]
        x_ub = self.X[idx_ub, 0, None]
        t_ub = self.X[idx_ub, 2, None]
        y_ub = yT * torch.ones_like(x_ub)
        y_db = yD * torch.ones_like(x_ub)

        # two-dimension problem has four boundary
        self.X_l_bound = torch.concat((x_lb, y_lb, t_lb), 1)
        self.X_r_bound = torch.concat((x_rb, y_lb, t_lb), 1)
        self.X_u_bound = torch.concat((x_ub, y_ub, t_ub), 1)
        self.X_d_bound = torch.concat((x_ub, y_db, t_ub), 1)

        ### total initial data points, t=0
        idx_init = np.where(self.X[:, 2] == 0.0)[0]
        self.X_init = self.X[idx_init, :]
        self.Y_init = self.Y[:, idx_init, :]

        # number information
        self.N_task = len(tasks)
        self.N = self.X.shape[0]
        self.N_init = self.X_init.shape[0]
        self.N_bound = self.X_r_bound.shape[0]

    def get_sample_points(self, N_eqns):
        idx_eqns = np.random.choice(self.N, min(N_eqns, self.N), replace=False)
        X_eqns = self.X[idx_eqns, :]
        Y_eqns = self.Y[:, idx_eqns, :]

        # for multitask: only one X with multi Y
        if self.N_task == 1:
            return X_eqns, Y_eqns[0]
        else:
            return X_eqns, Y_eqns

    def get_data_points(self, N_data):
        # get intra-domain data points (if any
        idx_data = np.random.choice(self.N, min(N_data, self.N), replace=False)
        X_data = self.X[idx_data, :]
        Y_data = self.Y[:, idx_data, :]

        # for multitask: only one X with multi Y
        if self.N_task == 1:
            return X_data, Y_data[0]
        else:
            return X_data, Y_data

    def get_bound_points(self, N_bound):
        # random choose boundary data points
        # we choose N_bound 个 boundary data points in four directions
        # note that for multitask, the boundary is the same
        idx_bound = np.random.choice(self.N_bound, min(N_bound, self.N_bound), replace=False)
        X_l_bound = self.X_l_bound[idx_bound, :]

        idx_bound = np.random.choice(self.N_bound, min(N_bound, self.N_bound), replace=False)
        X_r_bound = self.X_r_bound[idx_bound, :]

        idx_bound = np.random.choice(self.N_bound, min(N_bound, self.N_bound), replace=False)
        X_u_bound = self.X_u_bound[idx_bound, :]

        idx_bound = np.random.choice(self.N_bound, min(N_bound, self.N_bound), replace=False)
        X_d_bound = self.X_d_bound[idx_bound, :]

        X_bound = torch.concat((X_l_bound, X_r_bound, X_u_bound, X_d_bound), 0)

        return X_bound

    def get_init_points(self, N_init):
        # random choose initial data points
        idx_init = np.random.choice(self.N_init, min(N_init, self.N_init), replace=False)
        X_init = self.X_init[idx_init, :]
        Y_init = self.Y_init[:, idx_init, :]

        # for multitask: only one X with multi Y
        if self.N_task == 1:
            return X_init, Y_init[0]
        else:
            return X_init, Y_init

    def get_test_points(self, N_test):
        # get test points for evluation
        idx_test = np.random.choice(self.N, min(N_test, self.N), replace=False)
        X_test = self.X[idx_test, :]
        Y_test = self.Y[:, idx_test, :]

        # for multitask: only one X with multi Y
        if self.N_task == 1:
            return X_test, Y_test[0]
        else:
            return X_test, Y_test

    def get_all_points(self):
        # return total points
        # for multitask: only one X with multi Y
        if self.N_task == 1:
            return self.X, self.Y[0]
        else:
            return self.X, self.Y