import copy
import time
from torch import optim, nn
from utils.visualization import render_animation
from utils import *
from utils.evaluation import compute_stats
from utils.pose_gen import pose_generator
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../data_loader'))
from data_loader.dataset_fish import DatasetFish

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class Trainer:
    def __init__(self,
                 model,
                 diffusion,
                 dataset,
                 cfg,
                 logger,
                 tb_logger):
        super().__init__()

        self.generator_val = None
        self.val_losses = None
        self.t_s = None
        self.train_losses = None
        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None
        self.generator_train = None

        self.model = model
        self.diffusion = diffusion
        self.dataset = dataset
        self.cfg = cfg
        self.logger = logger
        self.tb_logger = tb_logger

        self.iter = 0
        self.lrs = []

        if self.cfg.ema is True:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None

    def loop(self):
        self.before_train()
        for self.iter in range(0, self.cfg.num_epoch):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()

    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.milestone,
                                                           gamma=self.cfg.gamma)
        self.criterion = nn.MSELoss()

    def before_train_step(self):
        self.model.train()
        # self.generator_train = self.dataset['train'].sampling_generator(num_samples=self.cfg.num_data_sample,
                                                                        # batch_size=self.cfg.batch_size)
        file_path = '/home/peter/TransFusion/data/fish-1222-demo19.npz'
        fish_dataset = DatasetFish(file_path,interval_length=75, stride=1,batch_size=8)
        self.generator_train = fish_dataset.sampling_generator()
        self.t_s = time.time()
        self.train_losses = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):

        for traj_np in self.generator_train:
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                # traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                # 不能删去头部关键点坐标
                traj_np = traj_np.reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)
                traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)

                if np.random.random() > self.cfg.mod_train:
                    traj_dct_mod = None

            # train
            t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
            x_t, noise = self.diffusion.noise_motion(traj_dct, t)
            predicted_noise = self.model(x_t, t, mod=traj_dct_mod)
            loss = self.criterion(predicted_noise, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]

            if args_ema is True:
                ema.step_ema(ema_model, self.model)

            self.train_losses.update(loss.item())
            self.tb_logger.add_scalar('Loss/train', loss.item(), self.iter)

            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np

    def after_train_step(self):
        self.lr_scheduler.step()
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} lr: {:.5f}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.train_losses.avg,
                                                                            self.lrs[-1]))

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_losses = AverageMeter()
        # self.generator_val = self.dataset['test'].sampling_generator(num_samples=self.cfg.num_val_data_sample,
        #                                                              batch_size=self.cfg.batch_size)
        file_path = '/home/peter/TransFusion/data/fish-1222-demo19.npz'
        fish_dataset = DatasetFish(file_path,interval_length=75, stride=1,batch_size=8)
        self.generator_val= fish_dataset.sampling_generator()
        
        self.logger.info(f"Starting val epoch {self.iter}:")

    def run_val_step(self):
        for traj_np in self.generator_val:
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                # traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                # 不能删去头部关键点坐标
                traj_np = traj_np.reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad,
                                        self.cfg.zero_index)
                # [n_pre × (t_his + t_pre)] matmul [(t_his + t_pre) × 3 * (joints - 1)]
                traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)

                if np.random.random() > self.cfg.mod_train:
                    traj_dct_mod = None

                t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
                x_t, noise = self.diffusion.noise_motion(traj_dct, t)
                predicted_noise = self.model(x_t, t, mod=traj_dct_mod)
                loss = self.criterion(predicted_noise, noise)

                self.val_losses.update(loss.item())
                self.tb_logger.add_scalar('Loss/val', loss.item(), self.iter)

            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np

    def after_val_step(self):
        self.logger.info('====> Epoch: {} Time: {:.2f} Val Loss: {}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.val_losses.avg))
        if self.cfg.save_model_interval > 0 and (self.iter + 1) % self.cfg.save_model_interval == 0:
            if self.cfg.ema is True:
                torch.save(self.ema_model.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt"))
