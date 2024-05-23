import csv
import pandas as pd
from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros


def compute_stats(diffusion, multimodal_dict, model, logger, cfg):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    def get_prediction(data, model_select):
        traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
        traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        # traj.shape: [*, t_his + t_pre, 3 * joints_num]

        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics')
        sampled_motion = diffusion.sample_ddim(model_select,
                                               traj_dct,
                                               traj_dct_cond,
                                               mode_dict)

        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est[None, ...]
        return traj_est

    gt_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    traj_gt_arr = multimodal_dict['traj_gt_arr']
    num_samples = multimodal_dict['num_samples']

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'ADE-m', 'FDE-m', 'MMADE-m', 'MMFDE-m', 'ADE-w', 'FDE-w', 'MMADE-w', 'MMFDE-w']
    stats_meter = {x: {y: AverageMeter() for y in ['TransFusion']} for x in stats_names}

    K = 50
    pred = []
    for i in tqdm(range(0, K), position=0):
        # It generates a prediction for all samples in the test set
        # So we need loop for K times
        pred_i_nd = get_prediction(data_group, model)
        pred.append(pred_i_nd)
        if i == K - 1:  # in last iteration, concatenate all candidate pred
            pred = np.concatenate(pred, axis=0)
            # pred [50, 5187, 125, 48] in h36m
            pred = pred[:, :, cfg.t_his:, :]
            print('Got 50 predictions')
            # Use GPU to accelerate
            try:
                gt_group = torch.from_numpy(gt_group).to('cuda')
            except:
                pass
            try:
                pred = torch.from_numpy(pred).to('cuda')
            except:
                pass

            for j in range(0, num_samples):
                apd, ade, fde, mmade, mmfde, ade_m,fde_m, mmade_m, mmfde_m, ade_w, fde_w, mmade_w, mmfde_w = compute_all_metrics(pred[:, j, :, :],
                                                                                                                                 gt_group[j][np.newaxis, ...],
                                                                                                                                 traj_gt_arr[j])
                stats_meter['APD']['TransFusion'].update(apd)
                stats_meter['ADE']['TransFusion'].update(ade)
                stats_meter['FDE']['TransFusion'].update(fde)
                stats_meter['MMADE']['TransFusion'].update(mmade)
                stats_meter['MMFDE']['TransFusion'].update(mmfde)
                stats_meter['ADE-m']['TransFusion'].update(ade_m)
                stats_meter['FDE-m']['TransFusion'].update(fde_m)
                stats_meter['MMADE-m']['TransFusion'].update(mmade_m)
                stats_meter['MMFDE-m']['TransFusion'].update(mmfde_m)
                stats_meter['ADE-w']['TransFusion'].update(ade_w)
                stats_meter['FDE-w']['TransFusion'].update(fde_w)
                stats_meter['MMADE-w']['TransFusion'].update(mmade_w)
                stats_meter['MMFDE-w']['TransFusion'].update(mmfde_w)
            for stats in stats_names:
                str_stats = f'{stats}: ' + ' '.join(
                    [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
                )
                logger.info(str_stats)
            pred = []

    # save stats in csv
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + ['TransFusion'])
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['TransFusion'] = new_meter['TransFusion'].cpu().numpy()
            new_meter['Metric'] = stats
            writer.writerow(new_meter)
    df1 = pd.read_csv(file_latest % cfg.result_dir)

    if os.path.exists(file_stat % cfg.result_dir) is False:
        df1.to_csv(file_stat % cfg.result_dir, index=False)
    else:
        df2 = pd.read_csv(file_stat % cfg.result_dir)
        df = pd.concat([df2, df1['TransFusion']], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)