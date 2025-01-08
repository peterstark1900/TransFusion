import argparse
import sys
from utils import create_logger, seed_set
from utils.demo_visualize import demo_visualize
from utils.script import *
import numpy as np
sys.path.append(os.getcwd())
from config import Config, update_config
import torch
from tensorboardX import SummaryWriter
from utils.training import Trainer
from utils.evaluation import compute_stats

from data_loader.dataset_amass import DatasetAMASS

trainer = Trainer(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            cfg=cfg,
            logger=logger,
            tb_logger=tb_logger)
trainer.loop()