import torch
import numpy as np
from model.temporal import TemporalUnet
from model.diffusion import GaussianDiffusion
from utils.training import Trainer
from utils.dataset import SequenceDataset
from utils.arrays import *
import warnings
warnings.filterwarnings('ignore')


def main():
    device = 'cuda'
    data_path = 'data/sample_data.pkl'
    train_dataset = SequenceDataset(data_path, horizon=100)

    model = TemporalUnet(horizon=100,
                         transition_dim=4+2,
                         dim=64,
                         cond_dim=4,
                         dim_mults=(1, 2, 4, 8),
                         attention=False).to(device)

    diffusion = GaussianDiffusion(model,
                                  horizon=100,
                                  observation_dim=4,
                                  action_dim=2,
                                  n_timesteps=20,
                                  loss_type='l2',
                                  clip_denoised=False,
                                  predict_epsilon=False,
                                  ## loss weighting
                                  action_weight=10,
                                  loss_weights=None,
                                  loss_discount=1).to(device)

    trainer = Trainer(diffusion,
                      dataset=train_dataset,
                      train_batch_size=32,
                      train_lr=2e-4,
                      gradient_accumulate_every=2,
                      ema_decay=0.995,
                      sample_freq=20000,
                      save_freq=4000,
                      label_freq=int(1E6 // 5),
                      save_parallel=False,
                      results_folder='results',
                      bucket=None,
                      n_reference=8)

    print('Testing forward...', end=' ', flush=True)
    batch = batchify(train_dataset[0])
    loss, _ = diffusion.loss(*batch)
    loss.backward()
    print('✓')

    # Main Loop
    n_epochs = int(1e6 // 10000)

    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} |')
        trainer.train(n_train_steps=10000)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
