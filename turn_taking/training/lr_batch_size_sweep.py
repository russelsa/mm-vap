import torch
import einops
import os
import yaml
from datetime import datetime
from turn_taking.training.callbacks import MaskVad
from turn_taking.model.model import StereoTransformerModel
from turn_taking.training.training_scripts import K_folds, train_n_epochs
from dataset_management.dataset_manager.dataloader.dataloader import AudioDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":    

    corpus = 'switchboard'

    config_file = "turn_taking/assets/config.yaml"
    wavdir = f"turn-taking-projects/corpora/{corpus}/{corpus}_wav"

    # model config
    cfg = yaml.safe_load(open(config_file))
    torch.cuda.set_device(cfg['device'])
    remove_crosstalk = MaskVad(probability=cfg['training_cfg']['mask_vad'], feature_hz=cfg['encoder_cfg']['audio_sample_rate'], audio_hz=cfg['encoder_cfg']['sample_rate'], scale=cfg['training_cfg']['mask_vad_scale'])

    # loss
    mode=cfg['model_cfg']['mode']
    if mode == 'ind-40' or mode == 'ind-4':
        def loss_fn(vad_true, vad_pred, vap_true, vap_pred): 
            # vad_true, vad_pred, vap_true, vap_pred
            vap_loss = F.binary_cross_entropy_with_logits(vap_pred, vap_true)
            vad_loss = F.binary_cross_entropy_with_logits(vad_pred, vad_true)
            loss = vap_loss + vad_loss
            return loss, vap_loss, vad_loss
    else:
        def loss_fn(vad_true, vad_pred, vap_true, vap_pred): 
            vap_pred = einops.rearrange(vap_pred, "b n d -> (b n) d")
            vap_true = einops.rearrange(vap_true, "b n -> (b n)")
            vap_loss = F.cross_entropy(vap_pred, vap_true)
            vad_loss = F.binary_cross_entropy_with_logits(vad_pred, vad_true)
            loss = vap_loss + vad_loss
            return loss, vap_loss, vad_loss


    # timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    top_prefix = f"/data/ssd1/russelsa/runs_{cfg['model_cfg']['mode']}_{corpus}_lr_sweep_phonwords_with_crosstalk/{timestamp}"

    # train K folds
    folds_dir = f"dataset_management/dataset_manager/assets/folds/{corpus}_phonwords"

    # save the configuration
    try:
        os.makedirs(top_prefix)
    except FileExistsError:
        pass
    yaml.safe_dump(cfg, open(top_prefix+'_params.yaml', 'w+'))

    for batch_size in [4]:
        for learning_rate in [3.63e-4, 1e-3, 1e-4, 1e-5, 1e-6]: # 1e-2, diverges

            prefix = top_prefix+f'/{learning_rate}_{batch_size}'
            writer = SummaryWriter(prefix)

            # files
            train_pickle = f"dataset_management/dataset_manager/assets/folds/{corpus}/fold_0/train.pkl"
            val_pickle = f"dataset_management/dataset_manager/assets/folds/{corpus}/fold_0/val.pkl"

            # model
            model = StereoTransformerModel(cfg).to(cfg['device'])

            # optimizer
            optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.999], weight_decay=0.001)
            scheduler=None

            # train set
            audio_dataset = AudioDataset(train_pickle, mode=cfg['model_cfg']['mode'], wavdir=wavdir)
            dataloader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4)

            # val set
            val_audio_dataset = AudioDataset(val_pickle, mode=cfg['model_cfg']['mode'], wavdir=wavdir)
            val_dataloader = DataLoader(val_audio_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4)

            train_n_epochs(model, 
                           loss_fn=loss_fn, 
                           optimizer=optimizer, 
                           scheduler=scheduler, 
                           dataloader=dataloader, 
                           val_dataloader=val_dataloader, 
                           num_epochs=cfg['training_cfg']['n_epochs'], 
                           writer_prefix=prefix, 
                           writer=writer
                           )
