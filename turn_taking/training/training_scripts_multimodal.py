import os
import tqdm
from typing import Optional
import torch
torch.manual_seed(0)
import einops
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable
import turn_taking.model.multimodal_model as model_classes
# import turn_taking.model.model as model_classes
import yaml
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from dataset_management.dataset_manager.dataloader.dataloader import ChunkedDataset, AudioVisualDataset
from turn_taking.training.callbacks import FlipChannel, MaskVad, video_feature_subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# to kill allocated memory if needed... kills **all** running python script!!  kill $(ps -elf | grep python | awk '{print $4}') 


def log_gradients_in_model(model, logger, step):
    params = model.named_parameters()
    for tag, value in params:
        # print(tag, value.grad is not None)
        if value.grad is not None:
            try:
                logger.add_histogram(tag + "/grad", value.grad.cpu(), step)
                logger.add_histogram(tag + "/data", value.data.cpu(), step)
            except Exception as e:
                continue
    return


def train_one_epoch(model: nn.Module, feature_subset: str, dataloader: DataLoader, optimizer: Callable, epoch_idx: int, loss_fn, writer: Optional[SummaryWriter]=None, writer_prefix: Optional[str]=None):
    
    model.train()

    cfg = model.cfg

    running_loss = 0
    total_loss = 0
    last_loss = 0

    running_loss_vap = 0
    total_loss_vap = 0
    last_loss_vap = 0

    running_loss_vad = 0
    total_loss_vad = 0
    last_loss_vad = 0

    # some data flipping and masking of noise
    remove_crosstalk = MaskVad(probability=cfg['training_cfg']['mask_vad'], feature_hz=cfg['encoder_cfg']['audio_sample_rate'], audio_hz=cfg['encoder_cfg']['sample_rate'], scale=cfg['training_cfg']['mask_vad_scale'])
    channel_flip = FlipChannel(probability=cfg['training_cfg']['flip_channel'])

    pbar = tqdm.tqdm(total=len(dataloader))
    for i, batch in enumerate(dataloader):

        id, start, end, vad, vap, inverse_vap, audio_chunk, ch_l, ch_r, frames = batch
        batch = {"id": id, "start_time": start, "end_time": end, "vad": vad, "vap": vap, "inverse_vap": inverse_vap, "audio_chunk": audio_chunk, "channel_ids": [ch_l, ch_r], "frames": frames}

        batch = remove_crosstalk(batch)
        batch = channel_flip(batch)
        batch = video_feature_subset(batch, feature_subset)

        optimizer.zero_grad()

        vad_pred, vap_pred = model(batch)

        vad_true, vap_true = batch['vad'].to("cuda"), batch['vap'].to("cuda")
        loss, vap_loss, vad_loss = loss_fn(vad_true, vad_pred, vap_true, vap_pred)
        
        loss.backward()

        if cfg['training_cfg']['gradient_clip_val'] != -1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training_cfg']['gradient_clip_val'])
        optimizer.step()

        loss, vap_loss, vad_loss = loss.item(), vap_loss.item(), vad_loss.item()

        last_loss = loss/cfg['training_cfg']['batch_size']
        running_loss += loss/cfg['training_cfg']['batch_size']
        total_loss += loss/cfg['training_cfg']['batch_size']

        last_loss_vap = vap_loss/cfg['training_cfg']['batch_size']
        running_loss_vap += vap_loss/cfg['training_cfg']['batch_size']
        total_loss_vap += vap_loss/cfg['training_cfg']['batch_size']

        last_loss_vad = vad_loss/cfg['training_cfg']['batch_size']
        running_loss_vad += vad_loss/cfg['training_cfg']['batch_size']
        total_loss_vad += vad_loss/cfg['training_cfg']['batch_size']

        if i%10 == 0 and i > 1:

            writer.add_scalar(f"loss", last_loss, i + (epoch_idx-1)*len(dataloader))
            writer.add_scalar(f"running_loss", running_loss/10, i + (epoch_idx-1)*len(dataloader))
            log_gradients_in_model(model, writer, i + (epoch_idx-1)*len(dataloader))
            running_loss = 0

            writer.add_scalar(f"vad_loss", last_loss_vad, i + (epoch_idx-1)*len(dataloader))
            writer.add_scalar(f"running_vad_loss", running_loss_vad/10, i + (epoch_idx-1)*len(dataloader))
            running_loss_vad = 0

            writer.add_scalar(f"vap_loss", last_loss_vap, i + (epoch_idx-1)*len(dataloader))
            writer.add_scalar(f"running_vap_loss", running_loss_vap/10, i + (epoch_idx-1)*len(dataloader))
            running_loss_vap = 0

            # writer.add_graph(model, (x_l, x_r))

        pbar.update()
        # torch.cuda.empty_cache()

    pbar.close()
    return total_loss/i, total_loss_vad/i, total_loss_vap/i


def validation_loss(dataloader, feature_subset, model, loss_fn):

    model.eval()
    running_loss = 0
    running_vap_loss = 0
    running_vad_loss = 0

    # keep enabled for validation?
    cfg = model.cfg
    remove_crosstalk = MaskVad(probability=cfg['training_cfg']['mask_vad'], feature_hz=cfg['encoder_cfg']['audio_sample_rate'], audio_hz=cfg['encoder_cfg']['sample_rate'], scale=cfg['training_cfg']['mask_vad_scale'])

    pbar = tqdm.tqdm(total=len(dataloader))
    for i, batch in enumerate(dataloader):

        id, start, end, vad, vap, inverse_vap, audio_chunk, ch_r, ch_r, frames = batch
        batch = {"id": id, "start_time": start, "end_time": end, "vad": vad, "vap": vap, "inverse_vap": inverse_vap, "audio_chunk": audio_chunk, "channel_ids": [ch_r, ch_r], "frames": frames}

        with torch.no_grad():
            
            batch = remove_crosstalk(batch)
            batch = video_feature_subset(batch, feature_subset)

            vad_pred, vap_pred = model(batch)

            vad_true, vap_true = batch['vad'].to("cuda"), batch['vap'].to("cuda")
            loss, vap_loss, vad_loss = loss_fn(vad_true, vad_pred, vap_true, vap_pred)

        running_loss += loss/cfg['training_cfg']['batch_size']
        running_vap_loss += vap_loss/cfg['training_cfg']['batch_size']
        running_vad_loss += vad_loss/cfg['training_cfg']['batch_size']

        pbar.update()
        # torch.cuda.empty_cache()

    pbar.close()

    # divide by i to make it comparable across differnt folds in k fold
    return running_loss/i, running_vap_loss/i, running_vad_loss/i
    

def train_n_epochs(model, feature_set, optimizer, scheduler, dataloader: DataLoader, val_dataloader: DataLoader, loss_fn, num_epochs: int, writer: SummaryWriter, writer_prefix: str, fold: str, resume_epoch, resume_fold):

    if fold == resume_fold and resume_epoch is not None:
        resume_epoch_idx = int(resume_epoch.split('_')[-1])
        epochs = range(resume_epoch_idx+1, num_epochs+1)
        resume_epoch_idx = None
        resume_epoch, resume_fold = None, None
    else:
        epochs = range(1, num_epochs+1)
        resume_epoch = None
        resume_fold = None

    for i in epochs:
        
        print("train")
        total_loss, total_loss_vad, total_loss_vap = train_one_epoch(model, feature_set, dataloader, optimizer, i, loss_fn, writer, f"{writer_prefix}"+f"_epoch_{i}")
        writer.add_scalar(f"TotalTrainLoss", total_loss, i)
        writer.add_scalar(f"TotalTrainLoss_Vap", total_loss_vad, i)
        writer.add_scalar(f"TotalTrainLoss_Vad", total_loss_vap, i)

        torch.save(model.state_dict(), f"{writer_prefix}"+f"_epoch_{i}")

        print("val")
        val_loss, val_vap_loss, val_vad_loss = validation_loss(val_dataloader, feature_set, model, loss_fn)

        if scheduler:
            scheduler.step(val_loss)

        writer.add_scalar(f"ValLossTotal", val_loss, i)
        writer.add_scalar(f"ValLossVAP", val_vap_loss, i)
        writer.add_scalar(f"ValLossVAD", val_vad_loss, i)


    return resume_epoch, resume_fold


def K_folds(top_prefix, feature_set, batch_size, learning_rate, loss_fn, cfg, n_folds, resume_epoch, resume_fold):

    folds = [f'fold_{i}' for i in range(0, n_folds)]
    if resume_fold is not None:
        idx = int(resume_fold.split('_')[-1])
        folds = [f'fold_{i}' for i in range(idx, n_folds)]

    for fold in folds:

        # re-initialise model weights each fold

        # model
        model = getattr(model_classes, model_name)(cfg).to(cfg['device'])

        # optimiser
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.999], weight_decay=0.001)

        # lr scheduler
        if cfg['training_cfg']['lr_scheduler']:
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        else:
            scheduler = None
            print("no scheduler")
        
        print("fold ", fold)
        writer_prefix = top_prefix + f"_{fold}"
        writer = SummaryWriter(writer_prefix)

        # load where picked up 
        if resume_epoch is not None:
            model = getattr(model_classes, model_name)(cfg).to(cfg['device'])
            checkpoint = writer_prefix + '_' + resume_epoch
            model.load_state_dict(state_dict=torch.load(checkpoint, map_location ='cpu'))

        pickle_file = os.path.join(folds_dir, fold, "train")
        val_pickle_file = os.path.join(folds_dir, fold, "val")

        # dataset:  train
        audio_dataset = AudioVisualDataset(video_directory, video_format, channelmaps, pickle_file=pickle_file, mode=cfg['model_cfg']['mode'], wavdir=wavdir)
        dataloader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=8, prefetch_factor=4)

        # dataset:  val
        val_audio_dataset = AudioVisualDataset(video_directory, video_format, channelmaps, pickle_file=val_pickle_file, mode=cfg['model_cfg']['mode'], wavdir=wavdir)
        val_dataloader = DataLoader(val_audio_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=8, prefetch_factor=4)

        resume_epoch, resume_fold = train_n_epochs(model, feature_set, optimizer, scheduler, dataloader, val_dataloader, loss_fn, cfg['training_cfg']['n_epochs'], writer, writer_prefix, fold, resume_epoch, resume_fold)


if __name__ == "__main__":    

    corpus = 'candor'
    logdir = "/data/ssd1/russelsa/checkpoints"
    video_directory = 'turn-taking-projects/corpora/candor/candor_openface_pkl/'
    video_format = '.pkl' 
    channelmaps = 'turn-taking-projects/corpora/candor/channelmaps.pkl'

    # feature set 

    # GPU 0
    # feature_set = 'gaze'
    feature_set = 'faus'

    # # GPU 1
    # feature_set = 'lmks'
    # feature_set = 'pose'

    timestamp = None
    resume_fold = None
    resume_epoch = None

    assert feature_set in ['all', 'pose', 'gaze', 'lmks', 'faus', 'faus+lmks']
    
    model_name = "LateVAFusion"
    run_name = f"{model_name}_{feature_set}"

    config_file = "turn_taking/assets/config.yaml"
    wavdir = f"turn-taking-projects/corpora/{corpus}/{corpus}_wav"

    # model config
    cfg = yaml.safe_load(open(config_file))
    torch.cuda.set_device(cfg['device'])

    # determine the model dimensionality from the feature set
    if feature_set != 'all':
        
        dim = 0

        if 'pose' in feature_set:
            dim += 6
        if 'gaze' in feature_set:
            dim += 6
        if 'lmks' in feature_set:
            dim += 30
        if 'faus' in feature_set:
            # faus + confidence
            dim += 18

        # modify 
        cfg['feature_set'] = feature_set
        cfg['multimodal_model_cfg']['video']['d_model'] = dim
        cfg['multimodal_model_cfg']['video']['d_ffn'] = dim * 3

        if  dim % cfg['multimodal_model_cfg']['video']['n_heads'] != 0:
            cfg['multimodal_model_cfg']['video']['n_heads'] = 2

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
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    top_prefix = f"runs_{cfg['model_cfg']['mode']}_{corpus}_{run_name}/{timestamp}"
    top_prefix = os.path.join(logdir, top_prefix)

    print(f"run {run_name} dir {top_prefix}")

    # train K folds
    folds_dir = f"/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/new_folds/{corpus}"

    # save the configuration
    try:
        os.makedirs(top_prefix)
    except FileExistsError:
        pass
    yaml.safe_dump(cfg, open(top_prefix+'_params.yaml', 'w+'))

    K_folds(top_prefix, feature_set, loss_fn=loss_fn, cfg=cfg, batch_size=cfg['training_cfg']['batch_size'], learning_rate=cfg['training_cfg']['learning_rate'], n_folds=5, resume_epoch=resume_epoch, resume_fold=resume_fold)