import einops
from turn_taking.model.model import GPT, GPTStereo, Combinator
from turn_taking.audio_encoders.encoders import StereoEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader



class VideoOnlyModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        
        self.cfg=cfg

        # encoder
        self.audio_encoder = StereoEncoder().to("cuda")
        
        # for the individual channels
        self.self_attention_transformer = GPT(
            dim=cfg['model_cfg']['d_model'],
            dff_k=cfg['model_cfg']['dff_k'],
            num_layers=cfg['model_cfg']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
        )

        # for channel information
        self.cross_attention_transformer = GPTStereo(
            dim=cfg['model_cfg']['d_model'],
            dff_k=cfg['model_cfg']['dff_k'],
            num_layers=cfg['model_cfg']['n_layers_cross'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
        )

        self.vad_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=1)
        self.vap_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=cfg['model_cfg']['d_out'])

        self.upsampler = nn.Upsample(scale_factor=1.67, mode='linear', align_corners=True)

    def upsample(self, x):
        B, N, D, C = x.shape
        x=einops.rearrange(x,"b n d c -> (b c) d n")
        x=self.upsampler(x)
        x=einops.rearrange(x,"(b c) d n -> b n d c", b=B, c=C)
        x = x[:, :1000, :]
        return x

    def forward(self, batch):
        
        with torch.no_grad():
            # audio = self.audio_encoder(batch['audio_chunk'].to("cuda"))
            # x_l, x_r = audio[:,:,:,0].to("cuda"), audio[:,:,:,1].to("cuda")
            # upsample the video
            # assert not torch.isnan(batch['frames']).any()==1
            batch['frames']=torch.nan_to_num(batch['frames'])
            video = self.upsample(batch['frames']).to(device='cuda')
            x_l, x_r = video[:,:,:,0].to("cuda"), video[:,:,:,1].to("cuda")

        # pass thru layers
        x_l = self.self_attention_transformer(x_l, return_attention=False)
        x_r = self.self_attention_transformer(x_r, return_attention=False)
        out = self.cross_attention_transformer(x_l["x"], x_r["x"], return_attention=False)

        # output
        v1 = self.vad_classifier(out["x1"])
        v2 = self.vad_classifier(out["x2"])

        vad = torch.cat((v1, v2), dim=-1)
        vap = self.vap_classifier(out['x'])

        return vad, vap


class Early_VA_Combiner(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # audio self attention
        self.self_attn_audio = GPT(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['audio']['n_layers_self'],
            num_heads=cfg['multimodal_model_cfg']['video']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # video self attn
        self.self_attn_video = GPT(
            dim=cfg['multimodal_model_cfg']['video']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['video']['n_layers_self'],
            num_heads=cfg['multimodal_model_cfg']['video']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # project video to audio dim
        self.video_proj_ln = nn.LayerNorm(cfg['multimodal_model_cfg']['audio']['d_model'])
        self.video_proj = nn.Linear(in_features=cfg['multimodal_model_cfg']['video']['d_model'], out_features=cfg['multimodal_model_cfg']['audio']['d_model'], bias=False)

        # cross attention of video with audio
        self.av_cross_attn = GPTStereo(
            dim=cfg['multimodal_model_cfg']['fuse_av']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_av']['n_layers_cross'],
            num_heads=cfg['multimodal_model_cfg']['fuse_av']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )   

        # Activation
        self.activation = getattr(nn, cfg['multimodal_model_cfg']['ffn_activation'])()

    def forward(self, x_l, x_r, v_l, v_r):
        
        # audio self attention
        x_l = self.self_attn_audio(x_l, return_attention=False)
        x_r = self.self_attn_audio(x_r, return_attention=False)

        # video 
        v_l = self.self_attn_video(v_l, return_attention=False)
        v_r = self.self_attn_video(v_r, return_attention=False)

        v_l['x'] = self.activation(self.video_proj_ln(self.video_proj(v_l['x'])))
        v_r['x'] = self.activation(self.video_proj_ln(self.video_proj(v_r['x'])))

        # cross attention
        out_l = self.av_cross_attn(x_l["x"], v_l["x"], return_attention=False)
        out_r = self.av_cross_attn(x_r["x"], v_r["x"], return_attention=False)

        return out_l, out_r


class Late_VA_Combiner(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # audio self attention
        self.self_attn_audio = GPT(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['audio']['n_layers_self'],
            num_heads=cfg['multimodal_model_cfg']['video']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # video self attn
        self.self_attn_video = GPT(
            dim=cfg['multimodal_model_cfg']['video']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['video']['n_layers_self'],
            num_heads=cfg['multimodal_model_cfg']['video']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # project video to audio dim
        self.video_proj_ln = nn.LayerNorm(cfg['multimodal_model_cfg']['audio']['d_model'])
        self.video_proj = nn.Linear(in_features=cfg['multimodal_model_cfg']['video']['d_model'], out_features=cfg['multimodal_model_cfg']['audio']['d_model'])

        # cross attention of video with audio
        self.cross_attn = GPTStereo(
            dim=cfg['multimodal_model_cfg']['fuse_av']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_av']['n_layers_cross'],
            num_heads=cfg['multimodal_model_cfg']['fuse_av']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # Activation
        self.activation = getattr(nn, cfg['multimodal_model_cfg']['ffn_activation'])()

    def forward(self, x_l, x_r, v_l, v_r):
        
        # audio self attention
        x_l = self.self_attn_audio(x_l, return_attention=False)
        x_r = self.self_attn_audio(x_r, return_attention=False)

        # video 
        v_l = self.self_attn_video(v_l, return_attention=False)
        v_r = self.self_attn_video(v_r, return_attention=False)

        v_l['x'] = self.activation(self.video_proj_ln(self.video_proj(v_l['x'])))
        v_r['x'] = self.activation(self.video_proj_ln(self.video_proj(v_r['x'])))

        # cross attention
        out_a = self.cross_attn(x_l["x"], x_r["x"], return_attention=False)
        out_v = self.cross_attn(v_l["x"], v_r["x"], return_attention=False)

        return out_a, out_v


class EarlyVAFusion(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        
        self.cfg=cfg
        
        # VA fusion
        self.va_fusion = Early_VA_Combiner(self.cfg)

        # encoder
        self.audio_encoder = StereoEncoder().to("cuda")

        # cross attention
        self.cross_attn = GPTStereo(
            dim=cfg['multimodal_model_cfg']['fuse_speakers']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_speakers']['n_layers_cross'],
            num_heads=cfg['multimodal_model_cfg']['fuse_speakers']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # for upsampling the N dimension of the video
        scale_factor = cfg['multimodal_model_cfg']['audio']['sequence_len'] / cfg['multimodal_model_cfg']['video']['sequence_len']
        scale_factor = scale_factor
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)

        self.video_proj = nn.Linear(in_features=cfg['multimodal_model_cfg']['video']['d_model'], out_features=cfg['multimodal_model_cfg']['audio']['d_model'])

        self.vad_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=1)
        self.vap_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=cfg['model_cfg']['d_out'])
    
    def upsample(self, x):
        B, N, D, C = x.shape
        x=einops.rearrange(x,"b n d c -> (b c) d n")
        x=self.upsampler(x)
        x=einops.rearrange(x,"(b c) d n -> b n d c", b=B, c=C)
        return x


    def forward(self, batch):

        with torch.no_grad():

            # get the L and R channels of the audio
            audio = self.audio_encoder(batch['audio_chunk'].to("cuda"))
            x_l, x_r = audio[:,:,:,0].to("cuda"), audio[:,:,:,1].to("cuda")

            # upsample the video
            batch['frames']=torch.nan_to_num(batch['frames'])
            video = self.upsample(batch['frames']).to(device='cuda')
            v_l, v_r = video[:,:,:,0].to("cuda"), video[:,:,:,1].to("cuda")

        # fuse A and V from each speaker independently
        s_l, s_r = self.va_fusion(x_l, x_r, v_l, v_r)

        out = self.cross_attn(s_l['x'], s_r['x'])

        # output
        v1 = self.vad_classifier(out["x1"])
        v2 = self.vad_classifier(out["x2"])

        vad = torch.cat((v1, v2), dim=-1)
        vap = self.vap_classifier(out['x'])

        return vad, vap


class LateVAFusion(nn.Module):


    def __init__(self, cfg):
        super().__init__()
        
        self.cfg=cfg
        
        # VA fusion
        self.fusion = Early_VA_Combiner(self.cfg)

        # encoder
        self.audio_encoder = StereoEncoder().to("cuda")

        # cross attention
        self.cross_attn = GPTStereo(
            dim=cfg['multimodal_model_cfg']['fuse_speakers']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_speakers']['n_layers_cross'],
            num_heads=cfg['multimodal_model_cfg']['fuse_speakers']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # for upsampling the N dimension of the video
        scale_factor = cfg['multimodal_model_cfg']['audio']['sequence_len'] / cfg['multimodal_model_cfg']['video']['sequence_len']
        scale_factor = scale_factor
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)

        self.video_proj = nn.Linear(in_features=cfg['multimodal_model_cfg']['video']['d_model'], out_features=cfg['multimodal_model_cfg']['audio']['d_model'], bias=False)

        self.vad_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=2)
        self.vap_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=cfg['model_cfg']['d_out'])
        
        self.combine_speakers = Combinator(cfg['model_cfg']['d_model'])
    
    def upsample(self, x):
        B, N, D, C = x.shape
        x=einops.rearrange(x,"b n d c -> (b c) d n")
        x=self.upsampler(x)
        x=einops.rearrange(x,"(b c) d n -> b n d c", b=B, c=C)
        return x


    def forward(self, batch):

        with torch.no_grad():

            # get the L and R channels of the audio
            audio = self.audio_encoder(batch['audio_chunk'].to("cuda"))
            x_l, x_r = audio[:,:,:,0].to("cuda"), audio[:,:,:,1].to("cuda")

            # upsample the video
            batch['frames']=torch.nan_to_num(batch['frames'])
            video = self.upsample(batch['frames']).to(device='cuda')
            v_l, v_r = video[:,:,:,0].to("cuda"), video[:,:,:,1].to("cuda")

        # fuse A and V from each speaker independently
        v, a = self.fusion(x_l, x_r, v_l, v_r)

        out = self.cross_attn(v['x'], a['x'])

        x = self.combine_speakers(out["x1"], out["x2"])

        # output
        vad = self.vad_classifier(x)

        # vad = torch.cat((v1, v2), dim=-1)
        vap = self.vap_classifier(out['x'])

        return vad, vap


class LateConcatenationModel(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg=cfg

        # encoder
        self.audio_encoder = StereoEncoder().to("cuda")
        
        # for the individual channels
        self.self_attention_transformer_audio = GPT(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['model_cfg']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
        )

        # for the individual channels
        self.self_attention_transformer_video = GPT(
            dim=cfg['multimodal_model_cfg']['video']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['model_cfg']['n_layers_self'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
        )

        # for channel information
        self.cross_attention_transformer = GPTStereo(
            dim=316,
            dff_k=316*3,
            num_layers=cfg['model_cfg']['n_layers_cross'],
            num_heads=cfg['model_cfg']['n_heads'],
            activation=cfg['model_cfg']['ffn_activation'],
            dropout=cfg['model_cfg']['dropout_prob']
        )

        self.vad_classifier = nn.Linear(in_features=316, out_features=1)
        self.vap_classifier = nn.Linear(in_features=316, out_features=cfg['model_cfg']['d_out'])

        self.upsampler = nn.Upsample(scale_factor=1.67, mode='linear', align_corners=True)

    def upsample(self, x):
        B, N, D, C = x.shape
        x=einops.rearrange(x,"b n d c -> (b c) d n")
        x=self.upsampler(x)
        x=einops.rearrange(x,"(b c) d n -> b n d c", b=B, c=C)
        x = x[:, :1000, :]
        return x

    def forward(self, batch):
        
        with torch.no_grad():
            audio = self.audio_encoder(batch['audio_chunk'].to("cuda"))
            x_l, x_r = audio[:,:,:,0], audio[:,:,:,1]
            
            # upsample the video
            batch['frames']=torch.nan_to_num(batch['frames'])
            video = self.upsample(batch['frames']).to(device='cuda')
            v_l, v_r = video[:,:,:,0], video[:,:,:,1]

        # pass thru layers
        x_l = self.self_attention_transformer_audio(x_l, return_attention=False)
        x_r = self.self_attention_transformer_audio(x_r, return_attention=False)

        v_l = self.self_attention_transformer_video(v_l, return_attention=False)
        v_r = self.self_attention_transformer_video(v_r, return_attention=False)

        x_l = torch.concat((x_l["x"], v_l["x"]),dim=2)
        x_r = torch.concat((x_r["x"], v_r["x"]),dim=2)

        out = self.cross_attention_transformer(x_l, x_r, return_attention=False)

        # output
        v1 = self.vad_classifier(out["x1"])
        v2 = self.vad_classifier(out["x2"])

        vad = torch.cat((v1, v2), dim=-1)
        vap = self.vap_classifier(out['x'])

        return vad, vap


def test_forward_pass():
    from dataset_management.dataset_manager.dataloader.dataloader import AudioVisualDataset
    import yaml


    config_file = "turn_taking/assets/config.yaml"
    cfg = yaml.safe_load(open(config_file))

    pickle_file = "/home/russelsa@ad.mee.tcd.ie/github/dataset_management/dataset_manager/assets/new_folds/candor/fold_0/train.pkl"
    wavdir = "turn-taking-projects/corpora/candor/candor_wav"
    video_directory = "turn-taking-projects/corpora/candor/candor_openface_pkl"
    channelmaps = "turn-taking-projects/corpora/candor/channelmaps"

    av_ds = AudioVisualDataset(pickle_file=pickle_file, video_directory=video_directory, wavdir=wavdir, channelmaps=channelmaps, video_format='.pkl', mode='VAP')
    dl = DataLoader(av_ds, batch_size=16, shuffle=False, pin_memory=True)

    while True:
        batch = next(iter(dl))
    
        model_test = BasicMultimodalModel(cfg).to("cuda")
        model_test(batch)

        x=1


if __name__=="__main__":
    test_forward_pass()
