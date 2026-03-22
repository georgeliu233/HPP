import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PositionalEncoding

from .aggregator import CrossAttention, OccFormer, SwinOccFormerStage

import math 


def gen_sineembed_for_position2d(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

class DecodeUpsample(nn.Module):
    def __init__(self, input_dim, kernel, timestep):
        super(DecodeUpsample, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(input_dim, input_dim//2, (1, kernel, kernel), padding='same'), nn.GELU())
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2))
        self.residual_conv = nn.Sequential(nn.Conv3d(input_dim//2, input_dim//2, (timestep, 1, 1)), nn.GELU())

    def forward(self, inputs, res):
        #b, t, c, h, w = inputs.shape
        inputs = self.upsample(inputs)
        inputs = self.conv(inputs) + self.residual_conv(res)
        return inputs

class ResDecodeUpsample(nn.Module):
    def __init__(self, input_dim, kernel, timestep):
        super(ResDecodeUpsample, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(input_dim, input_dim//2, (1, kernel, kernel), padding='same'), nn.GELU())
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2))

    def forward(self, inputs):
        #b, t, c, h, w = inputs.shape
        inputs = self.upsample(inputs)
        inputs = self.conv(inputs)
        return inputs

class DeepQuery(nn.Module):
    def __init__(self, res=(8, 8), input_dim=768, timestep=8):
        super(DeepQuery, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(input_dim, input_dim//2, (1, 3, 3), padding='same'), nn.GELU())
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2))
        self.temp_query = nn.Parameter(torch.zeros(1, input_dim, timestep, res[0], res[1]), requires_grad=True)
        nn.init.kaiming_uniform_(self.temp_query)
        self.timestep = timestep
    
    def forward(self, deep_inputs):
        b, c, h, w = deep_inputs.shape
        deep_inputs = deep_inputs.unsqueeze(2).expand(-1, -1, self.timestep, -1, -1)
        deep_inputs = deep_inputs + self.temp_query.expand(b, -1, -1, -1, -1)
        deep_inputs = self.conv(self.upsample(deep_inputs))
        return deep_inputs
    
from .FG_TSA import gen_sineembed_for_position

class PredFinalDecoder(nn.Module):
    def __init__(self, input_dim, kernel=3, large_scale=False,planning=False):
        super(PredFinalDecoder, self).__init__()
        '''
        input h,w = 128
        dual deconv for flow and ogms
        ''' 
        self.input_dim = input_dim
        if large_scale:
            self.ogm_conv = nn.Conv3d(input_dim, 4, (1, kernel, kernel), padding='same')
            self.flow_conv = nn.Conv3d(input_dim, 2, (1, kernel, kernel), padding='same')
        else:
            self.ogm_conv = nn.Sequential(nn.Conv3d(input_dim, input_dim//2, (1, kernel, kernel), padding='same'), 
                            nn.GELU(), nn.Upsample(scale_factor=(1, 2, 2)),
                            nn.Conv3d(input_dim//2, 4 if planning else 2, (1, kernel, kernel), padding='same'))

            self.flow_conv = nn.Sequential(nn.Conv3d(input_dim, input_dim//2, (1, kernel, kernel), padding='same'), 
                            nn.GELU(), nn.Upsample(scale_factor=(1, 2, 2)),
                            nn.Conv3d(input_dim//2, 2, (1, kernel, kernel), padding='same'))
    
    def forward(self, inputs ,flow=None):
        ogms = self.ogm_conv(inputs)
        if flow is not None:
            b, c, t, h, w = inputs.shape
            flow = gen_sineembed_for_position(flow, self.input_dim // 3)
            flow = flow.reshape(b, t, h//2, w//2, c).permute(0, 4, 1, 2, 3)
            flow = F.interpolate(flow, scale_factor=(1, 2, 2))
            inputs = inputs + flow
        flows = self.flow_conv(inputs)
        return torch.cat([ogms, flows], dim=1)

class STrajNetDecoder(nn.Module):
    def __init__(self, dim=384, heads=8, resolutions=16, len_fpn=2, kernel=3, timestep=5, dropout=0.1,
        flow_pred=False, large_scale=False):
        super(STrajNetDecoder, self).__init__()

        self.timestep = timestep
        self.len_fpn = len_fpn
        self.residual_conv = nn.Sequential(nn.Conv3d(dim, dim, (timestep, 1, 1)), nn.GELU())
        self.aggregator = nn.ModuleList([CrossAttention(dim, heads, dropout) for _ in range(timestep)])

        self.actor_layer = nn.Sequential(nn.Linear(256, dim), nn.GELU())

        self.fpn_decoders = nn.ModuleList([
            DecodeUpsample(dim // (2 ** i), kernel, timestep) for i in range(len_fpn)
        ])

        self.upsample = nn.Upsample(scale_factor=(1, 2, 2))
        if flow_pred:
            self.output_conv = PredFinalDecoder(dim // (2 ** len_fpn),large_scale=large_scale)
        else:
            self.output_conv = nn.Conv3d(dim // (2 ** len_fpn), 4, (1, kernel, kernel), padding='same')
    
    def forward(self, output_list, actor, actor_mask):

        # Aggregations:
        enc_output = output_list[-1]
        # print('enc_outt',torch.isnan(enc_output).any())
        b, c, h, w = enc_output.shape
        res_output = enc_output.unsqueeze(2).expand(-1, -1, self.timestep, -1, -1)
        enc_output = enc_output.reshape(b, c, h*w).permute(0, 2, 1)
        #[b, t, h*w, c]
        enc_output = enc_output.unsqueeze(1).expand(-1, self.timestep, -1, -1)
        #actor_mask:[b, h*w, a, c]
        # new_actor_mask = actor_mask.unsqueeze(1).expand(-1, h*w, -1)
        actor = self.actor_layer(actor)
        actor_mask[:, 0] = False

        agg_output =  torch.stack([self.aggregator[i](enc_output[:, i], actor, actor_mask) for i in range(self.timestep)], dim=2)
        agg_output = agg_output.permute(0, 3, 2, 1).reshape(b, -1, self.timestep, h, w)

        decode_output = agg_output + self.residual_conv(res_output)
        
        # fpn decoding:
        for j in range(self.len_fpn):
            decode_output = self.fpn_decoders[j](decode_output, output_list[-2-j].unsqueeze(2).expand(-1, -1, self.timestep, -1, -1))
        decode_output = self.output_conv(self.upsample(decode_output))

        #[b, t, c, h, w]
        return decode_output


class OccFormerDecoder(nn.Module):
    def __init__(self, dim=384, heads=8, resolutions=16, len_fpn=2, kernel=3, timestep=5, dropout=0.1,
        flow_warp=False,offset_downscale_factor=2,flow_pred=False,large_scale=False):
        super(OccFormerDecoder, self).__init__()

        self.timestep = timestep
        self.len_fpn = len_fpn
        self.residual_conv = nn.Sequential(nn.Conv3d(dim, dim, (timestep, 1, 1)), nn.GELU())
        self.aggregator = OccFormer(dim, resolutions, timestep, heads, dropout, flow_warp=flow_warp, 
                                offset_downscale_factor=offset_downscale_factor)

        self.fpn_decoders = nn.ModuleList([
            DecodeUpsample(dim // (2 ** i), kernel, timestep) for i in range(len_fpn)
        ])

        self.upsample = nn.Upsample(scale_factor=(1, 2, 2))
        if flow_pred:
            self.output_conv = PredFinalDecoder(dim // (2 ** len_fpn),large_scale=large_scale)
        else:
            self.output_conv = nn.Conv3d(dim // (2 ** len_fpn), 4, (1, kernel, kernel), padding='same')
    
    def forward(self, output_list, actor, actor_mask):
        # Aggregations:
        enc_output = output_list[-1]
        b, c, h, w = enc_output.shape
        res_output = enc_output.unsqueeze(2).expand(-1, -1, self.timestep, -1, -1)
        
        agg_output = self.aggregator(enc_output, actor, actor_mask)

        decode_output = agg_output + self.residual_conv(res_output)
        
        # fpn decoding:
        for j in range(self.len_fpn):
            decode_output = self.fpn_decoders[j](decode_output, output_list[-2-j].unsqueeze(2).expand(-1, -1, self.timestep, -1, -1))
        decode_output = self.output_conv(self.upsample(decode_output))

        #[b, t, c, h, w]
        return decode_output


from .FG_TSA import TemporalFG_TSALayer

class SwinOccFormerDecoder(nn.Module):
    def __init__(self, dim=384, heads=8, resolutions=16, len_fpn=3, kernel=3, timestep=5, dropout=0.1,head_list=[12, 6],
        offset_downscale_factor=2, glb_flow_warp=False,local_flow_warp=False, flow_pred=False,
        large_scale=False, large_res=False,ms_proj=False,planning=False,fg_tsa=False):
        super(SwinOccFormerDecoder, self).__init__()

        self.timestep = timestep
        self.len_fpn = len_fpn
        self.large_res = large_res
        self.ms_proj = ms_proj
        self.fg_tsa = fg_tsa

        if self.large_res:
            self.deep_query = DeepQuery(timestep=timestep)
        
        if self.fg_tsa:
            heads_fg_tsa = [12, 6, 3]
            self.fg_tsa_layers = nn.ModuleList([TemporalFG_TSALayer(embed_dims=dim//(2**i),num_heads=heads_fg_tsa[i],
            num_points=4) for i in range(len_fpn)])
        
        if self.ms_proj:
            self.proj_list = nn.ModuleList([
                nn.Sequential(nn.Conv3d(dim//(2**i), dim//(2**(i+1)), (1, kernel, kernel), padding='same'), nn.GELU(),
                nn.Conv3d(dim//(2**(i+1)), 1, (1, kernel, kernel), padding='same')) 
                for i in range(len_fpn)
            ])

        self.global_occformer = OccFormer(dim, resolutions, timestep, heads, dropout, offset_downscale_factor, glb_flow_warp)
        self.resnet_deconvs = nn.ModuleList([ResDecodeUpsample(dim//(2**i), kernel, timestep) for i in range(len_fpn)])
        assert len(head_list) == len_fpn - 1, f'length of decode stage: {len_fpn - 1} mismatch of heads: {head_list}'
        self.local_occformer = nn.ModuleList([
                                SwinOccFormerStage(
                                    timesteps=timestep, 
                                    in_channels=dim // (2**(i+1)), 
                                    input_resolution=[resolutions*(2**(i+1)), resolutions*(2**(i+1))],
                                    number_of_heads=head_list[i],
                                    window_size=8,
                                    dropout=dropout,
                                    dropout_path=dropout,
                                    offset_downscale_factor=offset_downscale_factor,
                                    flow_warp=local_flow_warp) for i in range(len_fpn-1) ])

        if flow_pred:
            self.output_conv = PredFinalDecoder(dim // (2 ** len_fpn),large_scale=large_scale, planning=planning)
        else:
            self.output_conv = nn.Conv3d(dim // (2 ** len_fpn), 4, (1, kernel, kernel), padding='same')
        
        self.ms_proj_list = None
        print('swin_decoder',sum([p.numel() for p in self.parameters()]))

    def forward(self, output_list, actor, actor_mask, flow):
        flow = None
        last_output = self.deep_query(output_list[-1]) if self.large_res else None
        self.bev_feat_list = []
        self.ms_proj_list = []
        for i in range(self.len_fpn):
            ind = -(i+2) if self.large_res else -(i+1)
            visual = output_list[ind]
            occformer = self.global_occformer if i==0 else self.local_occformer[i-1]
            occ_output = occformer(visual, actor, actor_mask, last_output)

            if self.fg_tsa:
                occ_output, flow = self.fg_tsa_layers[i](occ_output, visual, flow) 
                
            if self.ms_proj:
                self.ms_proj_list.append(self.proj_list[i](occ_output))
            
            self.bev_feat_list.append(occ_output[:, :, -1])
            occ_output = self.resnet_deconvs[i](occ_output)
            last_output = occ_output

        if flow is not None:
            decode_output = self.output_conv(occ_output, flow)
        else:
            decode_output = self.output_conv(occ_output)

        #[b, t, c, h, w]
        return decode_output


class EgoPlanner(nn.Module):
    def __init__(self, dim=256, use_dynamic=False,timestep=5):
        super(EgoPlanner,self).__init__()
        self.timestep = timestep
        self.planner = nn.Sequential(nn.Linear(256, 128), nn.ELU(),nn.Dropout(0.1),
                                    nn.Linear(128, timestep*2*10))
        self.scorer = nn.Sequential(nn.Linear(256, 128), nn.ELU(),nn.Dropout(0.1),
                                    nn.Linear(128, 1))
        self.use_dynamic = use_dynamic
    
    def physical(self, action, last_state):
        d_t = 0.1 
        d_v = action[:, :, :, 0].clamp(-5, 5)
        d_theta = action[:, :, :, 1].clamp(-1, 1)
        
        x_0 = last_state[:, 0]
        y_0 = last_state[:, 1]
        theta_0 = last_state[:, 4]
        v_0 = torch.hypot(last_state[:, 2], last_state[:, 3]) 

        v = v_0.reshape(-1,1,1) + torch.cumsum(d_v * d_t, dim=-1)
        v = torch.clamp(v, min=0)
        theta = theta_0.reshape(-1,1,1) + torch.cumsum(d_theta * d_t, dim=-1)
        theta = torch.fmod(theta, 2*torch.pi)
        x = x_0.reshape(-1,1,1) + torch.cumsum(v * torch.cos(theta) * d_t, dim=-1)
        y = y_0.reshape(-1,1,1) + torch.cumsum(v * torch.sin(theta) * d_t, dim=-1)
        traj = torch.stack([x, y, theta], dim=-1)
        return traj
    
    def forward(self, features, current_state):
        traj = self.planner(features).reshape(-1, 9, self.timestep*10, 2)
        if self.use_dynamic:
            traj = self.physical(traj, current_state)
        score = self.scorer(features)
        return traj, score

class IterativePlanningDecoder(nn.Module):
    def __init__(self, 
        num_layers=1,
        dim=256, 
        heads=8, 
        dropout=0.1, 
        use_dynamic=True,
        use_planning=False,
        timestep=5,
        bev_dim=384):
        super(IterativePlanningDecoder,self).__init__()

        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList(
            [
                SingleLayerPlanningDecoder(dim, heads, dropout, use_dynamic, use_planning, timestep,
                init_decode=(True if i==0 else False), bev_dim=384)
                for i in range(num_layers)
            ]
        )

        self.intention_embed = nn.Parameter(torch.zeros(1, 9, 256), requires_grad=True)
        nn.init.kaiming_uniform_(self.intention_embed)
    
    def forward(self, inputs):
        output_list = []
        b = inputs['encodings'].shape[0]
        intention_query = self.intention_embed.expand(b, -1, -1)

        inputs['intention_query'] = intention_query
        inputs['dynamic_query'] = intention_query
        for i in range(self.num_layers):
            inputs, traj, score = self.decoder_layers[i](inputs)
            inputs['dynamic_query'] = gen_sineembed_for_position2d(traj[..., -1, :2])
            output_list.append([traj, score])
        return output_list

class SingleLayerPlanningDecoder(nn.Module):
    def __init__(self, dim=256, heads=8, dropout=0.1, use_dynamic=True,use_planning=False,timestep=5,init_decode=False,
        bev_dim=384):
        super(SingleLayerPlanningDecoder,self).__init__()
        assert use_planning==False,'multi-stage planner decoder do not yet allow planning as queries'
        self.init_decode = init_decode
        if init_decode:
            self.region_embed = nn.Parameter(torch.zeros(1, 9, 256), requires_grad=True)
            nn.init.kaiming_uniform_(self.region_embed)
        
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.bev_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.bev_layer = nn.Sequential(nn.Linear(bev_dim, dim), nn.GELU())

        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim*4, dim), nn.Dropout(0.1))
        self.fusion_mlp = nn.Sequential(nn.Linear(3*dim, dim), nn.GELU(), nn.Linear(dim, dim))

        self.norm_0 = nn.LayerNorm(dim)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.planner = EgoPlanner(dim, use_dynamic, timestep)
    
    def forward(self, inputs):
        if self.init_decode:
            b = inputs['encodings'].shape[0]
            plan_query = self.region_embed.expand(b,-1,-1)
        else:
            plan_query = inputs['plan_query']

        int_query = inputs['intention_query']
        q = plan_query + int_query

        self_plan_query,_ = self.self_attention(q, q, plan_query)
        self_plan_query = self.norm_0(self_plan_query + plan_query)
        _, q, _ = self_plan_query.shape

        dyna_query = inputs['dynamic_query']

        map_actors = inputs['encodings'][:, 0]
        map_actors_mask = inputs['masks'][:, 0]
        map_actors_mask[:,0] = False

        ego_feature = map_actors[:, 0]
        ego_feature = ego_feature.unsqueeze(1).expand(-1, q, -1)
        
        dense_feature,_ = self.cross_attention(self_plan_query+dyna_query, map_actors, map_actors, key_padding_mask=map_actors_mask)
        dense_feature = self.norm_1(dense_feature + self.ffn(dense_feature))

        b, c, h, w = inputs['bev_feature'].shape
        bev_feature = inputs['bev_feature'].reshape(b, c, h*w).permute(0, 2, 1)
        bev_feature = self.bev_layer(bev_feature)
        bev_feature,_ = self.bev_attention(self_plan_query+dyna_query, bev_feature, bev_feature)
        bev_feature = self.norm_2(bev_feature + self.ffn(bev_feature))

        plan_query = torch.cat([ego_feature, dense_feature, bev_feature], dim=-1)
        plan_query = self.fusion_mlp(plan_query)

        #output
        ego_current = inputs['actors'][:, 0, -1, :]
        inputs['plan_query'] = plan_query
        traj, score = self.planner(plan_query, ego_current)
        return inputs, traj, score.squeeze(-1)


class PlanningDecoder(nn.Module):
    def __init__(self, dim=256, heads=8, dropout=0.1, use_dynamic=False,use_planning=True,timestep=5):
        super(PlanningDecoder,self).__init__()

        if not use_planning:
            self.region_embed = nn.Parameter(torch.zeros(1, 9, 256), requires_grad=True)
            nn.init.kaiming_uniform_(self.region_embed)
        else:
            self.init_planning_encoder = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, dim))
            self.init_planning_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
            self.init_position = PositionalEncoding(dim, 0.1, 50)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.bev_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.bev_layer = nn.Sequential(nn.Linear(384, dim), nn.GELU())

        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim*4, dim), nn.Dropout(0.1))

        self.norm_0 = nn.LayerNorm(dim)
        self.norm_1 = nn.LayerNorm(dim)

        self.use_planning = use_planning
        self.planner = EgoPlanner(dim, use_dynamic, timestep)
    
    def forward(self, inputs):
        #encode the poly plan as ref:
        if self.use_planning:
            plan_line = inputs['plan_lines']
            b, e, l, t, c = plan_line.shape
            plan_line = plan_line.reshape(b, e*l, t, c)
            # plan_mask = torch.eq(plan_line[:, :, :, :2].sum(-1), 0).reshape(b*e*l, t)
            plan_query = self.init_planning_encoder(plan_line.reshape(b*e*l, t, -1))
            plan_query = self.init_position(plan_query)
            #attention across time
            plan_query,_ = self.init_planning_attn(plan_query, plan_query, plan_query)
            plan_query = torch.max(plan_query, dim=-2).values
            #attention across modalities
            plan_query = plan_query.reshape(b, e*l, -1)
            plan_mask = torch.eq(plan_line[:, :, :, 0].sum(-1), 0)
            plan_mask[:,0]=False
            self_plan_query,_ = self.self_attention(plan_query, plan_query, plan_query, key_padding_mask=plan_mask)
        else:
            b = inputs['encodings'].shape[0]
            plan_query = self.region_embed.expand(b,-1,-1)
            self_plan_query,_ = self.self_attention(plan_query, plan_query, plan_query)
        #cross attention with bev and map-actors:
        map_actors = inputs['encodings'][:, 0]
        map_actors_mask = inputs['masks'][:, 0]
        map_actors_mask[:,0] = False
        
        dense_feature,_ = self.cross_attention(self_plan_query, map_actors, map_actors, key_padding_mask=map_actors_mask)
        # print(torch.isnan(dense_feature).any(),'ca', torch.isnan(map_actors).any())
        b, c, h, w = inputs['bev_feature'].shape
        bev_feature = inputs['bev_feature'].reshape(b, c, h*w).permute(0, 2, 1)
        bev_feature = self.bev_layer(bev_feature)
        bev_feature,_ = self.bev_attention(self_plan_query, bev_feature, bev_feature)

        attention_feature = self.norm_0(dense_feature + bev_feature + plan_query)
        output_feature = self.ffn(attention_feature) + attention_feature
        output_feature = self.norm_1(output_feature)

        # output:
        ego_current = inputs['actors'][:, 0, -1, :]
        traj, score = self.planner(output_feature, ego_current)
        return traj, score.squeeze(-1)















