
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmcv.runner.base_module import BaseModule
from projects.mmdet3d_plugin.models.utils.functional import (
    norm_points,
    pos2posemb2d,
    trajectory_coordinate_transform
)
import numpy as np


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CustomMotionTransformerDecoder(BaseModule):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, pc_range=None, embed_dims=256, transformerlayers=None, num_layers=3, **kwargs):
        super(CustomMotionTransformerDecoder, self).__init__()
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.intention_interaction_layers = IntentionInteraction()
        self.track_agent_interaction_layers = nn.ModuleList(
            [TrackAgentInteraction() for i in range(self.num_layers)])
        
        self.pred_agent_interaction_layers = nn.ModuleList(
            [TrackAgentInteraction() for i in range(self.num_layers - 1)])

        self.map_interaction_layers = nn.ModuleList(
            [MapInteraction() for i in range(self.num_layers)])

        self.bev_interaction_layers = nn.ModuleList(
            [build_transformer_layer(transformerlayers) for i in range(self.num_layers)])
        
        # self.mode_interaction_layers = nn.ModuleList(
        #     [MultiAgentInteraction() for i in range(self.num_layers)]
        # )

        self.static_dynamic_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.dynamic_embed_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*3, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.in_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.out_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*4, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

        self.prediction_fuser = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

    def forward(self,
                track_query,
                lane_query,
                track_query_pos=None,
                lane_query_pos=None,
                track_bbox_results=None,
                bev_embed=None,
                reference_trajs=None,
                traj_reg_branches=None,
                traj_cls_branches=None,
                agent_level_embedding=None,
                scene_level_ego_embedding=None,
                scene_level_offset_embedding=None,
                learnable_embed=None,
                agent_level_embedding_layer=None,
                scene_level_ego_embedding_layer=None,
                scene_level_offset_embedding_layer=None,
                rel_mode_embedding=None,
                rel_mode_embedding_layer=None,
                agent_rel_pos=None,
                **kwargs):

        """Forward function for `MotionTransformerDecoder`.
        Args:
            agent_query (B, A, D)
            map_query (B, M, D) 
            map_query_pos (B, G, D)
            static_intention_embed (B, A, P, D)
            offset_query_embed (B, A, P, D)
            global_intention_embed (B, A, P, D)
            learnable_intention_embed (B, A, P, D)
            det_query_pos (B, A, D)
        Returns:
            None
        """
        intermediate = []
        intermediate_reference_trajs = []

        B, _, P, D = agent_level_embedding.shape
        track_query_bc = track_query.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)
        track_query_pos_bc = track_query_pos.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)

        # static intention embedding, which is imutable throughout all layers
        agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)
        static_intention_embed = agent_level_embedding + scene_level_offset_embedding + learnable_embed
        reference_trajs_input = reference_trajs.unsqueeze(4).detach()

        query_embed = torch.zeros_like(static_intention_embed)

        # scene_rel_mode_embedding = rel_mode_embedding
       
        for lid in range(self.num_layers):
            # fuse static and dynamic intention embedding
            # the dynamic intention embedding is the output of the previous layer, which is initialized with anchor embedding
            dynamic_query_embed = self.dynamic_embed_fuser(torch.cat(
                [agent_level_embedding, scene_level_offset_embedding, scene_level_ego_embedding], dim=-1))
            
            # fuse static and dynamic intention embedding
            query_embed_intention = self.static_dynamic_fuser(torch.cat(
                [static_intention_embed, dynamic_query_embed], dim=-1))  # (B, A, P, D)
            
            # fuse intention embedding with query embedding
            query_embed = self.in_query_fuser(torch.cat([query_embed, query_embed_intention], dim=-1))

            # query_embed = self.mode_interaction_layers[lid](query_embed)
            
            # interaction between agents
            track_query_embed = self.track_agent_interaction_layers[lid](
                query_embed, track_query, query_pos=track_query_pos_bc, key_pos=track_query_pos)
            
            # interaction between agents and map
            map_query_embed = self.map_interaction_layers[lid](
                query_embed, lane_query, query_pos=track_query_pos_bc, key_pos=lane_query_pos)
            
            # interaction between agents and bev, ie. interaction between agents and goals
            # implemented with deformable transformer
            # print(query_embed.shape, bev_embed.shape, reference_trajs_input.shape)
            # assert 1==0
            bev_query_embed = self.bev_interaction_layers[lid](
                query_embed,
                value=bev_embed,
                query_pos=track_query_pos_bc,
                bbox_results=track_bbox_results,
                reference_trajs=reference_trajs_input,
                **kwargs)
            
            if lid > 0:
                pred_query_embed = self.pred_agent_interaction_layers[lid-1](
                    query_embed, pred_embed, query_pos=track_query_pos_bc
                )
                track_query_embed += pred_query_embed
            
            # fusing the embeddings from different interaction layers
            query_embed = [track_query_embed, map_query_embed, bev_query_embed, track_query_bc+track_query_pos_bc]
            query_embed = torch.cat(query_embed, dim=-1)
            query_embed = self.out_query_fuser(query_embed)

            if traj_reg_branches is not None:
                # update reference trajectory
                tmp = traj_reg_branches[lid](query_embed)
                bs, n_agent, n_modes, n_steps, _ = reference_trajs.shape
                tmp = tmp.view(bs, n_agent, n_modes, n_steps, -1)
                
                # we predict speed of trajectory and use cumsum trick to get the trajectory
         
                tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
            
                new_reference_trajs = torch.zeros_like(reference_trajs)
                new_reference_trajs = tmp[..., :2]
                reference_trajs = new_reference_trajs.detach()
                reference_trajs_input = reference_trajs.unsqueeze(4)  # BS NUM_AGENT NUM_MODE 12 NUM_LEVEL  2

                # update embedding, which is used in the next layer
                # only update the embedding of the last step, i.e. the goal
                ep_offset_embed = reference_trajs.detach()
                ep_ego_embed = trajectory_coordinate_transform(reference_trajs.unsqueeze(
                    2), track_bbox_results, with_translation_transform=True, with_rotation_transform=False).squeeze(2).detach()
                ep_agent_embed = trajectory_coordinate_transform(reference_trajs.unsqueeze(
                    2), track_bbox_results, with_translation_transform=False, with_rotation_transform=True).squeeze(2).detach()
                
                ep_offset_embed = norm_points(ep_offset_embed[..., -1, :], self.pc_range)

                agent_level_embedding = agent_level_embedding_layer(pos2posemb2d(
                    norm_points(ep_agent_embed[..., -1, :], self.pc_range))) # ego zero normed and rotated
                scene_level_ego_embedding = scene_level_ego_embedding_layer(pos2posemb2d(
                    norm_points(ep_ego_embed[..., -1, :], self.pc_range))) # global framed
                # scene_level_offset_embedding = scene_level_offset_embedding_layer(pos2posemb2d(
                #     ep_offset_embed)) # ego zero normed, but not rotated according to self angle
                
                # static + dynamic rel pos
                b, a, m, d = ep_offset_embed.shape
                # ep_offset_embed = ep_offset_embed.unsqueeze(2).unsqueeze(2).expand(-1, -1, m, a, -1, -1) #[b, a, m, a, m, 2]
                # dyna_rel_mode_embed = ep_offset_embed #+ agent_rel_pos[:, :, None, :, None, :]
                # dyna_rel_mode_embed = dyna_rel_mode_embed.reshape(b, a*m, a*m, d)
                # dyna_rel_mode_embedding = rel_mode_embedding_layer(pos2posemb2d(dyna_rel_mode_embed))
                # scene_rel_mode_embedding = rel_mode_embedding + dyna_rel_mode_embedding

                intermediate.append(query_embed)
                intermediate_reference_trajs.append(reference_trajs)

                if traj_cls_branches is not None:
                    # future_encodings:
                    reference_cls_score = traj_cls_branches[lid](query_embed)[..., 0]
                    bs, n_agent, n_modes = reference_cls_score.shape
                    reference_cls_score = reference_cls_score.softmax(-1).detach().unsqueeze(-1)
                    pred_embed = self.prediction_fuser(pos2posemb2d(norm_points(ep_ego_embed, self.pc_range)))
                    pred_embed = torch.max(pred_embed, dim=-2)[0]
                    pred_embed = (pred_embed * reference_cls_score).sum(-2)


        return torch.stack(intermediate), torch.stack(intermediate_reference_trajs)


class TrackAgentInteraction(BaseModule):
    """
    Modeling the interaction between the agents
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerDecoderLayer(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
                                                                  batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None):
        '''
        query: context query (B, A, P, D) 
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        mem = key.expand(B*A, -1, -1)
        # N, A, P, D -> N*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query


class MapInteraction(BaseModule):
    """
    Modeling the interaction between the agent and the map
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerDecoderLayer(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
                                                                  batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None):
        '''
        x: context query (B, A, P, D) 
        query_pos: mode pos embedding (B, A, P, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # N, A, P, D -> N*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        mem = key.expand(B*A, -1, -1)
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query

class MultiAgentInteraction(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerEncoderLayer(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
                                                                  batch_first=batch_first)

    def forward(self, query):
        B, A, P, D = query.shape
        # B, A, P, D -> B, A*P, D
        rebatch_x = torch.flatten(query, start_dim=1, end_dim=2)
        rebatch_x = self.interaction_transformer(rebatch_x)
        out = rebatch_x.view(B, A, P, D)
        return out

class IntentionInteraction(BaseModule):
    """
    Modeling the interaction between anchors
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerEncoderLayer(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
                                                                  batch_first=batch_first)

    def forward(self, query):
        B, A, P, D = query.shape
        # B, A, P, D -> B*A,P, D
        rebatch_x = torch.flatten(query, start_dim=0, end_dim=1)
        rebatch_x = self.interaction_transformer(rebatch_x)
        out = rebatch_x.view(B, A, P, D)
        return out
    
from torch.nn.init import constant_, xavier_uniform_

class CustomTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, init_cfg=None, dropout=0.1, layer_norm_eps=1e-5, norm_first=False):
        super(CustomTransformerLayer, self).__init__()


        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = CustomMHA(d_model, nhead, init_cfg, dropout)
    
    def forward(self, query, rel_pos_query=None, rel_pos_value=None, src_mask=None):

        '''
        rel_pos_query: [b, q, k, d]
        '''

        x = query
        if self.norm_first:
            x = x + self.self_attn(self.norm1(x), rel_pos_query, rel_pos_value, src_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self.self_attn(x, rel_pos_query, rel_pos_value, src_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)

class CustomMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, init_cfg=None, dropout=0.):
        super(CustomMHA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3*embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj.weight)
        xavier_uniform_(self.out_proj.weight)
        constant_(self.in_proj.bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(
        self,
        query,
        rel_pos_q=None,
        rel_pos_v=None,
        attn_mask=None):

        '''
        rel_pos_q: [b, q, k, d]
        rel_pos_v: [b, q, v, d]
        '''

        query = self.in_proj(query)
        b, t, d = query.shape
        query = query.reshape(b, t, self.num_heads, self.head_dim*3)

        res = torch.split(query, self.head_dim, dim=-1)
        q, k, v = res

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        
        dot_score = torch.matmul(q, k)
        if rel_pos_q is not None:
            rel_pos_q = rel_pos_q.reshape(b, t, t, self.num_heads, self.head_dim)
            rel_pos_q = rel_pos_q.permute(0, 3, 1, 4, 2) #[b, h, q, d, k]
            #[b, h, q, 1, d] * [b, h, q, d, k] -> [b, h, q, 1, k] 
            dot_score_2 = torch.matmul(q.unsqueeze(-2), rel_pos_q).squeeze(-2)
            dot_score += dot_score_2

        dot_score = dot_score / np.sqrt(self.head_dim)

        if attn_mask is not None:
            dot_score = dot_score - attn_mask.float() * 10e8
        dot_score = F.softmax(dot_score, dim=-1)
        dot_score = self.dropout(dot_score)

        value = torch.matmul(dot_score, v)
        if rel_pos_v is not None:
            rel_pos_v = rel_pos_v.reshape(b, t, t, self.num_heads, self.head_dim)
            rel_pos_v = rel_pos_v.permute(0, 3, 1, 2, 4) #[b, h, q, k, d]
            # dot_score: [b, h, q, 1, k] * [b, h, q, k, d]  -> [b, h, q, d]
            value_2 = torch.matmul(dot_score.unsqueeze(-2), rel_pos_v).squeeze(-2)
            value += value_2
        value = value.permute(0, 2, 1, 3) #[b, t, h, d//h]
        value = value.reshape(b, t, self.embed_dim)
        value = self.out_proj(value)

        return value



        





