#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmdet.models.builder import HEADS, build_loss
from einops import rearrange
from projects.mmdet3d_plugin.models.utils.functional import bivariate_gaussian_activation, pos2posemb2d, norm_points
from .planning_head_plugin import CollisionNonlinearOptimizer
import numpy as np
import copy

@HEADS.register_module()
class PlanningHeadMultiMode(nn.Module):
    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 embed_dims=256,
                 planning_steps=6,
                 prediction_steps=6,
                 ego_status=False,
                 score_pred=True,
                 loss_planning=None,
                 loss_collision=None,
                 loss_occol=None,
                 loss_predcol=None,
                 planning_eval=False,
                 use_col_optim=False,
                 col_optim_args=dict(
                    occ_filter_range=5.0,
                    sigma=1.0, 
                    alpha_collision=5.0,
                 ),
                 with_adapter=False,
                 norm=False,
                 pc_range=None,
                ):
        """
        Single Mode Planning Head for Autonomous Driving.

        Args:
            embed_dims (int): Embedding dimensions. Default: 256.
            planning_steps (int): Number of steps for motion planning. Default: 6.
            loss_planning (dict): Configuration for planning loss. Default: None.
            loss_collision (dict): Configuration for collision loss. Default: None.
            planning_eval (bool): Whether to use planning for evaluation. Default: False.
            use_col_optim (bool): Whether to use collision optimization. Default: False.
            col_optim_args (dict): Collision optimization arguments. Default: dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0).
        """
        super(PlanningHeadMultiMode, self).__init__()

        # Nuscenes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.navi_embed = nn.Embedding(3, embed_dims)
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, prediction_steps * 2),
        )
        self.loss_planning = build_loss(loss_planning)
        self.loss_predcol = None
        if loss_predcol is not None:
            self.loss_predcol = build_loss(loss_predcol)
        
        self.planning_steps = planning_steps
        self.prediction_steps = prediction_steps
        self.planning_eval = planning_eval
        self.norm = norm
        self.pc_range = pc_range
        self.score_pred = score_pred
        self.ego_status = ego_status


        #### planning head
        fuser_dim = 3
        layer_num = 3
        self.bev_attn = nn.ModuleList(
            [nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
            for _ in range(layer_num)])
        
        self.lane_attn = nn.ModuleList(
            [nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
            for _ in range(layer_num)])

        self.actor_attn = nn.ModuleList(
            [nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
            for _ in range(layer_num)])
        
        self.actor_adapter = nn.Sequential(
                nn.Linear(embed_dims*2, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims)
            ) 
        
        self.mode_fuser = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True)
            )
        
        self.all_fuser = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(embed_dims*fuser_dim, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            ) for _ in range(layer_num)])
            
        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*fuser_dim, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
        
        if self.ego_status:
            self.ego_encoder = nn.Sequential(
                nn.Linear(5, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
        # self.pos_embed = nn.Embedding(1, embed_dims)
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        
        self.loss_occol = None
        if loss_occol is not None:
            self.loss_occol = []
            for cfg in loss_occol:
                self.loss_occol.append(build_loss(cfg))
            self.loss_occol = nn.ModuleList(self.loss_occol)
 
        self.loss_collision = nn.ModuleList(self.loss_collision)
        
        self.use_col_optim = use_col_optim
        self.occ_filter_range = col_optim_args['occ_filter_range']
        self.sigma = col_optim_args['sigma']
        self.alpha_collision = col_optim_args['alpha_collision']

        # TODO: reimplement it with down-scaled feature_map
        self.with_adapter = with_adapter
        if with_adapter:
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1),
            )
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)
        
        self.occ_mask_adapter = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
           
    def forward_train(self,
                      bev_embed, 
                      outs_motion={}, 
                      sdc_planning=None, 
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      all_planning=None,
                      all_planning_mask=None,
                      bev_mask=None,
                      bev_target=None,
                      frame_mask=None,
                      ego_feat=None,
                      ):
        """
        Perform forward planning training with the given inputs.
        Args:
            bev_embed (torch.Tensor): The input bird's eye view feature map.
            outs_motion (dict): A dictionary containing the motion outputs.
            outs_occflow (dict): A dictionary containing the occupancy flow outputs.
            sdc_planning (torch.Tensor, optional): The self-driving car's planned trajectory.
            sdc_planning_mask (torch.Tensor, optional): The mask for the self-driving car's planning.
            command (torch.Tensor, optional): The driving command issued to the self-driving car.
            gt_future_boxes (torch.Tensor, optional): The ground truth future bounding boxes.
            img_metas (list[dict], optional): A list of metadata information about the input images.

        Returns:
            ret_dict (dict): A dictionary containing the losses and planning outputs.
        """
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']

        lane_query = outs_motion['lane_query']
        track_query = outs_motion['track_query']
        traj_query = outs_motion['traj_query']
        traj_pred = outs_motion['traj_preds']
        traj_score = outs_motion['traj_scores']

        track_query_pos = outs_motion['track_query_pos']
        lane_query_pos = outs_motion['lane_query_pos']
        rel_pos, rel_yaw = outs_motion['track_rel_pos'], outs_motion['track_rel_yaw']

        occ_mask = bev_mask
        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command, lane_query,
        traj_query, track_query, track_query_pos, lane_query_pos, traj_pred, traj_score, rel_pos, rel_yaw, ego_feat)

        losses = dict()
        # for i in range(len(outs_planning['outs_planning'])):
        #     if i!= len(outs_planning['outs_planning']) - 1:
        #         continue
        loss_inputs = [sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes, all_planning, all_planning_mask]
        # if self.loss_occol is not None:
        loss_inputs += [bev_mask, bev_target, frame_mask, traj_pred, traj_score, rel_pos]
        t_losses = self.loss(*loss_inputs)
        losses.update(t_losses)

        # outs_motion = dict(
        #     sdc_traj=outs_planning['sdc_trajs'][-1],
        #     sdc_traj_all=outs_planning['sdc_traj_all'][-1],
        # )
        ret_dict = dict(losses=losses, outs_motion=outs_planning)
        return ret_dict

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={}, command=None, ego_feat=None):
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']
        occ_mask = outs_occflow['seg_out']

        lane_query = outs_motion['lane_query']
        track_query = outs_motion['track_query']
        traj_query = outs_motion['traj_query']
        traj_pred = outs_motion['traj_preds']
        traj_score = outs_motion['traj_scores']

        track_query_pos = outs_motion['track_query_pos']
        lane_query_pos = outs_motion['lane_query_pos']

        rel_pos, rel_yaw = outs_motion['track_rel_pos'], outs_motion['track_rel_yaw']
        
        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command, 
        lane_query, traj_query, track_query, track_query_pos, lane_query_pos, traj_pred, traj_score,
        rel_pos, rel_yaw, ego_feat)

        return outs_planning
    
    def decode_layer(self, plan_query, bev_query, lane_query, actor_query, i, bev_mask=None):

        bev_res = self.bev_attn[i](plan_query, bev_query, tgt_key_padding_mask=bev_mask)
        lane_res = self.lane_attn[i](plan_query, lane_query)
        actor_res = self.actor_attn[i](plan_query, actor_query)

        all_res = torch.cat([bev_res, lane_res, actor_res], dim=-1)
        return self.all_fuser[i](all_res)

    def forward(self, 
                bev_embed, 
                occ_mask, 
                bev_pos, 
                sdc_traj_query, 
                sdc_track_query, 
                command,
                lane_query,
                traj_query,
                track_query,
                track_query_pos,
                lane_query_pos,
                pred_traj,
                pred_scores,
                rel_pos=None,
                rel_yaw=None,
                ego_feat=None):
        """
        Forward pass for PlanningHeadSingleMode.

        Args:
            bev_embed (torch.Tensor): Bird's eye view feature embedding.
            occ_mask (torch.Tensor): Instance mask for occupancy [b, t, 1, h, w].
            bev_pos (torch.Tensor): BEV position.
            sdc_traj_query (torch.Tensor): SDC trajectory query.
            sdc_track_query (torch.Tensor): SDC track query.
            command (int): Driving command.

        Returns:
            dict: A dictionary containing SDC trajectory and all SDC trajectories.
        """
        sdc_track_query = sdc_track_query.detach()
        sdc_traj_query = sdc_traj_query[-1]
        P = sdc_traj_query.shape[1]
        N = self.navi_embed.weight.shape[0]
        sdc_track_query = sdc_track_query[:, None, None, :].expand(-1, N, P, -1)
        sdc_traj_query = sdc_traj_query[:, None].expand(-1, N, -1, -1)
        
        navi_embed = self.navi_embed.weight#[command]
        navi_embed = navi_embed[None, : , None, :].expand(-1, -1, P, -1) #[b, navi_mode, pred_mode, dim]
        # print(navi_embed.shape, sdc_track_query.shape, sdc_traj_query.shape)
        plan_query = torch.cat([sdc_traj_query, sdc_track_query, navi_embed], dim=-1)

        plan_query = self.mlp_fuser(plan_query).max(2)[0]   # expand, then fuse  # [1, 3, 6, 768] -> [1, 3, 256]
        plan_query = rearrange(plan_query, 'b p c -> p b c')

        if self.ego_status:
            ego_feat = ego_feat.float()
            ego_feat = self.ego_encoder(ego_feat)
            ego_feat = ego_feat[None]
            plan_query = plan_query + ego_feat
        
        bev_pos = rearrange(bev_pos, 'b c h w -> (h w) b c')
        bev_feat = bev_embed +  bev_pos

        org_occ_mask = occ_mask.clone()
        occ_mask = occ_mask.float().squeeze(2)
        b, t, h, w = occ_mask.shape
        occ_mask = rearrange(occ_mask, 'b t h w -> (h w) b t')
        occ_mask = occ_mask[..., None].detach()
        pred_pos = bev_pos[..., None, :]
        pred_feat = torch.matmul(occ_mask, pred_pos) #[hw, b, t, 1]*[hw, b, 1, c]
        pred_feat = self.occ_mask_adapter(pred_feat)
        pred_feat = torch.max(pred_feat, dim=-2)[0]

        bev_feat = bev_feat + pred_feat
        
        ##### Plugin adapter #####
        if self.with_adapter:
            bev_feat = rearrange(bev_feat, '(h w) b c -> b c h w', h=self.bev_h, w=self.bev_w)
            bev_feat = bev_feat + self.bev_adapter(bev_feat)  # residual connection
            bev_feat = rearrange(bev_feat, 'b c h w -> (h w) b c')
        
        
        ##########################
      
        # pos_embed = self.pos_embed.weight
        # plan_query = plan_query + pos_embed[None, None]  # [1, 1, 256]

        # actor_trajs:
        pred_scores = pred_scores[-1]
        pred_traj = pred_traj[-1]
        b, a, m = pred_scores.shape 
        # print(pred_scores.shape , track_query.shape)
        # if a != 0:
        #     if self.score_pred:
        #         norm_pred_scores = pred_scores.softmax(-1)[:, :, :, None].detach()
        #         # traj_query = traj_query[-1]
        #         # print(pred_traj.shape, rel_pos.shape)
        #         pred_traj = pred_traj[..., :2]
        #         new_pred_traj = pred_traj + rel_pos[None, :, None, None, :2].to(pred_traj.device)
        #         if self.norm:
        #             new_pred_traj = norm_points(new_pred_traj, self.pc_range)
        #         pred_traj_pos = pos2posemb2d(new_pred_traj).detach()
        #         pred_traj_pos = self.mode_fuser(pred_traj_pos)
        #         pred_traj_pos = torch.max(pred_traj_pos, dim=-2)[0]
        #         # traj_query = traj_query + pred_traj_pos
        #         traj_query = pred_traj_pos
        #         traj_query = traj_query * norm_pred_scores
        #         traj_query = traj_query.mean(-2)
        #         # traj_query = traj_query.sum(-2)
        #     else:
        #         pred_traj = pred_traj[..., :2]
        #         traj_query = torch.max(self.mode_fuser(traj_query[-1] + pos2posemb2d(pred_traj[..., -1, :]).detach()), dim=-2)[0]    
        # else:
        #     traj_query = torch.zeros_like(track_query).to(track_query.device)
        # track_query = track_query + track_query_pos.detach()
        # print(traj_query.shape)
        actor_query = torch.cat([track_query.detach(), track_query_pos.detach()], dim=-1) #[b, q, 2*d]

        actor_query = self.actor_adapter(actor_query)
        actor_query = rearrange(actor_query, 'b q c -> q b c')
        lane_query = lane_query.detach() + lane_query_pos.detach()
        lane_query = rearrange(lane_query, 'b q c -> q b c')
        # print(lane_query.shape, lane_query_pos.shape)
        # actor_query_pos = pos2posemb2d(traj_score)
        
        # plan_query: [1, 1, 256]
        # bev_feat: [40000, 1, 256]
        # plan_query = self.attn_module(plan_query, bev_feat)   # [1, 1, 256]
        # occ_mask = occ_mask.detach().bool()
 
        for i in range(3):
            plan_query = self.decode_layer(plan_query, bev_feat, lane_query, actor_query, i, None)

        plan_query = rearrange(plan_query, 'p b c -> b p c')
        if self.ego_status:
            plan_query = plan_query + ego_feat

        plan_query = plan_query[:, command, :][:, 0, :]
        
        sdc_traj_all = self.reg_branch(plan_query).view((-1, self.prediction_steps, 2))
        sdc_traj_all[...,:2] = torch.cumsum(sdc_traj_all[...,:2], dim=2)

        sdc_traj_all[0] = bivariate_gaussian_activation(sdc_traj_all[0])
        org_sdc_traj = sdc_traj_all.clone()

        if self.use_col_optim and not self.training:
            # post process, only used when testing
            # print(pred_traj.shape, pred_scores.shape)
            if pred_scores.shape[-2]==0:
                pred_scores, pred_traj = None, None
            else:
                pred_scores, pred_traj = pred_scores[0], pred_traj[0]
        
            assert org_occ_mask is not None
            sdc_traj_all = self.collision_optimization(sdc_traj_all[..., :self.planning_steps, :], org_occ_mask,
            pred_traj, pred_scores, rel_pos)
         
        
        return dict(
            sdc_traj_org=org_sdc_traj,
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
        )
    

    def collision_optimization(self, sdc_traj_all, occ_mask,
        pred_trajs=None, pred_scores=None, rel_pos=None):
        """
        Optimize SDC trajectory with occupancy instance mask.

        Args:
            sdc_traj_all (torch.Tensor): SDC trajectory tensor.
            occ_mask (torch.Tensor): Occupancy flow instance mask. 
        Returns:
            torch.Tensor: Optimized SDC trajectory tensor.
        """

        pos_xy_t = []
        valid_occupancy_num = 0
        
        if occ_mask.shape[2] == 1:
            occ_mask = occ_mask.squeeze(2)
        occ_horizon = occ_mask.shape[1]
        assert occ_horizon == 5

        if pred_scores is not None:
            #[A, m], traj: [a, m, t, 2]
            num_pred_agents = pred_scores.shape[0]
            if num_pred_agents > 0:
                best_mode =  torch.argmax(pred_scores, dim=-1)
                best_mode_traj = pred_trajs[..., :2] + rel_pos[:, None, None, :2].to(sdc_traj_all.device)
                # best_mode_traj = best_mode_traj[torch.arange(num_pred_agents).long(), best_mode, :, :]
                best_mode_traj = torch.flatten(best_mode_traj, start_dim=0, end_dim=1)
            pred_pos_xy, pred_mask = [], []
        else:
            pred_pos_xy, pred_mask = None, None

        for t in range(self.planning_steps):
            # if t > occ_horizon-1:
            #     continue
            cur_t = min(t+1, occ_horizon-1)
            pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)
            pos_xy = pos_xy[:, [1, 0]]
            pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25
            pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w//2) * 0.5 + 0.25

            # filter the occupancy in range
            keep_index = torch.sum((sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2])**2, axis=-1) < self.occ_filter_range**2
            pos_xy_t.append(pos_xy[keep_index].cpu().detach().numpy())
            valid_occupancy_num += torch.sum(keep_index>0)
        
        if pred_scores is not None:
            if num_pred_agents == 0:
                return sdc_traj_all
            horizon = sdc_traj_all.shape[-2]
            # print(best_mode_traj.shape)
            best_mode_traj = best_mode_traj[..., :horizon, :2] 
            # print(best_mode_traj)
            # print(sdc_traj_all)
            # assert 1==0
            dist = torch.linalg.norm(best_mode_traj - sdc_traj_all, dim=-1)

            for t in range(self.planning_steps):
                less_dist = dist[:, t] <= 3.0
                if less_dist.float().sum()==0:
                    pred_pos_xy.append([[0, 0]])
                    pred_mask.append([0, 0])
                    continue
                close_points = best_mode_traj[less_dist, t].detach().cpu().numpy()
                # min_points = torch.argmin(torch.linalg.norm(close_points - sdc_traj_all[:, t, :], dim=-1), dim=0)
                # closest_points = close_points[min_points].cpu().detach().numpy()
                # diff = torch.abs(close_points - sdc_traj_all[:, t, :])
                # diff_x, arg_min_x = torch.min(diff[:, 0], dim=0)
                # close_points_x = close_points[arg_min_x, 0].cpu().detach().numpy()
                # diff_y, arg_min_y = torch.min(diff[:, 1], dim=0)
                # close_points_y = close_points[arg_min_y, 1].cpu().detach().numpy()

                x_mask = 1.0#(diff_x < 1.5).float().cpu().detach().numpy()
                y_mask = 1.0#(diff_y < 3.0).float().cpu().detach().numpy()

                pred_pos_xy.append([[close_points[i][0], close_points[i][1]] for i in range(close_points.shape[0])])
                pred_mask.append([x_mask, y_mask])

        if valid_occupancy_num == 0:
            return sdc_traj_all
        
        col_optimizer = CollisionNonlinearOptimizer(self.planning_steps, 0.5, self.sigma, self.alpha_collision, pos_xy_t,
        pred_pos_xy, pred_mask, pred=True)
        col_optimizer.set_reference_trajectory(sdc_traj_all[0].cpu().detach().numpy())
        # try:
        sol = col_optimizer.solve()
        sdc_traj_optim = np.stack([sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)], axis=-1)
        traj = torch.tensor(sdc_traj_optim[None], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype)
        # except:
        #     traj = sdc_traj_all
        return traj
    
    def loss(self, sdc_planning, sdc_planning_mask, outs_planning, future_gt_bbox=None, all_planning=None, all_planning_mask=None,
        bev_mask=None,bev_target=None,frame_mask=None, pred_trajs=None, pred_score=None, rel_pos=None, level=0):
        sdc_traj_all = outs_planning['sdc_traj_all'] # b, p, t, 5
        loss_dict = dict()
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](sdc_traj_all[:, :self.planning_steps, :], sdc_planning[0, :, :self.planning_steps, :3], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1), future_gt_bbox[0][1:self.planning_steps+1])
            # loss_collision_2 = self.loss_collision[i](sdc_traj_all, sdc_planning[0, :, :self.prediction_steps, :3], torch.any(sdc_planning_mask[0, :, :self.prediction_steps], dim=-1), future_gt_bbox[0][1:self.prediction_steps+1])
            loss_dict[f'{level}_loss_collision_{i}'] = loss_collision #+ 0.5*loss_collision_2   
            
        # if self.loss_predcol is not None:
        #     pred_trajs = pred_trajs[-1, ..., :self.planning_steps, :2] + rel_pos[None, :, None, None, :2].to(pred_trajs.device)

        #     ego_mask = torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1)
        #     loss_dict[f'{level}_loss_collision_pred'] = self.loss_predcol(sdc_traj_all[:, :self.planning_steps, :],
        #     pred_trajs, pred_score[-1], ego_mask)
            
        # if self.loss_occol is not None:
        #     for i in range(len(self.loss_occol)):
        #         loss_dict[f'loss_collision_occ_{i}'] = self.loss_occol[i](sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :3],
        #         torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1), bev_mask, bev_target, frame_mask)
        loss_ade = self.loss_planning(sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :2], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1),
        all_planning, torch.any(all_planning_mask, dim=-1))
        loss_dict[f'{level}_loss_ade']=loss_ade
        return loss_dict