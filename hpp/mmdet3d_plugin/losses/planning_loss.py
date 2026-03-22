#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pickle
from mmdet.models import LOSSES


@LOSSES.register_module()
class PlanningLoss(nn.Module):
    def __init__(self, loss_type='L2'):
        super(PlanningLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, sdc_traj, gt_sdc_fut_traj, mask, all_future_traj=None, all_mask=None):
        plan_horizon = gt_sdc_fut_traj.shape[-2]
        err = sdc_traj[..., :plan_horizon, :2] - gt_sdc_fut_traj[..., :2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=-1)
        err = torch.pow(err, exponent=0.5)
        mean_l2 = torch.sum(err * mask)/(torch.sum(mask) + 1e-5)

        return mean_l2

        # if all_future_traj is not None:
        #     traj_dim = sdc_traj.shape[-1]
        #     fde = [1, 3, 5, 11]
        #     all_future_traj = all_future_traj[:, 0]
        #     all_mask = all_mask[:, 0]
        #     err = F.smooth_l1_loss(sdc_traj, all_future_traj, reduction='none').sum(-1) * all_mask
        #     ade_err = torch.sum(err)/(torch.sum(all_mask) + 1e-5)
        #     ade_2_err = torch.sum(err[..., :plan_horizon])/(torch.sum(all_mask[..., :plan_horizon]) + 1e-5)
        #     fde_err = torch.sum(err[..., fde]) / (torch.sum(all_mask[..., fde]) + 1e-5)
        #     aux_err = ade_err + 0.5 * fde_err
        #     return ade_2_err + 0.5 * aux_err
        # else:
        #     return mean_l2

@LOSSES.register_module()
class OccCollisionLoss(nn.Module):
    def __init__(self, weight=1.0, delta=0.5, m_per_grid=0.25,
        bev_res = (200, 200)):
        super(OccCollisionLoss, self).__init__()   
        self.weight = weight 
        self.w = 1.85 + delta
        self.h = 4.084 + delta
        grid_x = torch.arange(-self.w/2, self.w/2, m_per_grid)
        grid_y = torch.arange(-self.h/2+0.5, self.h/2+0.5, m_per_grid)
        tmp_m, tmp_n = torch.meshgrid(grid_x, grid_y)
        self.plan_grid = torch.stack([tmp_m, tmp_n], dim=-1)

        self.bev_w, self.bev_h = bev_res

        # ind_x, ind_y = torch.range(bev_res[0]), torch.range(bev_res[1])
        # tmp_x, tmp_y = torch.meshgrid(ind_x, ind_y)
        # self.bev_grid = (torch.stack([tmp_x, tmp_y], dim=-1) - 100) * 0.5 + 0.25

    
    def forward(self, 
                sdc_traj_all, 
                sdc_planning_gt, 
                sdc_planning_gt_mask, 
                bev_mask,
                bev_target,
                frame_mask=None):


        bev_mask = bev_mask.sigmoid().max(1)[0]
        bev_mask = (bev_mask > 0.1).float() #[1, t, h, w]
        mask = torch.sum(bev_mask)
        if mask==0:
            return mask
        # n_gt, s, h, w = bev_mask.size()

        # keep_mask = (bev_target.long() != 255)
        # bev_target = bev_target.float()
        # # print(bev_target.shape, keep_mask.shape, bev_mask.shape)
        # bev_target = bev_target * keep_mask.float()
        # bev_mask   = bev_mask   * keep_mask.float()

        # # Ignore invalid frame
        # if frame_mask is not None:
        #     assert frame_mask.size(0) == s, f"{frame_mask.size()}"
        #     if frame_mask.sum().item() == 0:
        #         return bev_mask.sum() * 0.
        #     frame_mask = frame_mask.view(1, s, 1, 1)
        #     bev_target = bev_target * frame_mask.float()
        #     bev_mask   = bev_mask   * frame_mask.float()

        # bev_mask : [c, t, h ,w]        
        n_futures = bev_target.shape[1]
        consistency_loss = []
        for i in range(n_futures):
        
            if sdc_planning_gt_mask[0, i] == 0:
                #invalid plan
                continue
            plan_xy = sdc_traj_all[0, i, :]
            sdc_yaw = sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype)

            cur_t = min(i+1, n_futures-1)
            bev_preds = bev_mask[:, cur_t]#.detach()
            
            # plan_grids = self.to_plan_grids(plan_xy)
            # plan_grids = plan_grids / 0.5

            # plan_grids = (plan_grids - 100) / 100

            # plan_grids[..., [0, 1]] = plan_grids[..., [1, 0]]

            # col_bev = F.grid_sample(bev_preds, plan_grids.unsqueeze(0),
            #  mode='bilinear', padding_mode='zeros')[0].contiguous() #[c, h, w]


            ##########
            pos_xy = torch.nonzero(bev_preds[0], as_tuple=False)
            pos_xy = pos_xy[:, [1, 0]]
            pos_xy[:, 0] = (pos_xy[:, 0] - 100) * 0.5 + 0.25
            pos_xy[:, 1] = (pos_xy[:, 1] - 100) * 0.5 + 0.25

            keep_index = torch.sum((plan_xy[None, :2] - pos_xy[:, :2])**2, axis=-1) < 5**2
            col_xy = pos_xy[keep_index]
            if torch.sum(keep_index>0)==0:
                continue

            col_bev = 0.5*torch.sum(torch.exp(-0.5*((plan_xy[0] - col_xy[:, 0])**2 + (plan_xy[1] - col_xy[:, 1])**2)))/2.507
            consistency_loss.append(col_bev)

        if len(consistency_loss) ==0 :
            return bev_mask.sum() * 0.
        consistency_loss = torch.stack(consistency_loss, dim=0)
        # print(torch.mean(consistency_loss))
        # if frame_mask is not None:
        #     ratio = n_futures / frame_mask.float()[:, 1:].sum()
        #     consistency_loss *= ratio
        consistency_loss = torch.mean(consistency_loss) * self.weight
        return consistency_loss
    
    def to_plan_grids(self, plan_xy):
        # [mesh, 2]
        mesh_grids = self.plan_grid.to(plan_xy.device)
        mesh_grids = mesh_grids + 49.75
        # sin, cos = torch.sin(theta), torch.cos(theta)
        x, y = mesh_grids[:, :, 0] , mesh_grids[:, :, 1] 
        # new_x = cos * x + sin * y + plan_xy[None, None, 0]
        # new_y = -sin * x + cos * y + plan_xy[None, None, 1]
        new_x = x + plan_xy[None, None, 0]
        new_y = y + plan_xy[None, None, 1]
        new_corners = torch.stack([new_x, new_y], dim=-1).contiguous()
        return new_corners


@LOSSES.register_module()
class CollisionLoss(nn.Module):
    def __init__(self, delta=0.5, weight=1.0):
        super(CollisionLoss, self).__init__()
        self.w = 1.85 + delta
        self.h = 4.084 + delta
        self.weight = weight
    
    def forward(self, sdc_traj_all, sdc_planning_gt, sdc_planning_gt_mask, future_gt_bbox):
        # sdc_traj_all (1, 6, 2)
        # sdc_planning_gt (1,6,3)
        # sdc_planning_gt_mask (1, 6)
        # future_gt_bbox 6x[lidarboxinstance]
        n_futures = len(future_gt_bbox)
        inter_sum = sdc_traj_all.new_zeros(1, )
        dump_sdc = []
        for i in range(n_futures):
            if sdc_planning_gt_mask[0, i] == 0:
                #invalid plan
                continue
            if len(future_gt_bbox[i].tensor) > 0:
                future_gt_bbox_corners = future_gt_bbox[i].corners[:, [0,3,4,7], :2] # (N, 8, 3) -> (N, 4, 2) only bev 
                # sdc_yaw = -sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype) - 1.5708
                sdc_yaw = sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype)
                sdc_bev_box = self.to_corners([sdc_traj_all[0, i, 0], sdc_traj_all[0, i, 1], self.w, self.h, sdc_yaw])
                dump_sdc.append(sdc_bev_box.cpu().detach().numpy())
                for j in range(future_gt_bbox_corners.shape[0]):
                    inter_sum += self.inter_bbox(sdc_bev_box, future_gt_bbox_corners[j].to(sdc_traj_all.device))
        return inter_sum * self.weight
        
    def inter_bbox(self, corners_a, corners_b):
        xa1, ya1 = torch.max(corners_a[:, 0]), torch.max(corners_a[:, 1])
        xa2, ya2 = torch.min(corners_a[:, 0]), torch.min(corners_a[:, 1])
        xb1, yb1 = torch.max(corners_b[:, 0]), torch.max(corners_b[:, 1])
        xb2, yb2 = torch.min(corners_b[:, 0]), torch.min(corners_b[:, 1])
        
        xi1, yi1 = min(xa1, xb1), min(ya1, yb1)
        xi2, yi2 = max(xa2, xb2), max(ya2, yb2)
        intersect = max((xi1 - xi2), xi1.new_zeros(1, ).to(xi1.device)) * max((yi1 - yi2), xi1.new_zeros(1,).to(xi1.device))
        return intersect

    def to_corners(self, bbox):
        x, y, w, l, theta = bbox
        corners = torch.tensor([
            [w/2, -l/2], [w/2, l/2], [-w/2, l/2], [-w/2,-l/2]  
        ]).to(x.device) # 4,2
        rot_mat = torch.tensor(
            [[torch.cos(theta), torch.sin(theta)],
             [-torch.sin(theta), torch.cos(theta)]]
        ).to(x.device)
        new_corners = rot_mat @ corners.T + torch.tensor(bbox[:2])[:, None].to(x.device)
        return new_corners.T


@LOSSES.register_module()
class PlanPredCollisionLoss(nn.Module):
    """Planning constraint to push ego vehicle away from other agents.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        agent_thresh (float, optional): confidence threshold to filter agent predictions.
        x_dis_thresh (float, optional): distance threshold between ego and other agents in x-axis.
        y_dis_thresh (float, optional): distance threshold between ego and other agents in y-axis.
        point_cloud_range (list, optional): point cloud range.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        agent_thresh=0.5,
        x_dis_thresh=1.5,
        y_dis_thresh=3.0,
        dis_thresh=3.0
    ):
        super(PlanPredCollisionLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.agent_thresh = agent_thresh
        self.x_dis_thresh = x_dis_thresh
        self.y_dis_thresh = y_dis_thresh
        self.dis_thresh = dis_thresh

    def forward(self,
                ego_plan,
                agent_fut_preds,
                agent_score_preds,
                ego_plan_mask):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_fut_cls_preds (Tensor): [B, num_agent, fut_mode]
            agent_score_preds (Tensor): [B, num_agent, 10]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """

        # only use best mode pred
        if agent_score_preds.shape[1]==0:
            return ego_plan.new_zeros(1,)

        b, a, m = agent_score_preds.shape 
        best_mode_idxs = torch.argmax(agent_score_preds, dim=-1)
        # batch_idxs = [[i] for i in range(agent_score_preds.shape[0])]
        # agent_num_idxs = [[i for i in range(agent_score_preds.shape[1])] for j in range(agent_score_preds.shape[0])]
        score_mask = torch.max(agent_score_preds, dim=-1)[0] < 0.5 
        score_mask = score_mask.unsqueeze(-1)
        agent_fut_preds = agent_fut_preds[torch.arange(b)[:, None, None].long(), torch.arange(a)[None, :, None].long(), best_mode_idxs.unsqueeze(-1)][:, :, 0]
        #[b, a, t]
        dist = torch.linalg.norm(agent_fut_preds - ego_plan[:, None, :, :2], dim=-1)
        less_dist = (dist <= self.dis_thresh).float() > 0
        # print(less_dist.shape)
        less_dist = less_dist.float() #temproal mask [b, t]
        dist_mask = torch.logical_or(dist > self.dis_thresh, score_mask).float() * 1e2

        x_dist = torch.abs(ego_plan[:, None, :, 0] - agent_fut_preds[..., 0]) + dist_mask
        y_dist = torch.abs(ego_plan[:, None, :, 1] - agent_fut_preds[..., 1]) + dist_mask

        x_min_idxs = torch.argmin(x_dist, dim=1).tolist()
        y_min_idxs = torch.argmin(y_dist, dim=1).tolist()

        batch_idxs = [[i] for i in range(y_dist.shape[0])]
        ts_idxs = [[i for i in range(y_dist.shape[-1])] for j in range(y_dist.shape[0])]

        # [B, t]
        x_min_dist = x_dist[batch_idxs, x_min_idxs, ts_idxs]
        y_min_dist = y_dist[batch_idxs, y_min_idxs, ts_idxs]

        x_loss = (x_min_dist <= self.x_dis_thresh).float() * (self.x_dis_thresh - x_min_dist)  
        y_loss = (y_min_dist <= self.y_dis_thresh).float() * (self.y_dis_thresh - y_min_dist) 

        loss = torch.stack([x_loss, y_loss], dim=-1) * ego_plan_mask[..., None].float()

        loss_bbox = self.loss_weight * torch.mean(loss)
        return loss_bbox
