import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, List, Union, Any

import numpy as np
import sys
import os
from copy import deepcopy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from .swin_T import WindowMultiHeadAttention, DropPath, FeedForward, fold, unfold, bhwc_to_bchw, bchw_to_bhwc

# attn = WindowMultiHeadAttention(in_features=192, window_size=8, number_of_heads=6)

class SwinOccFormerBlock(nn.Module):
    """
    This class implements the Swin transformer block.
    """

    def __init__(self,
                 in_channels,
                 input_resolution,
                 number_of_heads,
                 window_size=7,
                 shift_size=0,
                 ff_feature_ratio=4,
                 dropout=0.0,
                 dropout_attention=0.0,
                 dropout_path= 0.0):
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param input_resolution: (Tuple[int, int]) Input resolution
        :param number_of_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param shift_size: (int) Shifting size to be used
        :param ff_feature_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param dropout: (float) Dropout in input mapping
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_path: (float) Dropout in main path
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        """
        # Call super constructor
        super(SwinOccFormerBlock, self).__init__()
        self.in_channels = in_channels
        self.input_resolution = input_resolution
        self.number_of_heads = number_of_heads
        # Catch case if resolution is smaller than the window size
        if min(self.input_resolution) <= window_size:
            self.window_size = min(self.input_resolution)
            self.shift_size = 0
            self.make_windows= False
        else:
            self.window_size = window_size
            self.shift_size = shift_size
            self.make_windows= True
        # Init normalization layers
        self.normalization_1 = nn.LayerNorm(normalized_shape=in_channels)
        self.normalization_2 = nn.LayerNorm(normalized_shape=in_channels)
        self.normalization_3 = nn.LayerNorm(normalized_shape=in_channels)
        # Init window attention module
        self.window_attention = WindowMultiHeadAttention(
            in_features=in_channels,
            window_size=self.window_size,
            number_of_heads=number_of_heads,
            dropout_attention=dropout_attention,
            dropout_projection=dropout,
            sequential_self_attention=False)

        self.cross_attention = nn.MultiheadAttention(embed_dim=in_channels, 
                                                    num_heads=number_of_heads,
                                                    dropout=dropout_attention,
                                                    batch_first=True)

        # Init dropout layer
        self.dropout = DropPath(
            drop_prob=dropout_path) if dropout_path > 0. else nn.Identity()
        # Init feed-forward network
        self.feed_forward_network = FeedForward(in_features=in_channels,
                                                hidden_features=int(in_channels * ff_feature_ratio),
                                                dropout=dropout,
                                                out_features=in_channels)

        self.temporal_mlp = nn.Sequential(nn.Linear(256, in_channels), nn.GELU(), nn.Dropout(dropout))
        # Make attention mask
        self.__make_attention_mask()

    def __make_attention_mask(self) -> None:
        """
        Method generates the attention mask used in shift case
        """
        # Make masks for shift case
        if self.shift_size > 0:
            height, width = self.input_resolution  # type: int, int
            mask: torch.Tensor = torch.zeros(height, width, device=self.window_attention.tau.device)
            height_slices: Tuple = (slice(0, -self.window_size),
                                    slice(-self.window_size, -self.shift_size),
                                    slice(-self.shift_size, None))
            width_slices: Tuple = (slice(0, -self.window_size),
                                   slice(-self.window_size, -self.shift_size),
                                   slice(-self.shift_size, None))
            counter: int = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    mask[height_slice, width_slice] = counter
                    counter += 1

            mask_windows: torch.Tensor = unfold(mask[None, None], self.window_size)
            mask_windows: torch.Tensor = mask_windows.reshape(-1, self.window_size * self.window_size)
            attention_mask: Optional[torch.Tensor] = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attention_mask: Optional[torch.Tensor] = attention_mask.masked_fill(attention_mask != 0, float(-100.0))
            attention_mask: Optional[torch.Tensor] = attention_mask.masked_fill(attention_mask == 0, float(0.0))
            # print(attention_mask.shape)
        else:
            attention_mask: Optional[torch.Tensor] = None
        # Save mask
        self.register_buffer("attention_mask", attention_mask)

    def update_resolution(self,
                          new_window_size: int,
                          new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Update input resolution
        self.input_resolution: Tuple[int, int] = new_input_resolution
        # Catch case if resolution is smaller than the window size
        if min(self.input_resolution) <= new_window_size:
            self.window_size: int = min(self.input_resolution)
            self.shift_size: int = 0
            self.make_windows: bool = False
        else:
            self.window_size: int = new_window_size
            self.shift_size: int = self.shift_size
            self.make_windows: bool = True
        # Update attention mask
        self.__make_attention_mask()
        # Update attention module
        self.window_attention.update_resolution(new_window_size=new_window_size)
    
    def forward(self, visual, vector=None, mask=None, visual_mask=None):
        batch_size, channels, height, width = visual.shape 
        vector = self.temporal_mlp(vector)
        if self.shift_size > 0:
            output_shift: torch.Tensor = torch.roll(input=visual, shifts=(-self.shift_size, -self.shift_size),
                                                    dims=(-1, -2))
            # if visual_mask is not None:
            #     #batch, 1, h ,w
            #     visual_mask = visual_mask.unsqueeze(1)
            #     visual_mask = torch.roll(input=visual_mask, shifts=(-self.shift_size, -self.shift_size), dims=(-1, -2))
        else:
            output_shift: torch.Tensor = visual
        # Make patches
        
        output_patches: torch.Tensor = unfold(input=output_shift, window_size=self.window_size) \
            if self.make_windows else output_shift
        
        #[b*win, h//win, w//win]
        # if visual_mask is not None:
        #     visual_mask = unfold(input=visual_mask, window_size=self.window_size) \
        #     if self.make_windows else visual_mask
        #     window_size =  (height//self.window_size) * (width//self.window_size)
        #     visual_mask = visual_mask.view(batch_size, window_size, self.window_size*self.window_size)
        #     ca_visual_mask = visual_mask.contiguous()
        #     visual_mask = visual_mask[..., :, None] * visual_mask[..., None, :] 

            
        # Perform window attention
        # print(output_patches.shape, torch.isnan(output_patches).any())

        #[b*wins, wins, win, win] *[1, wins, win, win]
        # attn_mask = (1 - visual_mask) * self.attention_mask.unsqueeze(0) if visual_mask is not None else self.attention_mask
        output_attention: torch.Tensor = self.window_attention(output_patches, mask=self.attention_mask)
        # Merge patches
        output_merge: torch.Tensor = fold(input=output_attention, window_size=self.window_size, height=height,
                                          width=width) if self.make_windows else output_attention
        # Reverse shift if utilized
        if self.shift_size > 0:
            output_shift: torch.Tensor = torch.roll(input=output_merge, shifts=(self.shift_size, self.shift_size),
                                                    dims=(-1, -2))
        else:
            output_shift: torch.Tensor = output_merge
        # Perform normalization
        output_normalize: torch.Tensor = self.normalization_1(output_shift.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # Skip connection
        output_skip: torch.Tensor = self.dropout(output_normalize) + visual

        # perform cross attention:
        if self.shift_size > 0:
            output_shift: torch.Tensor = torch.roll(input=output_skip, shifts=(-self.shift_size, -self.shift_size),
                                                    dims=(-1, -2))
            
        else:
            output_shift: torch.Tensor = output_skip
        
        if visual_mask is not None:
            visual_mask = visual_mask.reshape(batch_size, height, width, vector.shape[1]).permute(0, 3, 1, 2)
            if self.shift_size > 0:
                visual_mask: torch.Tensor = torch.roll(input=visual_mask, shifts=(-self.shift_size, -self.shift_size),
                                                        dims=(-1, -2))
            visual_mask: torch.Tensor = unfold(input=visual_mask, window_size=self.window_size) \
            if self.make_windows else visual_mask
            visual_mask = visual_mask.reshape(-1, vector.shape[1], self.window_size**2).permute(0, 2, 1)
            visual_mask = visual_mask.unsqueeze(1).expand(-1, self.number_of_heads, -1, -1).flatten(0,1)

        # Make patches
        output_patches: torch.Tensor = unfold(input=output_shift, window_size=self.window_size) \
            if self.make_windows else output_shift

        b, l, d = vector.shape
        vector = vector.unsqueeze(0).expand((self.input_resolution[0]//self.window_size)**2, -1, -1, -1).reshape(-1, l, d)
        if mask is not None:
            mask = mask.unsqueeze(0).expand((self.input_resolution[0]//self.window_size)**2, -1, -1).reshape(-1, l)
        
        
        # if ca_visual_mask is not None:
        #     ca_visual_mask = ca_visual_mask.reshape(-1, self.window_size**2)
        #     if mask is None:
        #         mask = torch.zeros((vector.shape[0], vector.shape[1])).to(vector.device)
        #     visual_ca_mask = 1- (1 - ca_visual_mask.unsqueeze(-1)) * (1 - mask.unsqueeze(1))
        #     visual_ca_mask = visual_ca_mask.bool()
        #     visual_ca_mask = visual_ca_mask.unsqueeze(1).expand(-1, self.number_of_heads, -1, -1)
        #     visual_ca_mask = visual_ca_mask.reshape(-1, self.window_size**2, l)

        B,D,_,_ = output_patches.shape
        output_patches = output_patches.reshape(B,D,self.window_size*self.window_size).permute(0, 2, 1)
        # (b, window**2, D)

        visual_vector, _ = self.cross_attention(output_patches, vector, vector, attn_mask=visual_mask, key_padding_mask=mask)
        visual_vector = self.normalization_2(visual_vector)
        visual_vector = visual_vector.permute(0, 2, 1).reshape(B, D, self.window_size, self.window_size)
        visual_vector: torch.Tensor = fold(input=visual_vector, window_size=self.window_size, height=height,
                                          width=width) if self.make_windows else visual_vector
        # Reverse shift if utilized
        if self.shift_size > 0:
            output_shift: torch.Tensor = torch.roll(input=visual_vector, shifts=(self.shift_size, self.shift_size),
                                                    dims=(-1, -2))
        else:
            output_shift: torch.Tensor = visual_vector
            
        output_skip = self.dropout(output_shift) + output_skip

        output_feed_forward: torch.Tensor = self.feed_forward_network(
            output_skip.view(batch_size, channels, -1).permute(0, 2, 1)).permute(0, 2, 1)
        output_feed_forward: torch.Tensor = output_feed_forward.view(batch_size, channels, height, width)
        output_normalize: torch.Tensor = bhwc_to_bchw(self.normalization_3(bchw_to_bhwc(output_feed_forward)))

        output: torch.Tensor = output_skip + self.dropout(output_normalize)
        return output

class FlowFormer(nn.Module):
    def __init__(self, device, in_channels, input_resolution, number_of_heads, offset_downscale_factor=2, dropout=0.):
        super(FlowFormer, self).__init__()
        '''
        conduct Spatial Transformer for each X^{t+1} for flow predictions
        '''
        self.input_resolution = input_resolution
        self.device = device
        self.offset_downscale_factor: int = offset_downscale_factor
        self.number_of_heads: int = number_of_heads
        # Make default offsets
        self.__make_default_offsets()
        # Init offset network
        self.context_network: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                      padding=1, groups=number_of_heads, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                      padding=0, bias=True))
        
        self.dropout = DropPath(dropout)

        self.flow_proj = nn.Conv2d(in_channels=in_channels, out_channels=2 * self.number_of_heads, 
                                kernel_size=1, stride=offset_downscale_factor, padding=0, bias=True)


    def __make_default_offsets(self) -> None:
        """
        Method generates the default sampling grid (inspired by kornia)
        """
        # Init x and y coordinates
        x: torch.Tensor = torch.linspace(0, self.input_resolution[1] - 1, self.input_resolution[1],
                                         device=self.device)
        y: torch.Tensor = torch.linspace(0, self.input_resolution[0] - 1, self.input_resolution[0],
                                         device=self.device)
        # Normalize coordinates to a range of [-1, 1]
        x: torch.Tensor = (x / (self.input_resolution[1] - 1) - 0.5) * 2
        y: torch.Tensor = (y / (self.input_resolution[0] - 1) - 0.5) * 2
        # Make grid [2, height, width]
        grid: torch.Tensor = torch.stack(torch.meshgrid([x, y])).transpose(1, 2)
        # Reshape grid to [1, height, width, 2]
        grid: torch.Tensor = grid.unsqueeze(dim=0).permute(0, 2, 3, 1)
        # Register in module
        self.register_buffer("default_grid", grid)
    
    def forward(self, next_visual, visual, flow_path=None):

        batch_size, channels, height, width = next_visual.shape

        concat_flow = next_visual + flow_path if flow_path is not None else next_visual
        concat_flow = self.context_network(concat_flow)
        flow_out = self.dropout(concat_flow)
        flow_out = flow_out + flow_path if flow_path is not None else flow_out

        projected_flow = self.flow_proj(flow_out)
        offsets: torch.Tensor = F.interpolate(input=projected_flow,
                                              size=(height, width), mode="bilinear", align_corners=True)
        # Reshape offsets to [batch size, number of heads, height, width, 2]
        offsets: torch.Tensor = offsets.reshape(batch_size, -1, 2, height, width).permute(0, 1, 3, 4, 2)
        offsets: torch.Tensor = offsets.reshape(-1, height, width, 2).tanh()

        if visual.dtype != self.default_grid.dtype:
            self.default_grid = self.default_grid.type(visual.dtype)
        # Construct offset grid
        offset_grid: torch.Tensor = self.default_grid.repeat_interleave(repeats=offsets.shape[0], dim=0) + offsets
        # Reshape input to [batch size * number of heads, channels / number of heads, height, width]
        visual: torch.Tensor = visual.view(batch_size, self.number_of_heads, channels // self.number_of_heads, height,
                                         width).flatten(start_dim=0, end_dim=1)
        # Apply sampling grid
        input_resampled: torch.Tensor = F.grid_sample(input=visual, grid=offset_grid.clip(min=-1, max=1),
                                                      mode="nearest", align_corners=False)
        # Reshape resampled tensor again to [batch size, channels, height, width]
        input_resampled: torch.Tensor = input_resampled.view(batch_size, channels, height, width)
        return input_resampled, flow_out

class SwinOccFormerStage(nn.Module):
    def __init__(self,
                 timesteps, 
                 in_channels,
                 input_resolution,
                 number_of_heads,
                 window_size=7,
                 shift_size=0,
                 ff_feature_ratio=4,
                 dropout=0.0,
                 dropout_attention=0.0,
                 dropout_path= 0.0,
                 offset_downscale_factor=2,
                 flow_warp=False):
        super(SwinOccFormerStage, self).__init__()
    
        self.swin_occformer_list = nn.ModuleList([ SwinOccFormerBlock(in_channels, input_resolution, number_of_heads,
                                                    window_size, shift_size, ff_feature_ratio,
                                                    dropout, dropout_attention, dropout_path) for _ in range(timesteps)])
        self.device = self.swin_occformer_list[0].window_attention.tau.device
        
        if flow_warp:
            self.flowformer_list = nn.ModuleList([FlowFormer(self.device, in_channels, input_resolution, number_of_heads,
            offset_downscale_factor, dropout_path) for _ in range(timesteps)])
        self.flow_warp = flow_warp
        self.timesteps = timesteps
        print('swin_agg', sum([p.numel() for p in self.parameters()]))
        
    def forward(self, visual, vectors, actor_mask, last_res=None):
        """
        visual: [b, d, h, w], vectors:[b, n, d], actor_mask:[b, n]
        outputs: [b, d, t, h, w]
        """
        b, d, h, w = visual.shape
        actor_mask[:, 0] = False
        output_list = []
        flow_path = None
        visual_input = visual
        for i in range(self.timesteps):
            visual_input = self.swin_occformer_list[i](visual_input, vectors, actor_mask)
            if last_res is not None:
                visual_input = visual_input + last_res[:,:,i]
            output = visual_input
            if self.flow_warp:
                last_visual = visual if i==0 else output_list[-1]
                foutput, flow_path = self.flowformer_list[i](output,last_visual, flow_path)
                output = torch.mul(output,foutput)
            output_list.append(output)
        return torch.stack(output_list, dim=2) 

class CrossAttention(nn.Module):
    def __init__(self, dim=384, heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True,)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim*4, dim), nn.Dropout(0.1))
        self.norm_0 = nn.LayerNorm(dim)
        self.norm_1 = nn.LayerNorm(dim)

    def forward(self, query, key, mask):
        output, _ = self.cross_attention(query, key, key, key_padding_mask=mask)
        attention_output = self.norm_0(output + query)
        n_output = self.ffn(attention_output)
        return self.norm_1(n_output + attention_output)

class OccFormerModule(nn.Module):
    def __init__(self, input_dim, heads=8, dropout=0.1):
        super(OccFormerModule, self).__init__()
        self.self_attention = nn.MultiheadAttention(input_dim, heads, dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(input_dim, heads, dropout, batch_first=True)

        self.norm = nn.LayerNorm(input_dim)

        self.temporal_mlp = nn.Sequential(nn.Linear(256, input_dim), nn.GELU(), nn.Dropout(dropout))
        self.ffn = nn.Sequential(nn.Linear(input_dim, 2*input_dim), nn.GELU(), nn.Dropout(dropout), 
                    nn.Linear(2*input_dim, input_dim), nn.LayerNorm(input_dim),  nn.Dropout(dropout))

    def forward(self, visual, vectors, mask):
        res_visual = visual
        vectors = self.temporal_mlp(vectors)
        self_visual, _ = self.self_attention(visual, visual, visual)
        self_visual = self.norm(self_visual) + res_visual
        cross_visual,_ = self.cross_attention(self_visual, vectors, vectors, key_padding_mask=mask)
        cross_visual = self.ffn(cross_visual)
        return cross_visual + self_visual

class OccFormer(nn.Module):
    def __init__(self, input_dim, resolutions=16, timestep=5, heads=8, dropout=0.1,
        offset_downscale_factor=2, flow_warp=False):
        super(OccFormer, self).__init__() 
        self.timestep = timestep
        self.resolutions = resolutions

        self.occformer_modules = nn.ModuleList([OccFormerModule(input_dim, heads, dropout) for _ in range(timestep)])
        self.device = self.occformer_modules[0].self_attention.in_proj_weight.device
        if flow_warp:
            self.flowformer_list = nn.ModuleList([FlowFormer(self.device, input_dim, [resolutions, resolutions], heads,
            offset_downscale_factor, dropout) for _ in range(timestep)])
        self.flow_warp = flow_warp
    
    def forward(self, visual, vectors, actor_mask, last_res=None):
        """
        visual: [b, d, h, w], vectors:[b, n, d], actor_mask:[b, n]
        outputs: [b, d, t, h, w]
        """
        b, d, h, w = visual.shape
        visual_input = visual.reshape(b, d, h*w).permute(0, 2, 1) #[b, hw, d]
        actor_mask[:, 0] = False
        output_list = []
        flow_path = None
        for i in range(self.timestep):
            visual_input = self.occformer_modules[i](visual_input, vectors, actor_mask)
            if last_res is not None:
                visual_input = visual_input + last_res[:,:,i].reshape(b, d, h*w).permute(0, 2, 1)
            output = visual_input.permute(0, 2, 1).reshape(b, d, h, w)
            if self.flow_warp:
                last_visual = visual if i==0 else output_list[-1]
                foutput, flow_path = self.flowformer_list[i](output, last_visual, flow_path)
                output = torch.mul(output,foutput)
                # output = output + foutput
            output_list.append(output)
        return torch.stack(output_list, dim=2)

def test_SwinOcc():
    blk = SwinOccFormerBlock(64, [50, 50], 8, window_size=10, shift_size=2)
    from time import time
    blk(torch.randn(1, 64, 50, 50), torch.randn(1, 42, 256), visual_mask=torch.randn(1, 50, 50),
    mask=torch.randn(4, 42))
    print(sum([p.numel() for p in blk.parameters()]))

def testFlowFormer():
    device = torch.device('cpu')
    net = FlowFormer(device, 384, [16, 16], 6)
    output, flow = net(torch.randn(4, 384, 16, 16), torch.randn(4, 384, 16, 16))
    print(output.shape, flow.shape)
    print(sum([p.numel() for p in net.parameters()]))

def test_warpAgg():
    # agg_net = OccFormer(384, timestep=8, heads=6, flow_warp=False)
    agg_net = SwinOccFormerStage(5, 192, [32, 32], number_of_heads=6, window_size=8,flow_warp=False)
    b = 4
    visual = torch.randn(b, 192, 32, 32)
    vector = torch.randn(b, 43, 256)
    vector_mask = torch.zeros((b, 43))
    output = agg_net(visual, vector, vector_mask)
    print(output.shape)
    print(sum([p.numel() for p in agg_net.parameters()]))

# test_SwinOcc()
