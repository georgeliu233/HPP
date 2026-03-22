import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from typing import Optional, no_type_check

def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    shapes = len(pos_tensor.shape)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale

    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t

    pos_x = torch.stack((pos_x[...,  0::2].sin(), pos_x[..., 1::2].cos()), dim=shapes).flatten(shapes-1)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=shapes).flatten(shapes-1)

    pos = torch.cat((pos_y, pos_x), dim=-1)
    shape_tuple = pos.shape
    pos = torch.reshape(pos, shape_tuple[:-2]+(shape_tuple[-2]*shape_tuple[-1],))

    return pos

def temporal_upsample(inputs, size=(2, 2), mode='nearest'):
    assert len(inputs.shape) == 5
    b, c, t, h, w = inputs.shape
    inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
    inputs = F.interpolate(input=inputs, size=(2*h, 2*w), mode=mode, align_corners=True)
    inputs = inputs.reshape(b, t, c, 2*h, 2*w).permute(0, 2, 1, 3, 4)
    return inputs

def multi_scale_deformable_attn_pytorch(
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        sample_mode: str = 'bilinear') -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode=sample_mode,
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class FlowGuidedTSA(nn.Module):
    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = False,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 value_proj_ratio: float = 1.0):
        super(FlowGuidedTSA, self).__init__()
        """An attention module used in Deformable-Detr.

        `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
        <https://arxiv.org/pdf/2010.04159.pdf>`_.

        Args:
            embed_dims (int): The embedding dimension of Attention.
                Default: 256.
            num_heads (int): Parallel attention heads. Default: 8.
            num_levels (int): The number of feature map used in
                Attention. Default: 4.
            num_points (int): The number of sampling points for
                each query in each head. Default: 4.
            im2col_step (int): The step used in image_to_column.
                Default: 64.
            dropout (float): A Dropout layer on `inp_identity`.
                Default: 0.1.
            batch_first (bool): Key, Query and Value are shape of
                (batch, n, embed_dim)
                or (n, batch, embed_dim). Default to False.
            norm_cfg (dict): Config dict for normalization layer.
                Default: None.
            init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
                Default: None.
            value_proj_ratio (float): The expansion ratio of value_proj.
                Default: 1.0.
        """
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        
        # self.last_layer_offsets = nn.Sequential(nn.Linear(2, embed_dims), nn.ReLU())

        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
                                           
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        nn.init.zeros_(self.sampling_offsets.weight)
        # nn.init.constant_init(self.last_layer_offsets, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self._is_init = True


    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                last_reference_offsets: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            last_reference_offsets (torch.Tensor): multi-head reference offsets 
                for last layer.
                shape [bs, num_query, heads, 2]

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        if last_reference_offsets is not None:
            #add reference point of last layer as a PE for ref generations to guided this layer:
            heads = last_reference_offsets.shape[-2]
            last_reference_offsets = last_reference_offsets * spatial_shapes[0, 0] / 2 # upscale normed offsets to curr spatial shapes
            last_reference_offsets = gen_sineembed_for_position(last_reference_offsets, self.embed_dims // heads)
            offset_inputs = query + last_reference_offsets
        else:
            offset_inputs = query

        sampling_offsets = self.sampling_offsets(offset_inputs).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        

        # attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
        #     .reshape(bs, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        
        # # print(sampling_offsets.shape)
        # sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5)\
        #     .reshape(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)


        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1) / 2
        
        norm_sampling_offsets = + sampling_offsets \
            / offset_normalizer[None, None, None, :, None, :]

        sampling_locations = reference_points[:, :, None, None, None, :] + norm_sampling_offsets
            
        # weighted offsets:
        weighted_sampling_offsets = attention_weights.unsqueeze(-1) * norm_sampling_offsets
        weighted_sampling_offsets = weighted_sampling_offsets[..., 0, :, :]
        weighted_sampling_offsets = weighted_sampling_offsets.sum(-2) # weighted sum (b, q, heads, 2)


        if False:#((IS_CUDA_AVAILABLE and value.is_cuda) or (IS_MLU_AVAILABLE and value.is_mlu)):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity, weighted_sampling_offsets

class TemporalFG_TSALayer(nn.Module):
    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_points: int = 4,
                 im2col_step: int = 64,
                 pred_t: int = 8,
                 dropout: float = 0.1,
                 value_proj_ratio: float = 1.0):
        super(TemporalFG_TSALayer, self).__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.pred_t = pred_t
        self.dropout = dropout

        self.fg_tsa_layers = nn.ModuleList([FlowGuidedTSA(embed_dims, num_heads,
        num_levels=1, num_points=num_points,im2col_step=im2col_step, dropout=dropout,
        batch_first=True) for _ in range(pred_t)])
    

    @staticmethod
    def get_reference_points(H, W, bs=1, pred_t=8, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, pred_t, 1, 1)
        return ref_2d
    
    def forward(self, 
        last_layer_query,
        enc_output,
        last_layer_ref_offsets=None):
        """
        last_layer_query : shape (b, c, t, h, w)
        enc_output: shape (b, c, h, w)
        last_layer_ref_offsets: (b, t, h//2*w/2, head, 2)
        """
        b, c, t, h, w = last_layer_query.shape

        reference_points = self.get_reference_points(h, w, b, t, last_layer_query.device)

        if last_layer_ref_offsets is not None:
            offset_heads = last_layer_ref_offsets.shape[-2]
            last_layer_ref_offsets = last_layer_ref_offsets.reshape(b, t, h//2, w//2, offset_heads*2)
            last_layer_ref_offsets = last_layer_ref_offsets.permute(0, 4, 1, 2, 3)
            last_layer_ref_offsets = temporal_upsample(last_layer_ref_offsets, mode='bilinear')
            last_layer_ref_offsets = last_layer_ref_offsets.reshape(b, offset_heads*2, t, h*w).permute(0, 2, 3, 1)
            last_layer_ref_offsets = last_layer_ref_offsets.reshape(b, t, h*w, offset_heads, 2)

        temperal_value = enc_output#torch.cat([enc_output.unsqueeze(2), last_layer_query[:, :, :-1, :, :]],dim=2)

        last_layer_query = last_layer_query.permute(0, 2, 3, 4, 1).reshape(b, t, h*w, c)
        temperal_value = temperal_value.permute(0, 2, 3, 4, 1).reshape(b, t, h*w, c)

        tsa_outputs, tsa_refs = [], []
        for i in range(self.pred_t):
            offset = last_layer_ref_offsets[:, i] if last_layer_ref_offsets is not None else None
            output, refs = self.fg_tsa_layers[i](
                query=last_layer_query[:, i],
                value=temperal_value[:, i],
                reference_points=reference_points[:, i],
                spatial_shapes=torch.tensor(
                        [[h, w]], device=last_layer_query.device),
                level_start_index=torch.tensor([0], device=last_layer_query.device),
                last_reference_offsets=offset
            )
            tsa_outputs.append(output)
            tsa_refs.append(refs)
        
        tsa_outputs = torch.stack(tsa_outputs, dim=1)
        tsa_refs = torch.stack(tsa_refs, dim=1)
        tsa_outputs = tsa_outputs.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        
        return tsa_outputs, tsa_refs

def test_fg_tsa():
    H = 16
    d = 384
    head = 12
    inputs = torch.randn((4, d, 8, H, H))
    enc_inputs = torch.randn((4, d, H, H))
    flow_offsets = torch.randn((4, 8, H * H//4, head, 2))

    tsa_layer = TemporalFG_TSALayer(embed_dims=d, num_heads=head)

    print(sum([v.numel() for v in tsa_layer.parameters()]))

    tsa_outputs, tsa_refs = tsa_layer(inputs, enc_inputs, flow_offsets)

if __name__=='__main__':
    test_fg_tsa()