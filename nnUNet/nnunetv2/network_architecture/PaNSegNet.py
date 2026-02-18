from typing import Union, Type, List, Tuple

import torch
from nnunetv2.dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD,StackedResidualBlocks
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch import Tensor
import json
import numpy as np
from typing import Optional
import math
import copy

def duplicate(module:nn.Module, N:int):
    """
    Produce N identical layers.
    Args:
        module: the model for copy
        N: the copy times
    """
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def linear_attention(query:Tensor, key:Tensor, value:Tensor, mask:Optional[Tensor]=None, dropout=None):
    """
    Compute the linear attention between q, k , v
    Implementation of:
        https://arxiv.org/abs/1812.01243
    Input query_size:
        [batch_size, nhead, N, d_k]
    """
    d_model = query.size(-1)
    query = F.softmax(query, dim=-1) / math.sqrt(d_model)
    '''
    key = F.gelu(key)
    value = F.gelu(value)
    '''
    if mask is not None:
        key = key.masked_fill_(~mask, -1e9)
        value = value.masked_fill_(~mask, 0)

    key = F.softmax(key, dim=-2)
    context = torch.einsum('bhnd, bhne->bhde', key, value)

    if dropout is not None:
        score_softmax = dropout(query)
    else:
        score_softmax = None

    out = torch.einsum('bhnd, bhde->bhne', query, context)

    return out, score_softmax


class Conv3dPosEmbedding(nn.Module):
    '''
    Positinal Encoding Generator using 3d convolution
    Args:
        dim: the input feature dimension
        dropout: the dropout ratio
        emb_kernel: the kernel size of convolution
            padding_size = emb_kernel // 2
    '''
    def __init__(self, dim, dropout:float, emb_kernel:int=3):
        super(Conv3dPosEmbedding, self).__init__()
        self.proj = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=emb_kernel,
                              stride=1, padding=emb_kernel//2, groups=dim)
        
        self.dropout = nn.Dropout3d(p=dropout)
    
    def forward(self, x):
        """
        Args:
            Input: 
                size, [batch, channels, heights, widths, depths]
            Output:
                size is same with Input
        """
        pos_enc = self.proj(x)
        x = x + pos_enc
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,nhead:int=8,dropout:float=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead # the default value is 8
        self.linears = duplicate(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.d_k = d_model // nhead

    def forward(self, query:Tensor, key:Tensor, value:Tensor, src_mask:Optional[Tensor]=None):
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1)
        n_batch = query.size(0)
        n_head = self.nhead
        query, key, value = [l(x).view(n_batch, -1, self.nhead, self.d_k).transpose(1, 2)for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = linear_attention(query, key, value, src_mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.nhead * self.d_k)
        return self.linears[-1](x)

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model:int, dim_feedforward:int,nhead:int=8, dropout:float=0.1,activation="gelu", layer_norm_eps=1e-6):
        super(SelfAttentionLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.layer_norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.layer_norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.activation = F.gelu if activation == "gelu" else F.relu
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        else:
            self.activation = nn.GELU()
    
    def forward(self, x:Tensor, src_mask:Optional[Tensor]=None):
        x1 = self.self_attn(x, x, x, src_mask=src_mask)
        x = x + self.dropout1(x1)
        x = self.layer_norm1(x)

        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.layer_norm2(x)
        return x


class SpatialAttention3DBlock(nn.Module):
    def __init__(self, in_channel1, in_channel2, 
                 inter_channel, kernel_size:int=1, stride:int=1):
        super().__init__()
        self.W_x = nn.Conv3d(in_channels=in_channel1, 
                             out_channels=inter_channel,
                             kernel_size=kernel_size,
                             stride=stride)
        self.W_g = nn.Conv3d(in_channels=in_channel2, 
                             out_channels=inter_channel,
                             kernel_size=kernel_size,
                             stride=stride)

        self.psi = nn.Sequential(nn.Conv3d(in_channels=inter_channel, 
                                           out_channels=1,
                                           kernel_size=kernel_size,
                                           stride=stride),
                                 nn.Sigmoid())
        self.W = nn.Sequential(nn.Conv3d(in_channels=in_channel1,
                                         out_channels=in_channel1,
                                         kernel_size=kernel_size,
                                         stride=stride),
                               nn.InstanceNorm3d(inter_channel))

    def forward(self, x, up):
        x_inter = self.W_x(x)
        up = self.W_g(up)
        attn = self.psi(F.relu(x_inter+up, inplace=False))
        return self.W(x*attn)


class SelfAttention3DBlock(nn.Module):
    def __init__(self, in_feature1, in_features2, d_model, nhead:int, dropout:float=0.3, N:int=8, is_up=False):
        super(SelfAttention3DBlock, self).__init__()
        self.in_feature1 = in_feature1
        self.d_model = d_model
        self.is_up = is_up
        self.W_x = nn.Conv3d(in_channels=in_feature1, 
                             out_channels=d_model,
                             kernel_size=1,
                             stride=1)
        if is_up:
            self.W_g = nn.Conv3d(in_channels=in_features2, 
                                 out_channels=d_model,
                                 kernel_size=1,
                                 stride=1)
        pos_encode = Conv3dPosEmbedding(dim=d_model, dropout=dropout, emb_kernel=3)
        attn_layer = SelfAttentionLayer(d_model=d_model, dim_feedforward=2*d_model, nhead=nhead, dropout=dropout)
        self.layer_num = N
        self.layers = duplicate(attn_layer, N)
        self.pos_encoders = duplicate(pos_encode, N)
        self.activation = nn.ReLU(inplace=False)
    
    def forward(self, x: torch.Tensor, up: Optional[torch.Tensor]):
        '''
        x: Input x size
            [nbatch, channel, height, width, depth]
        up: up layer from bottom
            [nbatch, channel, height, width, depth]
        '''
        nbatch, _, height, width, depth = x.shape
        x = self.W_x(x)
        if self.is_up:
            up = self.W_g(up)
            x = self.activation(x+up)
        else:
            x = self.activation(x)

        for i in range(self.layer_num):
            x = x.flatten(start_dim=2).transpose_(1, 2)
            x = self.layers[i](x)
            x = x.transpose_(1, 2).reshape(nbatch, -1, height, width, depth)
            x = self.pos_encoders[i](x)
            
        return x


class LinTranUnetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        # we start with the bottleneck and work out way up
        stages = []
        att_stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedResidualBlocks(
                n_blocks=n_conv_per_stage[s - 1],
                conv_op=encoder.conv_op,
                input_channels=2 * input_features_skip,
                output_channels=input_features_skip,
                kernel_size=encoder.kernel_sizes[-(s + 1)],
                initial_stride=1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ))
            # the core implementation of the LinTranUnetDecoder is in the stages
            if s<3:
                current_stage = SelfAttention3DBlock(in_feature1=input_features_skip, in_features2=input_features_skip, d_model=input_features_skip,
                                                     nhead=8, N=8)
            else:
                current_stage = SpatialAttention3DBlock(in_channel1=input_features_skip, in_channel2=input_features_skip, 
                                                        inter_channel=input_features_skip)

            att_stages.append(current_stage)

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.att_stages = nn.ModuleList(att_stages)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output


class PaNSegNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
    ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, (
            "n_blocks_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_blocks_per_stage: {n_blocks_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        # PaNSegNet/LinTranUnetDecoder encoder is a residual encoder
        self.encoder = ResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            disable_default_stem=False,
            stem_channels=stem_channels,
        )
        self.decoder = LinTranUnetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


def get_network_architecture(config):
    pass


if __name__ == "__main__":
    # Load configuration from testPlans.json
    config = json.load(open("/home/pyq6817/Multi-modal-Pancreas-Segmentation/nnUNet/nnunetv2/network_architecture/testPlans.json"))
    
    # Get 3D full resolution configuration
    cfg = config['configurations']['3d_fullres']
    arch = cfg['architecture']
    arch_kwargs = arch['arch_kwargs']
    print(arch_kwargs)
    print("="*50)
    print("Testing PaNSegNet Encoder")
    print("="*50)
    
    # Convert string-based class references to actual classes
    import pydoc
    if 'conv_op' in arch_kwargs and isinstance(arch_kwargs['conv_op'], str):
        arch_kwargs['conv_op'] = pydoc.locate(arch_kwargs['conv_op'])
    if 'norm_op' in arch_kwargs and isinstance(arch_kwargs['norm_op'], str):
        arch_kwargs['norm_op'] = pydoc.locate(arch_kwargs['norm_op'])
    if 'nonlin' in arch_kwargs and isinstance(arch_kwargs['nonlin'], str):
        arch_kwargs['nonlin'] = pydoc.locate(arch_kwargs['nonlin'])
    if 'dropout_op' in arch_kwargs and isinstance(arch_kwargs['dropout_op'], str):
        arch_kwargs['dropout_op'] = pydoc.locate(arch_kwargs['dropout_op'])
    
    # Create PaNSegNet model with encoder only
    try:
        # Extract necessary parameters from config
        input_channels = 1  # adjust based on your data
        num_classes = 2  # adjust based on your segmentation task
        
        # Create model
        model = PaNSegNet(
            input_channels=input_channels,
            n_stages=arch_kwargs['n_stages'],
            features_per_stage=arch_kwargs['features_per_stage'],
            conv_op=arch_kwargs['conv_op'],
            kernel_sizes=arch_kwargs['kernel_sizes'],
            strides=arch_kwargs['strides'],
            n_blocks_per_stage=arch_kwargs['n_conv_per_stage'],
            num_classes=num_classes,
            n_conv_per_stage_decoder=arch_kwargs['n_conv_per_stage_decoder'],
            conv_bias=arch_kwargs['conv_bias'],
            norm_op=arch_kwargs['norm_op'],
            norm_op_kwargs=arch_kwargs['norm_op_kwargs'],
            dropout_op=arch_kwargs['dropout_op'],
            dropout_op_kwargs=arch_kwargs['dropout_op_kwargs'],
            nonlin=arch_kwargs['nonlin'],
            nonlin_kwargs=arch_kwargs['nonlin_kwargs']
        )
        
        print(f"✓ Successfully created PaNSegNet model!")
        print(f"  Model type: {type(model)}")
        
        # Create test input tensor (batch_size=1, channels=1, depth=80, height=160, width=192)
        # Using patch size from config
        patch_size = cfg['patch_size']
        test_input = torch.randn(1, input_channels, patch_size[0], patch_size[1], patch_size[2])
        print(f"  Input shape: {test_input.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ Encoder forward pass successful!")
        print(f"  Output type: {type(output)}")
        
        # Print encoder output information
        if isinstance(output, list):
            print(f"  Number of skip connections: {len(output)}")
            for i, skip in enumerate(output):
                print(f"    Skip {i}: {skip.shape}")
        else:
            print(f"  Output shape: {output.shape}")
            
    except Exception as e:
        print(f"✗ Error creating or testing PaNSegNet model: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("Testing completed")
    print("="*50)

