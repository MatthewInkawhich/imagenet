# Matthew Inkawhich

"""
Define the Strider model.
"""
from collections import namedtuple
import math
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import  Conv2d
from torch.nn import  ConvTranspose2d
from . import fpn as fpn_module


################################################################################
### Helpers
################################################################################
# Convert scalar pad to tuple pad (left, right, top, bottom)
# Note: scalar padding represents how much padding to add on all sides
# In .5 cases, favor right, bottom
# Ex: scalar_pad=3.5 --> (3, 4, 3, 4)
def get_pad_tuple(scalar_pad):
    left = math.floor(scalar_pad)
    right = math.ceil(scalar_pad)
    return (left, right, left, right)



################################################################################
### Strider Backbone Module
################################################################################
class StriderClassifier(nn.Module):
    def __init__(self, cfg, num_classes=1000):
        super(StriderClassifier, self).__init__()

        # Assert correct config format
        assert (len(cfg['BODY_CHANNELS']) == len(cfg['BODY_CONFIG'])), "Body channels config must equal body config"
        assert (len(cfg['BODY_CHANNELS']) == len(cfg['OUTPUT_SIZES'])), "Body channels config must equal output_sizes"
        assert (len(cfg['BODY_CHANNELS']) == len(cfg['RETURN_FEATURES'])), "Body channels config must equal return features"

        # Build Strider backbone
        self.body = Strider(cfg)

        # Build FPN
        self.use_fpn = cfg['USE_FPN']
        if self.use_fpn:
            out_channels = cfg['FPN_OUT_CHANNELS']
            in_channels_list = [a[1][-1] for a in zip(cfg['RETURN_FEATURES'], cfg['BODY_CHANNELS']) if a[0]]

            self.fpn = fpn_module.FPN(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                conv_block = fpn_module.conv_block(),
                top_blocks=fpn_module.LastLevelMaxPool(),
            )
        else:
            out_channels = cfg['OUT_CHANNELS']

        # Build classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, num_classes)

        
    def forward(self, x):
        # Forward thru backbone
        #print("input:", x.shape)
        x = self.body(x)
        #print("backbone out:")
        #for i in range(len(x)):
        #    print("\ni:", i)
        #    print("shape:", x[i].shape)
        #    print("mean activation:", x[i].mean())

        # Forward thru FPN
        if self.use_fpn:
            x = self.fpn(x)

            #print("fpn out:")
            #for i in range(len(x)):
            #    print("\ni:", i)
            #    print("shape:", x[i].shape)
            #    print("mean activation:", x[i].mean())

            # Merge FPN outputs
            out = x[0]
            for i in range(1, len(x)):
                out = out + F.interpolate(x[i], size=out.shape[-2:], mode='nearest')
            out = out / len(x)
        else:
            if len(x) > 1:
                print("Error: Length of backbone output > 1, but no FPN is used. Check RETURN_FEATURES config...")
                exit()
            out = x[0]


        #print("out:", out.shape, out.mean())
        #exit()

        # Forward thru classification head
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out




################################################################################
### Strider Backbone Module
################################################################################
class Strider(nn.Module):
    def __init__(self, cfg):
        """
        Arguments:
            cfg object which contains necessary configs for model building and running
        """
        super(Strider, self).__init__()
        self.norm_func = nn.BatchNorm2d

        # Construct Stem
        if cfg['STEM_CONFIG'] == "BASE":
            self.stem = BaseStem(cfg['STEM_CHANNELS'][0], self.norm_func)
        else:
            self.stem = Stem(cfg['STEM_CONFIG'], cfg['STEM_CHANNELS'], self.norm_func)

        # Construct Blocks
        self.block_names = []
        self.return_features = {}
        self.output_sizes = cfg['OUTPUT_SIZES']
        version = cfg['VERSION']
        body_channels = cfg['BODY_CHANNELS']
        body_config = cfg['BODY_CONFIG']
        return_features = cfg['RETURN_FEATURES']
        stride_option = cfg['STRIDE_OPTION']
        full_residual = cfg['FULL_RESIDUAL']
        dilations = cfg['DILATIONS']
        weighted_fusion = cfg['WEIGHTED_FUSION']
        for i in range(len(body_channels)):
            name = "block" + str(i)
            in_channels = body_channels[i][0]
            bottleneck_channels = body_channels[i][1]
            out_channels = body_channels[i][2]

            # If the current element of reg_bottlenecks is not empty, build a regular Bottleneck
            if body_config[i][0] == 0:
                stride = body_config[i][1][0]
                dilation = body_config[i][1][1]
                block = Bottleneck(
                            in_channels=in_channels,
                            bottleneck_channels=bottleneck_channels,
                            out_channels=out_channels,
                            stride=stride,
                            dilation=dilation,
                            norm_func=self.norm_func,
                            full_residual=full_residual,
                        )
            
            # Else, build a StriderBlock
            else:
                if version == 1:
                    block = StriderBlock(
                                in_channels=in_channels,
                                bottleneck_channels=bottleneck_channels,
                                out_channels=out_channels,
                                stride_option=stride_option,
                                dilations=dilations,
                                norm_func=self.norm_func,
                                full_residual=full_residual,
                                weighted_fusion=weighted_fusion,
                            )
                elif version == 2:
                    block = StriderBlockv2(
                                in_channels=in_channels,
                                bottleneck_channels=bottleneck_channels,
                                out_channels=out_channels,
                                stride_option=stride_option,
                                dilations=dilations,
                                norm_func=self.norm_func,
                                full_residual=full_residual,
                                weighted_fusion=weighted_fusion,
                            )
                else:
                    print("Error: Strider version not recognized")
                    exit()

            self.add_module(name, block)
            self.block_names.append(name)
            self.return_features[name] = return_features[i]



        # Initialize layers
        for n, m in self.named_modules():
            #print("\nn:", n)
            #print("m:", m)
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                #print("Conv2d initialization!")
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.kaiming_uniform_(m.weight, a=1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                #print("Norm initialization!")
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
        for n, p in self.named_parameters():
            #print("\nn:", n)
            if 'conv2_weight' in n:
                #print("Initializing conv2_weight parameter")
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')


        # Initialize specialty layers
        for m in self.modules():
            if isinstance(m, WeightedFusionModule):
                nn.init.normal_(m.conv.weight, std=0.01)
                nn.init.constant_(m.conv.bias, 0)

        

    def forward(self, x):
        #print("input:", x.shape)
        outputs = []
        x = self.stem(x)
        #print("stem:", x.shape, x.mean())
        for i, block_name in enumerate(self.block_names):
            x = getattr(self, block_name)(x, self.output_sizes[i])
            #print(i, block_name, x.shape, x.mean())
            if self.return_features[block_name]:
                #print("Adding to return list")
                outputs.append(x)
        
        
        #for i in range(len(outputs)):
        #    print("\ni:", i)
        #    print("shape:", outputs[i].shape)
        #    print("mean activation:", outputs[i].mean())
        #    print("frac of nonzero activations:", (outputs[i] != 0).sum().float() / outputs[i].numel())
        #exit()

        return outputs



################################################################################
### Stem
################################################################################
class Stem(nn.Module):
    """
    Stem module
    Use group norm
    """
    def __init__(self, stem_config, stem_channels, norm_func):
        super(Stem, self).__init__()

        # Initialize layers
        layers = []

        # Iterate over stem_config, build stem
        in_channels = 3
        for i in range(len(stem_channels)):
            # Construct padding
            pad_tuple = get_pad_tuple(stem_config[i][2])
            layers.append(nn.ZeroPad2d(pad_tuple))
            # Construct layer
            conv = Conv2d(in_channels, stem_channels[i], kernel_size=stem_config[i][0], stride=stem_config[i][1], bias=False)
            layers.append(conv)
            # Initialize norm
            layers.append(norm_func(stem_channels[i]))
            # Initialize nonlinearity
            layers.append(nn.ReLU(inplace=True))
            # Update in_channels
            in_channels = stem_channels[i]

        # Combine layers into module
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)



class BaseStem(nn.Module):
    def __init__(self, out_channels, norm_func):
        super(BaseStem, self).__init__()
        # Define conv layer
        self.conv1 = Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_func(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x



################################################################################
### WeightedFusionModule
################################################################################
class WeightedFusionModule(nn.Module):
    def __init__(self, in_channels, num_branches):
        super(WeightedFusionModule, self).__init__()
        self.conv = Conv2d(in_channels, num_branches, kernel_size=3, stride=1, padding=1)

    def forward(self, x, output_size):
        # Forward thru conv
        x = self.conv(x)
        # Normalize channel values to percentages
        x = F.softmax(x, dim=1)
        # Resize output
        if output_size == 0:
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        elif output_size == 2:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        # Depth-wise normalize
        s = x.sum(dim=1, keepdim=True)
        x = x / s
        # Convert to a list with each element representing a channel (i.e. branch)
        x = x.permute(1, 0, 2, 3).unsqueeze_(2)
        out = [a for a in x]
        return out
        


################################################################################
### StriderBlockv2
################################################################################
class StriderBlockv2(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        stride_option,
        dilations,
        norm_func,
        full_residual=False,
        weighted_fusion=False,
    ):
        super(StriderBlockv2, self).__init__()
        
        self.stride_option = stride_option
        self.dilations = dilations
        self.weighted_fusion = weighted_fusion
        self.num_branches = 3

        ### Initialize Weighted Fusion Module if necessary
        if self.weighted_fusion:
            self.wfm = WeightedFusionModule(in_channels, num_branches=self.num_branches)

        ### Residual layer
        self.downsample = None
        if in_channels != out_channels or full_residual:
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=1, bias=False
                ),
                norm_func(out_channels),
            )


        ### Conv1
        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        

        ### Conv2 
        # The Conv2 stage consists of parallel branches with different strides/dilations
        conv2_subchannels = math.ceil(bottleneck_channels / self.num_branches)
        self.conv2_0 = Conv2d(
            bottleneck_channels,
            conv2_subchannels,
            kernel_size=3,
            stride=2,
            dilation=self.dilations[0],
            padding=self.dilations[0],
            bias=False,
        )
        self.bn2_0 = norm_func(conv2_subchannels)
        
        self.conv2_1 = Conv2d(
            bottleneck_channels,
            conv2_subchannels,
            kernel_size=3,
            stride=1,
            dilation=self.dilations[1],
            padding=self.dilations[1],
            bias=False,
        )
        self.bn2_1 = norm_func(conv2_subchannels)
        
        self.conv2_2 = ConvTranspose2d(
            bottleneck_channels,
            conv2_subchannels,
            kernel_size=3,
            stride=2,
            dilation=self.dilations[2],
            padding=self.dilations[2],
            output_padding=1,
            bias=False,
        )
        self.bn2_2 = norm_func(conv2_subchannels)


        # Conv layers for resize stage
        self.resize0 = Conv2d(
            conv2_subchannels,
            bottleneck_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn_r0 = norm_func(bottleneck_channels)

        self.resize1 = Conv2d(
            conv2_subchannels,
            bottleneck_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn_r1 = norm_func(bottleneck_channels)

        self.resize2 = Conv2d(
            conv2_subchannels,
            bottleneck_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn_r2 = norm_func(bottleneck_channels)


        ### Conv3
        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn3 = norm_func(out_channels)



    def forward(self, x, output_size):
        # Store copy of input feature
        identity = x

        # Forward thru WeightedFusionModule if necessary
        if self.weighted_fusion:
            fusion_weights = self.wfm(identity, output_size)

        # Forward thru residual conv
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Forward thru conv1
        out = self.conv1(x)
        out = self.bn1(out)
        conv1_out = F.relu_(out)
    
        ### Forward thru conv2
        conv2_branch_outputs = []
        # Processing stage
        out = self.conv2_0(conv1_out)
        out = self.bn2_0(out)
        out = F.relu_(out)
        conv2_branch_outputs.append(out)

        out = self.conv2_1(conv1_out)
        out = self.bn2_1(out)
        out = F.relu_(out)
        conv2_branch_outputs.append(out)

        out = self.conv2_2(conv1_out)
        out = self.bn2_2(out)
        out = F.relu_(out)
        conv2_branch_outputs.append(out)

        # Resize stage
        if output_size == 0:
            conv2_branch_outputs[1] = F.avg_pool2d(conv2_branch_outputs[1], kernel_size=3, stride=2, padding=1)
            conv2_branch_outputs[2] = F.avg_pool2d(F.avg_pool2d(conv2_branch_outputs[2], kernel_size=3, stride=2, padding=1), kernel_size=3, stride=2, padding=1)
        elif output_size == 1:
            conv2_branch_outputs[0] = F.interpolate(conv2_branch_outputs[0], size=conv2_branch_outputs[1].shape[-2:], mode='nearest')
            conv2_branch_outputs[2] = F.avg_pool2d(conv2_branch_outputs[2], kernel_size=3, stride=2, padding=1)
        elif output_size == 2:
            conv2_branch_outputs[0] = F.interpolate(conv2_branch_outputs[0], size=conv2_branch_outputs[2].shape[-2:], mode='nearest')
            conv2_branch_outputs[1] = F.interpolate(conv2_branch_outputs[1], size=conv2_branch_outputs[2].shape[-2:], mode='nearest')
        else:
            print("Error: Invalid output_size parameter in StriderBlock forward function")
            exit()

        conv2_branch_outputs[0] = self.resize0(conv2_branch_outputs[0])
        conv2_branch_outputs[0] = self.bn_r0(conv2_branch_outputs[0])
        conv2_branch_outputs[0] = F.relu_(conv2_branch_outputs[0])

        conv2_branch_outputs[1] = self.resize1(conv2_branch_outputs[1])
        conv2_branch_outputs[1] = self.bn_r1(conv2_branch_outputs[1])
        conv2_branch_outputs[1] = F.relu_(conv2_branch_outputs[1])

        conv2_branch_outputs[2] = self.resize2(conv2_branch_outputs[2])
        conv2_branch_outputs[2] = self.bn_r2(conv2_branch_outputs[2])
        conv2_branch_outputs[2] = F.relu_(conv2_branch_outputs[2])

        # Weight stage (if option is set)
        if self.weighted_fusion:
            conv2_branch_outputs[0] = conv2_branch_outputs[0] * fusion_weights[0]
            conv2_branch_outputs[1] = conv2_branch_outputs[1] * fusion_weights[1]
            conv2_branch_outputs[2] = conv2_branch_outputs[2] * fusion_weights[2]

        # Fuse branch outputs
        out = conv2_branch_outputs[0] + conv2_branch_outputs[1] + conv2_branch_outputs[2]
        if not self.weighted_fusion:
            out = out / len(conv2_branch_outputs)

        # Conv3 stage
        out = self.conv3(out)
        out = self.bn3(out)

        # Add residual
        if output_size == 0:
            identity = F.avg_pool2d(identity, kernel_size=3, stride=2, padding=1)
        elif output_size == 2:
            identity = F.interpolate(identity, size=out.shape[-2:], mode='nearest')
        out += identity
        out = F.relu_(out)
        
        return out



################################################################################
### StriderBlock
################################################################################
class StriderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        stride_option,
        dilations,
        norm_func,
        full_residual=False,
        weighted_fusion=False,
    ):
        super(StriderBlock, self).__init__()
        
        self.stride_option = stride_option
        self.dilations = dilations
        self.weighted_fusion = weighted_fusion
        self.num_branches = 3

        ### Initialize Weighted Fusion Module if necessary
        if self.weighted_fusion:
            self.wfm = WeightedFusionModule(in_channels, num_branches=self.num_branches)

        ### Residual layer
        self.downsample = None
        if in_channels != out_channels or full_residual:
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=1, bias=False
                ),
                norm_func(out_channels),
            )


        ### Conv1
        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        

        ### Conv2
        # Conv2 must be represented by a tensor instead of a Module, as we
        # need to use the weights with different strides.
        self.conv2_weight = nn.Parameter(
            torch.Tensor(bottleneck_channels, bottleneck_channels, 3, 3)
        )
        self.bn2 = norm_func(bottleneck_channels)


        ### Conv3
        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn3 = norm_func(out_channels)



    def forward(self, x, output_size):
        # Store copy of input feature
        identity = x

        # Forward thru WeightedFusionModule if necessary
        if self.weighted_fusion:
            fusion_weights = self.wfm(identity, output_size)

        # Forward thru residual conv
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Forward thru conv1
        out = self.conv1(x)
        out = self.bn1(out)
        conv1_out = F.relu_(out)
    
        # Forward thru all branches
        branch_outputs = []

        # 2x DOWN branch
        if self.stride_option in [0, 1, 2]:
            dilation = self.dilations[0]
            # Conv2
            out = F.conv2d(conv1_out, self.conv2_weight, stride=2, padding=dilation, dilation=dilation)
            out = self.bn2(out)
            out = F.relu_(out)
            # Conv3
            out = self.conv3(out)
            out = self.bn3(out)
            # Add residual
            out += F.avg_pool2d(identity, kernel_size=3, stride=2, padding=1)
            out = F.relu_(out)
            branch_outputs.append(out)

        # 1x SAME branch
        if self.stride_option in [0, 1, 3]:
            dilation = self.dilations[1]
            # Conv2
            out = F.conv2d(conv1_out, self.conv2_weight, stride=1, padding=dilation, dilation=dilation)
            out = self.bn2(out)
            out = F.relu_(out)
            # Conv3
            out = self.conv3(out)
            out = self.bn3(out)
            # Add residual
            out += identity
            out = F.relu_(out)
            branch_outputs.append(out)

        # 2x UP branch
        if self.stride_option in [0, 2, 3]:
            dilation = self.dilations[2]
            # (T)Conv2
            # Note: We want the Transposed conv with stride=2 to act like a conv with stride=1/2, 
            #       so we have to permute the in/out channels and flip the kernels to match the implementation.
            out = F.conv_transpose2d(conv1_out, self.conv2_weight.flip([2, 3]).permute(1, 0, 2, 3), stride=2, padding=dilation, output_padding=1, dilation=dilation)
            out = self.bn2(out)
            out = F.relu_(out)
            # Conv3
            out = self.conv3(out)
            out = self.bn3(out)
            # Add residual
            out += F.interpolate(identity, size=out.shape[-2:], mode='nearest')
            out = F.relu_(out)
            branch_outputs.append(out)


        # Resize branch outputs
        if output_size == 0:
            branch_outputs[1] = F.avg_pool2d(branch_outputs[1], kernel_size=3, stride=2, padding=1)
            branch_outputs[2] = F.avg_pool2d(F.avg_pool2d(branch_outputs[2], kernel_size=3, stride=2, padding=1), kernel_size=3, stride=2, padding=1)
        elif output_size == 1:
            branch_outputs[0] = F.interpolate(branch_outputs[0], size=branch_outputs[1].shape[-2:], mode='nearest')
            branch_outputs[2] = F.avg_pool2d(branch_outputs[2], kernel_size=3, stride=2, padding=1)
        elif output_size == 2:
            branch_outputs[0] = F.interpolate(branch_outputs[0], size=branch_outputs[2].shape[-2:], mode='nearest')
            branch_outputs[1] = F.interpolate(branch_outputs[1], size=branch_outputs[2].shape[-2:], mode='nearest')
        else:
            print("Error: Invalid output_size parameter in StriderBlock forward function")
            exit()

        #print("\n\n")
        #for fw in fusion_weights:
        #    print(fw, fw.shape)

        # Scale each branch output by weights (optional)
        if self.weighted_fusion:
            branch_outputs[0] = branch_outputs[0] * fusion_weights[0]
            branch_outputs[1] = branch_outputs[1] * fusion_weights[1]
            branch_outputs[2] = branch_outputs[2] * fusion_weights[2]

        # Fuse branch outputs
        out = branch_outputs[0] + branch_outputs[1] + branch_outputs[2]
        if not self.weighted_fusion:
            out = out / len(branch_outputs)

        return out



################################################################################
### Bottleneck
################################################################################
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        stride,
        dilation,
        norm_func,
        use_downsample=False,
        num_groups=1,
        full_residual=False,
    ):
        super(Bottleneck, self).__init__()

        ### Downsample layer (on residual)
        self.downsample = None
        if in_channels != out_channels or full_residual:
            down_stride = stride
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )


        ### First conv
        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        

        ### Middle conv
        padding = dilation
        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            groups=num_groups,
        )
        self.bn2 = norm_func(bottleneck_channels)


        ### Third conv
        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn3 = norm_func(out_channels)



    def forward(self, x, dummy):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out



