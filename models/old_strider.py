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
"""
Convert scalar pad to tuple pad (left, right, top, bottom)
Note: scalar padding represents how much padding to add on all sides
In .5 cases, favor right, bottom
Ex: scalar_pad=3.5 --> (3, 4, 3, 4)
"""
def get_pad_tuple(scalar_pad):
    left = math.floor(scalar_pad)
    right = math.ceil(scalar_pad)
    return (left, right, left, right)

"""
Takes a torch tensor (feature) of shape NxCxHxW, and resizes the resolution
until it matches target_resolution. Note that target_resolution height/width 
should be larger/smaller by a factor of 2.
"""
def FeatureResize(x, target_resolution):
    # If x's shape already matches target, return
    if x.shape[-2:][0] == target_resolution[0] and x.shape[-2:][1] == target_resolution[1]:
        return x
    # If x's H or W is smaller than target's, interpolate up
    elif x.shape[-2:][0] < target_resolution[0] or x.shape[-2:][1] < target_resolution[1]:
        return F.interpolate(x, size=target_resolution, mode='nearest')
    # x's H/W are larger than target, pool to size
    else:
        return F.adaptive_avg_pool2d(x, target_resolution)
              


################################################################################
### Strider Backbone Module
################################################################################
class StriderClassifier(nn.Module):
    def __init__(self, cfg, num_classes=1000):
        super(StriderClassifier, self).__init__()

        # Assert correct config format
        assert (len(cfg['BODY_CHANNELS']) == len(cfg['BODY_CONFIG'])), "Body channels config must equal body config"
        assert (len(cfg['BODY_CHANNELS']) == len(cfg['OUTPUT_INDEXES'])), "Body channels config must equal output indexes"
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
            #print("\ni:", i)
            #print("shape:", x[i].shape)
            #print("mean activation:", x[i].mean())

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
        self.output_indexes = cfg['OUTPUT_INDEXES']
        version = cfg['VERSION']
        body_channels = cfg['BODY_CHANNELS']
        body_config = cfg['BODY_CONFIG']
        branch_config = cfg['BRANCH_CONFIG']
        return_features = cfg['RETURN_FEATURES']
        full_residual = cfg['FULL_RESIDUAL']
        lr_residual = cfg['LR_RESIDUAL']
        sb_adaptive_fusion = cfg['SB_ADAPTIVE_FUSION']
        lr_adaptive_fusion = cfg['LR_ADAPTIVE_FUSION']
        fpn_adaptive_fusion = cfg['FPN_ADAPTIVE_FUSION']


        ### Construct blocks
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
                block = StriderBlockv2(
                            in_channels=in_channels,
                            bottleneck_channels=bottleneck_channels,
                            out_channels=out_channels,
                            branch_config = branch_config,
                            norm_func=self.norm_func,
                            full_residual=full_residual,
                            adaptive_fusion=sb_adaptive_fusion,
                        )

            self.add_module(name, block)
            self.block_names.append(name)
            self.return_features[name] = return_features[i]


        ### Construct LRR layers
        # Construct lr_residual_dict as we go
        self.lr_residual_dict = {}
        for l in range(len(body_channels)):
            self.lr_residual_dict[l] = []
            for lrr in lr_residual:
                p = l - lrr
                if p >= 0:
                    name = ""
                    # Only construct/use a lrr_module if output depth does not match
                    if body_channels[l][2] != body_channels[p][2]:
                        # Naming convention: to_ENDINDEX_from_SOURCEINDEX
                        name = "lrr_to_{}_from_{}".format(l, p)
                        lrr_module = nn.Sequential(
                            Conv2d(
                                body_channels[p][2],
                                body_channels[l][2],
                                kernel_size=1,
                                bias=False,
                            ),
                            self.norm_func(body_channels[l][2]))
                        self.add_module(name, lrr_module)
                    self.lr_residual_dict[l].append((p, name))

        print("lr_residual_dict:")
        for k, v in self.lr_residual_dict.items():
            print(k, v)

        ### Manually initialize layers
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

        
        #for n, p in self.named_parameters():
        #    #print("\nn:", n)
        #    if 'conv2_weight' in n:
        #        #print("Initializing conv2_weight parameter")
        #        nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')


        # Initialize specialty layers
        for m in self.modules():
            if isinstance(m, AdaptiveFusionModule):
                nn.init.normal_(m.conv.weight, std=0.01)
                nn.init.constant_(m.conv.bias, 0)


    def forward(self, x):
        #print("input:", x.shape)
        all_outputs = []
        outputs = []
        # Forward thru stem
        x = self.stem(x)
        #print("stem:", x.shape, x.mean())
        for i, block_name in enumerate(self.block_names):
            # Forward thru current block
            x = getattr(self, block_name)(x, self.output_indexes[i])
            # Perform long range residual fusion
            for (lrr_idx, lrr_name) in self.lr_residual_dict[i]:
                lrr_feat = all_outputs[lrr_idx]
                # First, process the lrr feature
                lrr_feat = FeatureResize(lrr_feat, x.shape[-2:])
                if lrr_name:
                    lrr_feat = getattr(self, lrr_name)(lrr_feat)
                # Next, add the lrr_feat to x
                x += lrr_feat
            # Normalize the fusion; only perform if fusion was done
            if self.lr_residual_dict[i]:
                x /= len(self.lr_residual_dict[i])
            # Perform nonlinearity AFTER fusion
            x = F.relu(x)

            all_outputs.append(x)
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
### AdaptiveFusionModule
################################################################################
class AdaptiveFusionModule(nn.Module):
    def __init__(self, in_channels, num_branches):
        super(AdaptiveFusionModule, self).__init__()
        self.conv = Conv2d(in_channels, num_branches, kernel_size=3, stride=1, padding=1)

    def forward(self, x, output_resolution):
        # Forward thru conv
        x = self.conv(x)
        # Resize output
        x = FeatureResize(x, output_resolution)
        # Normalize channel values to percentages
        x = F.softmax(x, dim=1)
        # Convert to a list with each element representing a channel (i.e. branch)
        x = x.permute(1, 0, 2, 3).unsqueeze(2)
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
        branch_config,
        norm_func,
        full_residual=False,
        adaptive_fusion=False,
    ):
        super(StriderBlockv2, self).__init__()
        
        self.branch_config = branch_config
        self.num_branches = len(branch_config)
        self.adaptive_fusion = adaptive_fusion

        ### Initialize Weighted Fusion Module if necessary
        if self.adaptive_fusion:
            self.afm = AdaptiveFusionModule(in_channels, num_branches=self.num_branches)

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

        for i in range(self.num_branches):
            transposed = self.branch_config[i][0]
            stride = tuple(self.branch_config[i][1])
            dilation = tuple(self.branch_config[i][2])
        
            # Construct 3x3 conv
            if transposed:
                conv2 = ConvTranspose2d(
                    bottleneck_channels,
                    conv2_subchannels,
                    kernel_size=3,
                    stride=stride,
                    dilation=dilation,
                    padding=dilation,
                    output_padding=1,
                    bias=False,
                )

            else:
                conv2 = Conv2d(
                    bottleneck_channels,
                    conv2_subchannels,
                    kernel_size=3,
                    stride=stride,
                    dilation=dilation,
                    padding=dilation,
                    bias=False,
                )
            bn2 = norm_func(conv2_subchannels)

            # Construct Resize layers
            resize = Conv2d(
                conv2_subchannels,
                bottleneck_channels,
                kernel_size=1,
                bias=False,
            )
            bnr = norm_func(bottleneck_channels)

            self.add_module("conv2_{}".format(i), conv2)
            self.add_module("bn2_{}".format(i), bn2)
            self.add_module("resize_{}".format(i), resize)
            self.add_module("bnr_{}".format(i), bnr)


        ### Conv3
        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn3 = norm_func(out_channels)



    def forward(self, x, output_index):
        # Store copy of input feature
        identity = x

        # Forward thru conv1
        out = self.conv1(x)
        out = self.bn1(out)
        conv1_out = F.relu(out)
    
        ### Forward thru conv2
        # Forward thru 3x3s
        conv2_branch_outputs = []
        for i in range(self.num_branches):
            out = getattr(self, "conv2_"+str(i))(conv1_out)
            out = getattr(self, "bn2_"+str(i))(out)
            out = F.relu(out)
            conv2_branch_outputs.append(out)

        # Store desired output resolution
        output_resolution = conv2_branch_outputs[output_index].shape[-2:]

        #print("\nBefore resize:")
        #for i in range(self.num_branches):
        #    print(i, conv2_branch_outputs[i].shape)

        # Forward thru AFM if necessary
        if self.adaptive_fusion:
            fusion_weights = self.afm(identity, output_resolution)

        # Resize each branch to correct output resolution
        for i in range(self.num_branches):
            conv2_branch_outputs[i] = FeatureResize(conv2_branch_outputs[i], output_resolution)
            # Expand feature depth back to bottleneck channels
            conv2_branch_outputs[i] = getattr(self, "resize_"+str(i))(conv2_branch_outputs[i])
            conv2_branch_outputs[i] = getattr(self, "bnr_"+str(i))(conv2_branch_outputs[i])
            conv2_branch_outputs[i] = F.relu(conv2_branch_outputs[i])
            if self.adaptive_fusion:
                conv2_branch_outputs[i] = conv2_branch_outputs[i] * fusion_weights[i]

        #print("output_resolution:", output_resolution)
        #print("\nAfter resize:")
        #for i in range(self.num_branches):
        #    print(i, conv2_branch_outputs[i].shape)
        #exit()

        # Fuse branch outputs
        out = torch.stack(conv2_branch_outputs, dim=0).sum(dim=0)
        if not self.adaptive_fusion:
            out /= self.num_branches

        # Conv3 stage
        out = self.conv3(out)
        out = self.bn3(out)

        # Add residual
        if self.downsample is not None:
            identity = self.downsample(identity)
        identity = FeatureResize(identity, output_resolution)
        out += identity
        #out = F.relu(out) # Commenting this out because Strider with LRR connections takes care of this in Strider forward
        
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



