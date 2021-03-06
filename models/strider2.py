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
### Strider Classifier Module
################################################################################
class StriderClassifier(nn.Module):
    def __init__(self, cfg, num_classes=1000):
        super(StriderClassifier, self).__init__()

        # Assert correct config format
        assert (len(cfg['BODY_CHANNELS']) == len(cfg['BODY_CONFIG'])), "Body channels config must equal body config"
        assert (len(cfg['BODY_CHANNELS']) == len(cfg['RETURN_FEATURES'])), "Body channels config must equal return features"
        assert (len(cfg['BODY_CHANNELS']) == len(cfg['DOWNSAMPLE_BOUNDS'])), "Body channels config must equal downsample bounds"

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

        
    def forward(self, x, stride_combo=[]):
        # Forward thru backbone
        #print("input:", x.shape)
        x = self.body(x, stride_combo)
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
        self.stem = Stem(cfg['STEM_CHANNELS'], self.norm_func)

        # Construct Blocks
        self.block_names = []
        self.return_features = {}
        body_channels = cfg['BODY_CHANNELS']
        body_config = cfg['BODY_CONFIG']
        stride_options = cfg['STRIDE_OPTIONS']
        return_features = cfg['RETURN_FEATURES']
        full_residual = cfg['FULL_RESIDUAL']
        lr_residual = cfg['LR_RESIDUAL']
        lr_adaptive_fusion = cfg['LR_ADAPTIVE_FUSION']
        fpn_adaptive_fusion = cfg['FPN_ADAPTIVE_FUSION']

        for i in range(len(body_channels)):
            name = "block" + str(i)
            in_channels = body_channels[i][0]
            bottleneck_channels = body_channels[i][1]
            out_channels = body_channels[i][2]

            # If the current element is 0, build a regular Bottleneck
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
                block = StriderBlock(
                            in_channels=in_channels,
                            bottleneck_channels=bottleneck_channels,
                            out_channels=out_channels,
                            stride_options=stride_options,
                            norm_func=self.norm_func,
                            full_residual=full_residual,
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
                            self.norm_func(body_channels[l][2])
                        )
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

        for n, p in self.named_parameters():
            #print("\nn:", n)
            if 'conv2_weight' in n:
                #print("Initializing conv2_weight parameter")
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')

        # Zero-init last BN in each block
        # https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, (Bottleneck, StriderBlock)):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constatn_(m.bn2.weight, 0)

        # Initialize specialty layers
        for m in self.modules():
            if isinstance(m, AdaptiveFusionModule):
                nn.init.normal_(m.conv.weight, std=0.01)
                nn.init.constant_(m.conv.bias, 0)



    def forward(self, x, stride_combo):
        #print("input:", x.shape)
        all_outputs = []
        outputs = []
        curr_strider_block = 0
        # Forward thru stem
        x = self.stem(x)
        #print("stem:", x.shape)
        for i, block_name in enumerate(self.block_names):
            # Forward thru current block
            if isinstance(getattr(self, block_name), StriderBlock):
                # If it is a StriderBlock pass the extra args and get extra return values
                x = getattr(self, block_name)(x, stride_combo[curr_strider_block])
                curr_strider_block += 1
            else:
                # If it is NOT a StriderBlock, forward as usual
                x = getattr(self, block_name)(x)
            #print("i:{} x:".format(i), x.shape)

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
    def __init__(self, out_channels, norm_func=nn.BatchNorm2d):
        super(Stem, self).__init__()
        # Define conv layer
        self.conv1 = Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        #self.conv1 = Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
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
### StriderBlock
################################################################################
class StriderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        stride_options,
        norm_func,
        full_residual=False,
    ):
        super(StriderBlock, self).__init__()

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
        self.conv2_stride_options = stride_options
        self.conv2_weight = nn.Parameter(
            torch.Tensor(bottleneck_channels, bottleneck_channels, 3, 3))
        self.bn2 = norm_func(bottleneck_channels)


        ### Conv3
        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn3 = norm_func(out_channels)

    
    def forward(self, x, stride_choice):
        # Store copy of input feature
        identity = x

        # Forward thru conv1
        out = self.conv1(x)
        out = self.bn1(out)
        conv1_out = F.relu(out)
    
        # Forward thru conv2
        dilation = 1
        use_tconv = self.conv2_stride_options[stride_choice][0]
        stride = tuple(self.conv2_stride_options[stride_choice][1])

        if use_tconv:
            output_padding = (stride[0]-1, stride[1]-1)
            out = F.conv_transpose2d(conv1_out, self.conv2_weight.flip([2, 3]).permute(1, 0, 2, 3), stride=stride, padding=dilation, output_padding=output_padding, dilation=dilation)
        else:
            out = F.conv2d(conv1_out, self.conv2_weight, stride=stride, padding=dilation, dilation=dilation)
        out = self.bn2(out)
        out = F.relu_(out)

        # Conv3 stage
        out = self.conv3(out)
        out = self.bn3(out)

        # Add residual
        if self.downsample is not None:
            identity = self.downsample(identity)
        identity = FeatureResize(identity, out.shape[-2:])
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



    def forward(self, x):
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
        #out = F.relu_(out)  # Commenting this out because Strider with LRR connections takes care of this in Strider forward

        return out



class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels, 
        stride,
        dilation=1,
        norm_func=nn.BatchNorm2d,
    ):
        super(BasicBlock, self).__init__()

        ### Downsample layer (on residual)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
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
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
        )
        self.bn1 = norm_func(out_channels)

        ### Second conv
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = norm_func(out_channels)


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out



################################################################################
### StrideSelectorModule
################################################################################
class StrideSelectorModule(nn.Module):
    def __init__(self, num_stride_options):
        super(StrideSelectorModule, self).__init__()
        self.num_stride_options = num_stride_options

        self.stem = Stem(64)

        # C2
        self.bb1 = BasicBlock(64, 64, 1)
        self.bb2 = BasicBlock(64, 64, 1)
        self.bb3 = BasicBlock(64, 64, 1)

        # C3
        self.bb4 = BasicBlock(64, 128, 2)
        self.bb5 = BasicBlock(128, 128, 1)
        self.bb6 = BasicBlock(128, 128, 1)
        self.bb7 = BasicBlock(128, 128, 1)

        # C4
        self.bb8 = BasicBlock(128, 256, 2)
        self.bb9 = BasicBlock(256, 256, 1)
        self.bb10 = BasicBlock(256, 256, 1)
        self.bb11 = BasicBlock(256, 256, 1)
        self.bb12 = BasicBlock(256, 256, 1)
        self.bb13 = BasicBlock(256, 256, 1)

        # C5
        self.bb14 = BasicBlock(256, 512, 2)
        self.bb15 = BasicBlock(512, 512, 1)
        self.bb16 = BasicBlock(512, 512, 1)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_stride_options)

        # Manual initializations
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-init last BN in each block
        # https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        # Run convs
        x = self.stem(x)
        x = self.bb1(x)
        x = self.bb2(x)
        x = self.bb3(x)
        x = self.bb4(x)
        x = self.bb5(x)
        x = self.bb6(x)
        x = self.bb7(x)
        x = self.bb8(x)
        x = self.bb9(x)
        x = self.bb10(x)
        x = self.bb11(x)
        x = self.bb12(x)
        x = self.bb13(x)
        x = self.bb14(x)
        x = self.bb15(x)
        x = self.bb16(x)
        # GAP
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # FC
        x = self.fc(x)
        return x


