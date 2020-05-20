
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

