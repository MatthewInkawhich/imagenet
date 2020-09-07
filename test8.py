import torch
import random
import math
import itertools

#body_config = [[1], [0, [1,1]], [0, [1,1]], [0, [2,1]], [0, [1,1]], [0, [1,1]], [0, [1,1]], [0, [2,1]], [0, [1,1]], [0, [1,1]], [0, [1,1]], [0, [1,1]], [0, [1,1]], [0, [2,1]], [0, [1,1]], [0, [1,1]]]
body_config = [[1], [0, [1,1]], [0, [1,1]], [1], [0, [1,1]], [0, [1,1]], [0, [1,1]], [0, [2,1]], [0, [1,1]], [0, [1,1]], [0, [1,1]], [0, [1,1]], [0, [1,1]], [0, [2,1]], [0, [1,1]], [0, [1,1]]]
#body_config = [[1], [0, [1,1]], [0, [1,1]], [1], [0, [1,1]], [0, [1,1]], [0, [1,1]], [1], [0, [1,1]], [0, [1,1]], [0, [1,1]], [0, [1,1]], [0, [1,1]], [0, [2,1]], [0, [1,1]], [0, [1,1]]]
#body_config = [[1], [0, [1,1]], [0, [1,1]], [1], [0, [1,1]], [0, [1,1]], [0, [1,1]], [1], [0, [1,1]], [0, [1,1]], [1], [0, [1,1]], [0, [1,1]], [1], [0, [1,1]], [0, [1,1]]]
downsample_bounds = [[64,4], [64,4], [64,4], [64,4], [64,4], [64,4], [64,4], [64,8], [64,8], [64,8], [64,8], [64,8], [64,8], [64,8], [64,8], [64,8]]
num_ss_blocks = sum([1 if x[0] == 1 else 0 for x in body_config])
stride_options = [[False, [1, 1]], [False, [1, 2]], [False, [2, 1]], [False, [2, 2]], [True, [1, 2]], [True, [2, 1]], [True, [2, 2]]]
stride_options_scales = [[1/x[1][0], 1/x[1][1]] if x[0] else x[1] for x in stride_options]
num_stride_options = len(stride_options)

# Create list of all possible combinations
option_list = list(range(len(stride_options)))
all_combos = list(itertools.product(option_list, repeat=num_ss_blocks))
valid_combos = []

# Trim stride options that are invalid due to bounds
for i in range(len(all_combos)):
    valid = True
    curr_downsample = [4, 4]  # [dH, dW]
    adaptive_idx = 0
    # Iterate over network configs to check downsample rate
    for layer_idx in range(len(body_config)):
        # If the curr layer is adaptive
        if body_config[layer_idx][0] == 1:
            stride = stride_options_scales[all_combos[i][adaptive_idx]]
            curr_downsample = [s1*s2 for s1, s2 in zip(curr_downsample, stride)]
            adaptive_idx += 1    
        # If the curr layer is NOT adaptive
        else:
            stride_side = body_config[layer_idx][1][0]
            stride = [stride_side, stride_side]
            curr_downsample = [s1*s2 for s1, s2 in zip(curr_downsample, stride)]
        # Check if curr_downsample is now out of bounds
        curr_bounds = downsample_bounds[layer_idx]
        if curr_downsample[0] > curr_bounds[0] or curr_downsample[1] > curr_bounds[0] or curr_downsample[0] < curr_bounds[1] or curr_downsample[1] < curr_bounds[1]:
            valid = False
            break   # Out of bounds, do NOT consider this stride combo
    if valid:    
        valid_combos.append(all_combos[i])

        

print("All:", all_combos, len(all_combos))
print("\n\nValid:", valid_combos, len(valid_combos))



####################################################################################
# Now that we have all valid_combos, we need to give all unique prefixes
prefixes = []
for i in range(len(valid_combos)):
    for j in range(len(valid_combos[i])):
        prefixes.append(valid_combos[i][:j])

prefixes = list(set(prefixes))
prefixes.sort()  # Sort by value
prefixes.sort(key=lambda t: len(t))

print("\nvalid prefixes:", prefixes)


####################################################################################
# Create play data
num_samples = 1
all_losses_shape = [num_stride_options for i in range(num_ss_blocks)]
all_losses_shape.insert(0, num_samples)
all_losses = torch.rand(all_losses_shape)
print("all_losses:", all_losses, all_losses.shape)

####################################################################################
# Assuming that we have recorded losses for all samples/combos, now we create the selector_truth lookup
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

print("\n\n\n")
selector_truth = [{} for i in range(num_samples)]
for sample_idx in range(num_samples):
    # Start new dict for this sample
    curr_dict = {}
    # Iterate over prefixes
    for prefix in prefixes:
        print("\nprefix:", prefix)
        # Get coord of min loss for the current prefix
        prefix_with_sample_idx = (sample_idx,) + prefix
        # Record the truth_idx (we want to find the stride option that comes immediately AFTER prefix)
        truth_idx = len(prefix_with_sample_idx)
        print("prefix_with_sample_idx:", prefix_with_sample_idx)
        min_idx = torch.argmin(all_losses[prefix_with_sample_idx])
        print("size_of_argmin_tensor:", tuple(all_losses[prefix_with_sample_idx].shape))

        print("min_idx:", min_idx)
        min_coord = unravel_index(min_idx.item(), tuple(all_losses[prefix_with_sample_idx].shape))
        min_coord_full = prefix_with_sample_idx + min_coord
        print("min_coord:", min_coord)
        print("min_coord_full:", min_coord_full)

        # The truth is the index value corresponding to the truth_idx
        print("truth_idx:", truth_idx)
        truth = min_coord_full[truth_idx]
        print("truth:", truth)
        # Update current sample's dict
        curr_dict[prefix] = truth

    selector_truth[sample_idx].update(curr_dict)


print("\n\nselector_truth")
for k, v in selector_truth[0].items():
    print(k, v)
