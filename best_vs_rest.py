import os
import numpy as np
import matplotlib.pyplot as plt
import torch


truth_path = os.path.join(os.path.expanduser("~"), "WORK", "imagenet", "out", "strider2", "strider_R50_B_random", "selector_truth_val2.pth.tar")
selector_truth = torch.load(truth_path)
num_valid_combos = len(selector_truth[0][0])
print("len(selector_truth):", len(selector_truth))
print("num_valid_combos:", num_valid_combos)

#bvr_loss_diff = [[] for i in range((num_valid_combos - 1))]
#bvr_pt_diff = [[] for i in range((num_valid_combos - 1))]
pts = [[] for i in range(num_valid_combos)]
rank_correct_counts = [0] * num_valid_combos
maxconf_sel_count = 0
maxconf_task_count = 0

for sample_idx in range(len(selector_truth)):
    # Separate vals corresponding to current sample
    curr_losses = np.array(selector_truth[sample_idx][0])
    curr_pts = np.array(selector_truth[sample_idx][1])
    curr_corrects = np.array(selector_truth[sample_idx][2])
    curr_task_truth = selector_truth[sample_idx][3]
    curr_outputs = np.array(selector_truth[sample_idx][4])
    # Sort choices based on loss
    sorted_loss_idxs = np.argsort(curr_losses)
    # Record accuracy based on most confident prediction
    maxconf_per_option = np.max(curr_outputs, axis=1)
    maxconf_sel_pred = np.argmax(maxconf_per_option)
    maxconf_task_pred = np.argmax(curr_outputs[maxconf_sel_pred])
    if maxconf_sel_pred == sorted_loss_idxs[0]:
        maxconf_sel_count += 1
    if maxconf_task_pred == curr_task_truth[0]:
        maxconf_task_count += 1
    #print("\n\n\nsample_idx:", sample_idx)
    #print("curr_losses:", curr_losses)
    #print("curr_corrects:", curr_corrects)
    #print("sorted_loss_idx:", sorted_loss_idx)
    #print("best_idx:", best_idx)
    #print("rest_idxs:", rest_idxs)
    #print("best_correct:", best_correct)
    # Iterate over 'rest's and record losses and pred diffs
    for i, sorted_loss_idx in enumerate(sorted_loss_idxs):
        rank_correct_counts[i] += curr_corrects[sorted_loss_idx]
        pts[i].append(curr_pts[sorted_loss_idx])
       

print("Rank-based Correct:", [x / len(selector_truth) for x in rank_correct_counts])
print("Conf-based Selection Accuracy:", maxconf_sel_count / len(selector_truth))
print("Conf-based Task Accuracy:", maxconf_task_count / len(selector_truth))
#labels = ['1v{}'.format(x) for x in range(2, num_valid_combos+1)]
#plt.boxplot(bvr_loss_diff, labels=labels,  whis=(5, 95), showfliers=False, sym='k.')
#plt.boxplot(bvr_pt_diff, labels=labels, whis=(5, 95), showfliers=False, sym='k.')
plt.boxplot(pts, whis=(5, 95), showfliers=False)
plt.title("Confidence in True Label: Validation Set")
plt.xlabel("Stride Option Rank")
plt.ylabel("P(true label)")
plt.grid(True, 'major', 'y', linewidth=.25)
plt.show()
