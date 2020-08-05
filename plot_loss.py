# MatthewInkawhich

"""
This script plots a line graph that depicts the training loss.
Designed to take a log.txt file (logger output) as input.
"""

import argparse
import os
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt


##################################
### HELPERS
##################################
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)



##################################
### MAIN
##################################
def main():
    # Get log path from arg
    parser = argparse.ArgumentParser(description="Training Loss Plotter")
    parser.add_argument(
        "logpath",
        type=str,
    )
    # parser.add_argument(
    #     "--title",
    #     type=str,
    #     default="Loss Plot",
    # )
    args = parser.parse_args()

    # Read file into list of lines
    lines = [line.rstrip('\n') for line in open(args.logpath)]

    # Filter down to lines of interest
    good_lines = []
    for i in range(len(lines)):
        if lines[i] == "SS Choice Counts:":
            good_lines.append(lines[i - 1].lstrip())

    # Separate and parse training/validation lines
    s1_training_epochs = []
    s1_training_losses = []
    s1_training_accs = []
    s2_training_epochs = []
    s2_training_losses = []
    s2_training_accs = []
    #validation_epochs = []
    validation_accs = []
    for i in range(len(good_lines)):
        print(good_lines[i])
        split = good_lines[i].split()
        # Validation case
        if split[0] == '*':
            prev_split = good_lines[i - 1].split()
            #epoch = int(prev_split[2].replace('[', '').replace(']', ''))
            acc = float(split[2])
            #validation_epochs.append(epoch)
            validation_accs.append(split[2])
        # Stage 1 case
        if 'S1' in good_lines[i]:
            epoch = int(split[2].replace('[', '').replace(']', ''))
            loss = float(split[14].replace('(', '').replace(')', ''))
            acc = float(split[18].replace('(', '').replace(')', ''))
            s1_training_epochs.append(epoch)
            s1_training_losses.append(loss)
            s1_training_accs.append(acc)
        # Stage 2 case
        if 'S2' in good_lines[i]:
            epoch = int(split[2].replace('[', '').replace(']', ''))
            loss = float(split[14].replace('(', '').replace(')', ''))
            acc = float(split[18].replace('(', '').replace(')', ''))
            s2_training_epochs.append(epoch)
            s2_training_losses.append(loss)
            s2_training_accs.append(acc)

    # Update stage2 epochs
    s2_training_epochs_cont = [x + s1_training_epochs[-1] + 1 for x in s2_training_epochs]

    print("S1 training")
    for i in range(len(s1_training_epochs)):
        print(s1_training_epochs[i], s1_training_losses[i], s1_training_accs[i])

    print("S2 training")
    for i in range(len(s2_training_epochs)):
        print(s2_training_epochs_cont[i], s2_training_losses[i], s2_training_accs[i])

    print("Validation")
    for i in range(len(validation_accs)):
        print(validation_accs[i])


    # Plot losses/accs
    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.2))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)

    host.set_xlabel('Epoch')
    host.set_ylabel('Training Accuracy')
    par1.set_ylabel('Loss (Cross-Entropy)')
    par2.set_ylabel('Loss (Smooth-L1)')

    acc_color = 'green'
    # ce_loss_color = '#fa433e'
    # l1_loss_color = '#A42004'
    ce_loss_color = 'tab:red'
    l1_loss_color = 'tab:purple'

    p1, = host.plot(s1_training_epochs, s1_training_accs, color=acc_color, linestyle='-')
    p2, = host.plot(s2_training_epochs_cont, s2_training_accs, color=acc_color, linestyle='--')
    p3, = par1.plot(s1_training_epochs, s1_training_losses, color=ce_loss_color, linestyle='-')
    p4, = par2.plot(s2_training_epochs_cont, s2_training_losses, color=l1_loss_color, linestyle='--')

    host.set_xlim(0, 135)
    host.set_ylim(0, 80)
    #par1.set_ylim(0, 4)
    #par2.set_ylim(0, 1)

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p3.get_color())
    par2.yaxis.label.set_color(p4.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p4.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    fig.suptitle(args.logpath.split('/')[-1])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()







if __name__ == "__main__":
    main()
