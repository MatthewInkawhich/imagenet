# MatthewInkawhich

"""
This script plots a line graph that depicts the training loss.
Designed to take a log.txt file (logger output) as input.
"""

import argparse
import os
import matplotlib.pyplot as plt


##################################
### HELPERS
##################################




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
    parser.add_argument(
        "--title",
        type=str,
        default="Loss Plot",
    )
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
    plt.plot(s1_training_epochs, s1_training_losses)
    #plt.title("Training: {}".format(args.logpath))
    plt.title("Training Loss: {}".format(args.title))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()







if __name__ == "__main__":
    main()
