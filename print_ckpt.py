import argparse
import torch

# Get path to checkpoint file from command line
parser = argparse.ArgumentParser(description='Print Checkpoint Information')
parser.add_argument('ckpt', metavar='FILE', help='path to ckpt file')
args = parser.parse_args()

ckpt_dict = torch.load(args.ckpt, map_location='cpu')

print("best_acc1:", ckpt_dict['best_acc1'])
