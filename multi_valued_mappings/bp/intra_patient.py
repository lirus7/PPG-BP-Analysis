import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
from numpy import linalg as LA

from tqdm.contrib.concurrent import process_map


def align(ref, can, ref_sign, can_sign):
    """
    Function for aligning two signals.

    Args:
        ref (numpy_arr): normalized reference signal
        can (numpy_arr): normalized candidate signal
        ref_sign (numpy_arr): reference signal
        can_sign (numpy_arr): candidate signal

    Returns:
        _type_: 2 numpy arrays
    """
    corr = np.correlate(ref, can, mode="same")
    delay = np.argmax(corr) - (len(ref) // 2)
    if delay == 0:
        return ref_sign, can_sign
    elif delay > 0:
        return ref_sign[delay:], can_sign[:-delay]
    else:
        return ref_sign[:delay], can_sign[abs(delay) :]


def check(i):
    """
    For each record index, It finds signals which are above the cross-correlation threshold and systolic blood pressure threshold

    Args:
        i (int): Record index
    """
    count = 0
    ans = []
    temp = np.zeros((len(ppg[i])))
    for j in range(len(ppg[i])):
        abp_arr = abp[i][j]
        sbp_val = sbp[i][j]
        for k in range(len(ppg[i])):
            corr = np.correlate(normalized_ppg[i][j], normalized_ppg[i][k]) / max(
                len(normalized_ppg[i][j]), len(normalized_ppg[i][k])
            )  # Obtain the correlation value between the normalized signals
            if corr >= args.cross_corr_threshold:
                ppg_1, ppg_2 = align(
                    normalized_ppg[i][j], normalized_ppg[i][k], ppg[i][j], ppg[i][k]
                )
                dist = np.linalg.norm(ppg_1 - ppg_2) / max(
                    np.max(ppg_1), np.max(ppg_2)
                )  # Calculate the Euclidean Distance
                if dist < args.euclid_threshold and (
                    abs(np.mean(abp_arr) - np.mean(abp[i][k])) > args.sbp_threshold
                    or abs(sbp_val - sbp[i][k]) > args.sbp_threshold
                ):
                    count += 1
                    temp[k] = 1
                    ans.append([j, k, corr, dist])
    np.save(
        f"{args.dst_path}/raw/{i}.npy",
        np.asarray([len(normalized_ppg[i]), count, np.sum(temp)]),
    )  # Store the total number of collisions present in each record
    np.save(
        f"{args.dst_path}/val/{i}.npy", np.asarray(ans, dtype=object)
    )  # Stores all the collisions with indices and values


parser = argparse.ArgumentParser()
parser.add_argument(
    "--normalized_ppg_path",
    help="Path of Normalized PPG numpy array with each element representing the PPG signal",
    default="../../data/sbp/norm_ppg.npy",
)

parser.add_argument(
    "--ppg_path",
    help="Path of PPG numpy array with each element representing the PPG signal",
    default="../../data/sbp/ppg.npy",
)

parser.add_argument(
    "--sbp_path",
    help="Path of corresponding SBP numpy array with each element representing the SBP value",
    default="../../data/sbp/bp.npy",
)
parser.add_argument(
    "--abp_path",
    help="Path of corresponding ABP numpy array with each element representing the ABP signal",
    default="../../data/sbp/abp.npy",
)

parser.add_argument(
    "--dst_path",
    help="Path for storing the logs after calculation",
    default="../../logs/bp/intra_pat_logs",
)

parser.add_argument(
    "--cross_corr_threshold",
    help="Cross-Correlation threshold for filtering",
    type=float,
    default=0.95,
)

parser.add_argument(
    "--euclid_threshold", help="Euclidean threshold filtering", type=float, default=5
)


parser.add_argument(
    "--max_workers",
    help="Maximum number of workers for multi-processing",
    type=int,
    default=12,
)
parser.add_argument("--sbp_threshold", help="SBP threshold", type=int, default=8)


args = parser.parse_args()

normalized_ppg = np.load(f"{args.normalized_ppg_path}", allow_pickle=True)
ppg = np.load(f"{args.ppg_path}", allow_pickle=True)
abp = np.load(f"{args.abp_path}", allow_pickle=True)
sbp = np.load(f"{args.sbp_path}", allow_pickle=True)

os.makedirs(args.dst_path + "/raw/", exist_ok=True)
os.makedirs(args.dst_path + "/val/", exist_ok=True)

process_map(check, [i for i in range(0, ppg.shape[0])], max_workers=args.max_workers)
# On a 12 core machine this function should finish in <5 minutes.
