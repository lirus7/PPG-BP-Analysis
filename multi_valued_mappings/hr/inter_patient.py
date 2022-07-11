import argparse
import os
import random
import time
from multiprocessing import Pool

import numpy as np
from numpy import linalg as LA

from scipy.signal import find_peaks
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


def find_collisions(pair):
    """
    For each record index, It finds signals which are above the cross-correlation threshold and systolic blood pressure threshold by comparing with each signal in other record index.

    Args:
        i (int): Record index
    """
    i, j = pair
    ans = []
    os.makedirs(f"{args.dst_path}/{i}", exist_ok=True)
    for index_1 in range(len(ppg[i])):
        sample_hr_val = hr[i][index_1]
        for index_2 in range(len(ppg[j])):
            corr = np.correlate(
                normalized_ppg[i][index_1], normalized_ppg[j][index_2]
            ) / max(
                len(normalized_ppg[i][index_1]), len(normalized_ppg[j][index_2])
            )  # Obtain the correlation value between the normalized signals
            if corr >= args.cross_corr_threshold:
                ppg_1, ppg_2 = align(
                    normalized_ppg[i][index_1],
                    normalized_ppg[j][index_2],
                    ppg[i][index_1],
                    ppg[j][index_2],
                )

                peaks_1, _ = find_peaks(ppg_1, distance=60)
                peaks_2, _ = find_peaks(ppg_2, distance=60)

                dist = np.linalg.norm(ppg_1 - ppg_2) / max(
                    np.max(abs(ppg_1)), np.max(abs(ppg_2))
                )  # Calculate the Euclidean Distance
                if (
                    dist < args.euclid_threshold
                    and abs(sample_hr_val - hr[j][index_2]) > args.hr_threshold
                ):
                    ans.append([index_1, index_2, corr, dist])
    if ans:
        np.save(
            f"{args.dst_path}/{i}/{j}.npy", np.asarray(ans, dtype=object)
        )  # Stores all the collisions with indices and values


def check(i):
    """
    Helper function to create all the possible pairs and then multiprocess
    """
    pair_list = [(i, j) for j in range(len(normalized_ppg)) if j != i]
    process_map(find_collisions, pair_list, max_workers=args.max_workers)
    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--normalized_ppg_path",
    help="Path of Normalized PPG numpy array with each element representing the PPG signal",
    default="../../data/hr/norm_ppg.npy",
)

parser.add_argument(
    "--ppg_path",
    help="Path of PPG numpy array with each element representing the PPG signal",
    default="../../data/hr/ppg.npy",
)

parser.add_argument(
    "--hr_path",
    help="Path of corresponding hr numpy array with each element representing the HR value",
    default="../../data/hr/hr.npy",
)

parser.add_argument(
    "--dst_path",
    help="Path for storing the logs after calculation",
    default="../../logs/hr/inter_pat_logs",
)

parser.add_argument(
    "--cross_corr_threshold",
    help="Cross-Correlation threshold for filtering",
    type=float,
    default=0.9,
)

parser.add_argument(
    "--euclid_threshold", help="Euclidean threshold filtering", type=float, default=1.0
)


parser.add_argument(
    "--max_workers",
    help="Maximum number of workers for multi-processing",
    type=int,
    default=12,
)
parser.add_argument("--hr_threshold", help="HR threshold", type=int, default=8)


args = parser.parse_args()

normalized_ppg = np.load(f"{args.normalized_ppg_path}", allow_pickle=True)
ppg = np.load(f"{args.ppg_path}", allow_pickle=True)
hr = np.load(f"{args.hr_path}", allow_pickle=True)


random_list = random.sample(range(len(ppg)), 100)
print(random_list)

for i in random_list:
    check(i)
# process_map(check, [i for i in range(0, 20)], max_workers=args.max_workers)
