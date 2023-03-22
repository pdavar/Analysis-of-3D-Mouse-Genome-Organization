import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import collections
import time
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import scipy as sp
from tqdm import tqdm
from sklearn.manifold import MDS
from run_dist_mat import *
from chromosome_alignment import *
from scipy.cluster.hierarchy import dendrogram, linkage
import itertools
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from itertools import repeat


def robustness_analysis():
    reads_to_inlcude = "inliers" #"all"
    clustering_method = "pckmeans" # "igs"
    num_chrs = 19

    data = read_data(clustering_method, reads_to_inlcude) #cells with less than 150 reads are deleted: 80.,  84., 105., 113.

    cum_lens = get_chr_cumulative_lengths()

    fig, axes = plt.subplots(4,4, figsize = (20,20)) 
    for i, bin_size in tqdm(enumerate([200e6, 100e6, 50e6, 25e6])):
        for j, num_samples_for_resampling in tqdm(enumerate([5, 25, 50, 75])):
            print("\n bin size: ", bin_size)
            print("\n num samples: ", num_samples)
            
            proportion_matching = []
            variances = []
            cell_i_index = 91
            cell_j_index = 93


            cell_i = data.loc[(data.cell_index==cell_i_index) & (data.chr < 20)].copy()
            cell_i['abs_pos'] = -1
            cell_i['abs_pos'] = cell_i.pos.copy() + [cum_lens[ch-1] for ch in cell_i.chr] #encodes the absolute position of the reads along the linear genome
            cell_j = data.loc[(data.cell_index==cell_j_index) & (data.chr < 20)].copy()
            cell_j['abs_pos'] = -1
            cell_j['abs_pos'] = cell_j.pos.copy() + [cum_lens[ch-1] for ch in cell_j.chr] #encodes the absolute position of the reads along the linear genome


            bins, num_bins_per_chr = get_bins(bin_size, cum_lens, num_chrs)
            
            
            num_trials = 40
            min_dists = []
            for trial in range(num_trials):
            
                bin_resampling_dists = []
                for bin_resampling in range(num_samples_for_resampling):
                    cell_i_dist,_ = pckmeans_get_dist_mat_binned_resample(cell_i, bins, num_bins_per_chr)
                    cell_j_dist,_ = pckmeans_get_dist_mat_binned_resample(cell_j, bins, num_bins_per_chr)


                    num_samples_for_ordering = 50
                    ordering_dists = []
                    random_orders = np.zeros((num_samples_for_ordering, 19))
                    for counter, sample in enumerate(range(num_samples_for_ordering)):
                        order = np.arange(1,20)
                        np.random.shuffle(order)
                        random_orders[counter, :] = order
                    
                    
                     ### parallelizing:
                    num_workers = 4
                    with Pool(num_workers) as p:                       
                        ordering_dists.append(p.starmap(get_aligned_inter_cell_dist, zip(repeat(cell_i_dist), repeat(cell_j_dist), repeat(num_bins_per_chr), repeat(19), random_orders))[0][0])#the first [0] gives the distance component of the output, the second [0] gets the actual distance and not the size of the intersection 
              
                    bin_resampling_dists.append(np.round(np.min(ordering_dists), 4))
                min_dists.append(np.min(bin_resampling_dists))
            
            axes[j,i].scatter(np.zeros_like(min_dists), min_dists)
            axes[j,i].set_title("bin size {}".format(bin_size/1e6))
            axes[j,i].set_ylabel("sample size: {}".format(num_samples_for_resampling))

    plt.suptitle("cell indeces {} and {}".format(cell_i_index, cell_j_index))
    plt.savefig("figures/sequential_algorithm_bin_resampling_analysis_cells{}_{}.png".format(cell_i_index, cell_j_index))

