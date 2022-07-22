import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import collections
import pickle
import time
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import scipy as sp
from sklearn.manifold import MDS
from run_dist_mat import *
from chromosome_alignment import *
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from sklearn.metrics import silhouette_score, silhouette_samples 
from scipy.spatial import distance


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)





def main():
    reads_to_include = "inliers" #"all"
    clustering_method = "pckmeans" # "igs"
    num_chrs = 19
    for bin_size in [1e6]:

        data = read_data(clustering_method, reads_to_include) #cells with less than 150 reads are deleted: 80.,  84., 105., 113.

        cum_lens = get_chr_cumulative_lengths()

        cell_i_inds = [90, 90, 91, 74, 99, 99, 81]
        cell_j_inds = [81, 91, 92, 75, 100, 101,83]
        fig, axes = plt.subplots(1,7, figsize = (30,8)) 
        for i in range(len(cell_i_inds)):
            cell_i_index = cell_i_inds[i]
            cell_j_index = cell_j_inds[i]

            cell_i = data.loc[(data.cell_index==cell_i_index) & (data.chr < 20)].copy()
            cell_i['abs_pos'] = -1
            cell_i['abs_pos'] = cell_i.pos.copy() + [cum_lens[ch-1] for ch in cell_i.chr] #encodes the absolute position of the reads along the linear genome
            cell_j = data.loc[(data.cell_index==cell_j_index) & (data.chr < 20)].copy()
            cell_j['abs_pos'] = -1
            cell_j['abs_pos'] = cell_j.pos.copy() + [cum_lens[ch-1] for ch in cell_j.chr] #encodes the absolute position of the reads along the linear genome


            bins, num_bins_per_chr = get_bins(bin_size, cum_lens, num_chrs)
            dists = []
            colors = []
            for random_seed in tqdm(range(0,50)):
                cell_i_dist, _ = pckmeans_get_dist_mat_binned_resample(cell_i, bins, num_bins_per_chr, random_seed)
                cell_j_dist, _ = pckmeans_get_dist_mat_binned_resample(cell_j, bins, num_bins_per_chr, random_seed)


                num_samples = 50
                for sample in range(num_samples):
                    order = np.arange(1,20)
                    np.random.shuffle(order)
                    d, bit_seq, bin_seq,_ = get_aligned_inter_cell_dist(cell_i_dist, cell_j_dist, num_bins_per_chr, chr_seq = order) #np.arange(19,0,-1)
                dists.append(d[0])
                colors.append('black')


            #distance with taking the mean
            cell_i_dist,_ = pckmeans_get_dist_mat_binned(cell_i, bins, num_bins_per_chr)
            cell_j_dist,_ = pckmeans_get_dist_mat_binned(cell_j, bins, num_bins_per_chr)
            num_samples = 50
            for sample in range(num_samples):
                order = np.arange(1,20)
                np.random.shuffle(order)
                d, bit_seq, bin_seq,_= get_aligned_inter_cell_dist(cell_i_dist, cell_j_dist, num_bins_per_chr, chr_seq = order) #np.arange(19,0,-1)
            dists.append(d[0])
            colors.append('red')

            axes[i].scatter(np.ones_like(dists), dists, color = colors)
            axes[i].boxplot(dists)
            axes[i].set_title("aligning cell {} with cell {}".format(cell_i_index, cell_j_index), fontsize = 15)

        plt.suptitle("minimum distances of different resamplings (50 orders in each sampling): \n bin size {} Mbp {} reads".format(int(bin_size/1e6), reads_to_include), y = 1.0, fontsize = 20)
        plt.tight_layout()
        plt.savefig("figures/sequential_alignment_resample_analysis_bin_size_{}_{}.png".format(int(bin_size/1e6), reads_to_include), bbox_inches='tight')

    
    
    
    
if __name__ == "__main__":
    main()