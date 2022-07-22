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





def main(bin_size, reads_to_include):
    clustering_method = "pckmeans" 
    num_chrs = 19

    data = read_data(clustering_method, reads_to_include) #cells with less than 150 reads are deleted: 80.,  84., 105., 113.

    cum_lens = get_chr_cumulative_lengths()


    zygote_ids = data.loc[data.stage == "zygote", 'cell_index'].unique()
    matches = np.zeros((len(zygote_ids),len(zygote_ids)))

    for i in tqdm(range(len(zygote_ids))):
        for j in range(i, len(zygote_ids)):
            cell_i_index = zygote_ids[i]
            cell_j_index = zygote_ids[j]

            cell_i = data.loc[(data.cell_index==cell_i_index) & (data.chr < 20)].copy()
            cell_i['abs_pos'] = -1
            cell_i['abs_pos'] = cell_i.pos.copy() + [cum_lens[ch-1] for ch in cell_i.chr] #encodes the absolute position of the reads along the linear genome
            cell_j = data.loc[(data.cell_index==cell_j_index) & (data.chr < 20)].copy()
            cell_j['abs_pos'] = -1
            cell_j['abs_pos'] = cell_j.pos.copy() + [cum_lens[ch-1] for ch in cell_j.chr] #encodes the absolute position of the reads along the linear genome


            bins, num_bins_per_chr = get_bins(bin_size, cum_lens, num_chrs)

            cell_i_dist, groups_i = pckmeans_get_dist_mat_binned(cell_i, bins, num_bins_per_chr)
            cell_j_dist, groups_j = pckmeans_get_dist_mat_binned(cell_j, bins, num_bins_per_chr)




            seqs = []
            bin_seqs_i = []
            bin_seqs_j = []
            dists = []

            num_samples = 50
            for sample in range(num_samples):
                order = np.arange(1,20)
                np.random.shuffle(order)
                d, bit_seq, bin_seq_j, bin_seq_i = get_aligned_inter_cell_dist(cell_i_dist, cell_j_dist, num_bins_per_chr, chr_seq = order) #np.arange(19,0,-1)


                bin_seqs_i.append(bin_seq_i)
                bin_seqs_j.append(bin_seq_j)
                dists.append(d[0])

            min_seq_i = bin_seqs_i[np.argmin(dists)]
            min_seq_j = bin_seqs_j[np.argmin(dists)]

            i_hap = np.array(groups_i.iloc[min_seq_i].pckmeans_cluster_hap)
            j_hap = np.array(groups_j.iloc[min_seq_j].pckmeans_cluster_hap)
            mask = ~(np.isnan(i_hap) | np.isnan(j_hap))
            try:
                assert np.all(np.logical_xor(i_hap[mask], j_hap[mask])) or np.all(i_hap[mask] == j_hap[mask])
            except:
                print("cells {} and {} did not match".format(cell_i_index, cell_j_index))
  
  

            
if __name__ == "__main__":
    for bin_size in [200e6, 100e6, 50e6, 25e6]:
        for reads_to_include in ["all", "inliers"]:
            print(bin_size, reads_to_include)
            main(bin_size, reads_to_include)
            print("\n")
            
            
            
