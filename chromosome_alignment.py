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
from scipy.cluster.hierarchy import dendrogram, linkage
import itertools
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from itertools import repeat


def get_sex_of_cell(cell_data):
    assert cell_data.loc[cell_data.chr == 20].shape[0] > 1, print("data matrix must have sex chromosomes")
    
    if cell_data.loc[cell_data.chr == 21].shape[0] > 1: return 'm'  ##check this
    else: return 'f'
    
def make_groups_by_bins(cell_data, bin_size, cum_lens, include_sex_chromosomes = False):
    

    if include_sex_chromosomes == False:
        bins, num_bins_per_chr = get_bins(bin_size, cum_lens, num_chrs = 19)
        num_bins = np.sum(list(num_bins_per_chr.values())) 
        
        cell_data = cell_data.loc[cell_data.chr < 20].copy()
        cell_data['abs_pos'] = -1
        cell_data['abs_pos'] = cell_data.pos.copy() + [cum_lens[ch-1] for ch in cell_data.chr] #encodes the absolute position of the reads along the linear genome
 
        groups = cell_data.groupby([pd.cut(cell_data.abs_pos, bins),pd.cut(cell_data.pckmeans_cluster, [-0.1,0.9,2])]).mean().reindex(pd.MultiIndex.from_product([bins[1:], [0,1]]), fill_value = np.nan)
        assert groups.shape[0] == 2 * num_bins
        return groups
    elif include_sex_chromosomes == True:
        cell_data = cell_data.loc[cell_data.chr < 22].copy()
        if get_sex_of_cell(cell_data) == 'f':
            print("female cell")
            bins, num_bins_per_chr = get_bins(bin_size, cum_lens, num_chrs = 20)
            autosome_num_bins = np.sum(list(num_bins_per_chr.values())[0:20]) #sum of all autosome chromosomes
            x_num_bins = num_bins_per_chr[20]
            
            cell_data = cell_data.loc[cell_data.chr != 21] #getting rid of the noisy y chromosome reads
            assert cell_data.loc[cell_data.chr == 20, 'pckmeans_cluster'].unique().shape[0] == 2, "x chromosome must have 2 clusters"
            assert cell_data.loc[cell_data.chr == 21, 'pckmeans_cluster'].unique().shape[0] == 0, "y chromosome must have no clusters"
            cell_data['abs_pos'] = -1
            cell_data['abs_pos'] = cell_data.pos.copy() + [cum_lens[ch-1] for ch in cell_data.chr] #encodes the absolute position of the reads along the linear genome

            groups = cell_data.groupby([pd.cut(cell_data.abs_pos, bins),pd.cut(cell_data.pckmeans_cluster, [-0.1,0.9,2])]).mean().reindex(pd.MultiIndex.from_product([bins[1:], [0,1]]), fill_value = np.nan)
            assert groups.shape[0] == 2 * autosome_num_bins + x_num_bins
            return groups
        else: #male cells

            assert cell_data.loc[cell_data.chr == 20, 'pckmeans_cluster'].unique().shape[0] == 1, "x chromosome must have 2 clusters in male embryo"
            assert cell_data.loc[cell_data.chr == 21, 'pckmeans_cluster'].unique().shape[0] == 1, "y chromosome must have 2 clusters in male embryo"
            cell_data['abs_pos'] = -1
            cell_data['abs_pos'] = cell_data.pos.copy() + [cum_lens[ch-1] for ch in cell_data.chr] #encodes the absolute position of the reads along the linear genome

            bins, num_bins_per_chr = get_bins(bin_size, cum_lens, num_chrs = 21)
            
            autosome_num_bins = np.sum(list(num_bins_per_chr.values())[0:20]) #sum of all autosome chromosomes
            x_num_bins = num_bins_per_chr[20]
            y_num_bins = num_bins_per_chr[21]
            
            autosome_bins = bins[0:autosome_num_bins+1]
            x_bins = bins[autosome_num_bins: autosome_num_bins+x_num_bins+1]
            y_bins = bins[autosome_num_bins+x_num_bins:]
            
            autosome_chrs = cell_data.loc[cell_data.chr <= 19]
            x_chr = cell_data.loc[cell_data.chr == 20]
            y_chr = cell_data.loc[cell_data.chr == 21]
            
            autosome_chr_groups = autosome_chrs.groupby([pd.cut(autosome_chrs.abs_pos, autosome_bins),pd.cut(autosome_chrs.pckmeans_cluster, [-0.1,0.9,2])]).mean().reindex(pd.MultiIndex.from_product([autosome_bins[1:], [0,1]]), fill_value = np.nan)
            x_chr_groups = x_chr.groupby([pd.cut(x_chr.abs_pos, x_bins),pd.cut(x_chr.pckmeans_cluster, [-0.5,0.5])]).mean().reindex( pd.MultiIndex.from_product([x_bins[1:], [0]]), fill_value = np.nan)
            y_chr_groups = y_chr.groupby([pd.cut(y_chr.abs_pos, y_bins),pd.cut(y_chr.pckmeans_cluster, [-0.5,0.5])]).mean().reindex(pd.MultiIndex.from_product([y_bins[1:], [0]]), fill_value = np.nan)
            groups = pd.concat([autosome_chr_groups,x_chr_groups, y_chr_groups], axis = 0)
        
        
            assert groups.shape[0] == 2 * autosome_num_bins + x_num_bins + y_num_bins
            return groups
    else:
        raise ValueError
        print("please indicate whether sex chromosomes should be included or not")


def get_inter_cell_dist(m0,m1):
    
    n = m0.shape[0]
    
    k = 1 #we don't want to include the diagonal for the 38x38 representations!!!! b/c the 0s on the diagonal artifically raise the correlation value!
    ut_ind = np.triu_indices(n, k)
    assert ut_ind[0].shape[0] == n*(n-1)/2
    
    
    m0_unrav = m0[ut_ind] #len is n*(n+1)/2
    m1_unrav = m1[ut_ind]
  
    #find indices where both unraveled matrices are not nan
    filt = (np.isnan(m0_unrav)+np.isnan(m1_unrav))==0
    
    #reduce the matrices to only indices that are not nan for both
    m0_filt = m0_unrav[filt]
    m1_filt = m1_unrav[filt]
    
    #if the two matrices share one or no indices that are not nan, return nan. Otherwise, findn the pearson correlation.
    if sum(~np.isnan(m0_filt))<=1:
        r=np.nan
    else:
        #get pearson's r
        r = sp.stats.pearsonr(m0_filt,m1_filt)[0]
   
    return 1 - r, np.sum(filt) #r is the correlation, len(filt) is the size of the intersection









 
"""wrapper (utility) function. using this to do data parallelism"""
def align_cell_i(cell_id_i, bin_size):
#     random_state = 500
    num_samples = 50
    
    print("aligning cell {}".format(cell_id_i))
    bins, num_bins_per_chr = get_bins(bin_size, cum_lens, num_chrs)
    cell_i = data.loc[(data.cell_index==cell_id_i) & (data.chr <= num_chrs)].copy()
        #encodes the absolute position of the reads along the linear genome--> used for binning
    cell_i['abs_pos'] = -1
    cell_i['abs_pos'] = cell_i.pos.copy() + [cum_lens[ch-1] for ch in cell_i.chr] 
    
    
    
    for random_seed in np.arange(0,1e6, 40): #trying 40 different random seeds and picking the minimum
        cell_i_dist_mat, _ = pckmeans_get_dist_mat_binned_resample(cell_i, bins, num_bins_per_chr, random_seed)

        cell_i_dists = []
        cell_i_intersection_sizes = []

        cids_after_i = data.loc[data.cell_index >= cell_id_i, 'cell_index'].unique()
        for cell_id_j in cids_after_i:

            cell_j = data.loc[(data.cell_index==cell_id_j) &  (data.chr <= num_chrs)].copy()
            cell_j['abs_pos'] = -1
            cell_j['abs_pos'] = cell_j.pos.copy() + [cum_lens[ch-1] for ch in cell_j.chr] 
            cell_j_dist_mat, _ = pckmeans_get_dist_mat_binned_resample(cell_j, bins, num_bins_per_chr, random_seed)



            cell_j_dists = []
            cell_j_intersection_sizes = []
            for sample in range(num_samples): #in order to align cell j with cell i, we run the sequential algorithm on 50 random sequences
                order = np.arange(1,20)
                np.random.shuffle(order)
                #bit_seq is something like x = [0,1,1,1,0,...] of length 19 where x[i]=0 means that in cell j we don't swap the copies of chromosome i. #bin_seq is something like [23,24,12,11,...] which has the actual sequnce of the aligned bins
                (dist, intersection_size), bit_seq, bin_seq, _ = get_aligned_inter_cell_dist(cell_i_dist_mat, cell_j_dist_mat, num_bins_per_chr, chr_seq = order) #np.arange(19,0,-1)
                cell_j_dists.append(dist)
                cell_j_intersection_sizes.append(intersection_size)


            
        cell_i_dists.append(np.min(cell_j_dists))
        cell_i_intersection_sizes.append(cell_j_intersection_sizes[np.argmin(cell_j_dists)])
        
        
    np.save("data/temp/aligned_dist_{}_bin_size_{}_numchrs_{}_cell{}.npy".format(reads_to_include, int(bin_size/1e6), num_chrs, cell_id_i),np.array(cell_i_dists))
    np.save("data/temp/aligned_dist_{}_intersection_size_bin_size_{}_numchrs_{}_cell{}.npy".format(reads_to_include, int(bin_size/1e6), num_chrs, cell_id_i),np.array(cell_i_intersection_sizes))

    return 
    
    
def read_data(clustering_method, reads_to_inlcude):
    if clustering_method == "igs":
        data = pd.read_csv('data/embryo_data.csv')
        data = data.loc[~data.cell_index.isin([ 80.,  84., 105., 113.])] #getting rid of cells with less than 150 reads
        if reads_to_inlcude == "inliers":
            data = data.loc[data.inlier == 1] 
    elif clustering_method == "pckmeans":
        data = pd.read_csv('data/pckmeans_embryo_data.csv')
        data = data.loc[~data.cell_index.isin([ 80.,  84., 105., 113.])] 
        if reads_to_inlcude == "inliers":
            data = data.loc[data.outlier == 0]
    return data
    

    
#the order of chromosomes to consider is 0,1,2,3...
"""
finds the best chromosome alignment sequentially, for now considering the chromosomes in the order chr_seq
num_bins_per_chr: dictionary holding the number or bins for each chromosome (first element is 0:0) 
{0: 0,
 1: 2,
 2: 2,
 3: 2,
 4: 2,
 5: 2,...}
num_chrs: the number of chromosomes to align
assumes the distance matrices to have the following order:
chr 1 cluster 0 bin 1
chr 1 cluster 0 bin 2 
chr 1 cluster 1 bin 1
chr 1 cluster 1 bin 2 
...
chr 19 cluster 0 bin 1
chr 19 cluster 0 bin 2 
chr 19 cluster 1 bin 1
chr 19 cluster 1 bin 2 
"""
def get_aligned_inter_cell_dist(cell_i_dist, cell_j_dist, num_bins_per_chr, num_chrs= 19,  
                                chr_seq = None, visualize = False):
    
    
    
    if chr_seq is None: 
        print("default chromosome sequence")
        chr_seq = np.arange(1,20)
    
    
    if visualize: fig, axes = plt.subplots(num_chrs,2, figsize = (7,15))
    
    total_haploid_bins = np.sum([val for key,val in num_bins_per_chr.items()][:num_chrs+1])  #total number of bins for the first num_chrschromosomes
    cum_num_bins = np.cumsum([val for key,val in num_bins_per_chr.items()]) #[0,bins_chr1, bins_chr1+chr2,...] HAPLOID number of bins
 
    cell_i_seq = []
    cell_j_seq = [] 
    bit_wise_seq = {}  # i: 0 --> chromosome i hasn't been switched, 1 means it has been switched
    for i in chr_seq:
        
        if visualize:
            sns.heatmap(cell_i_dist_subset, square = True, ax = axes[i,0], vmin = 0, vmax = 22, cbar = False)
            sns.heatmap(cell_j_dist_subset, square = True, ax = axes[i,1], vmin = 0, vmax = 22, cbar = False)
        
        cell_i_seq = cell_i_seq + list(np.arange(2*cum_num_bins[i-1], 2*cum_num_bins[i-1] + 2*num_bins_per_chr[i])) #this is the default sequence where we don't touch the order of copies
        seq1 = cell_j_seq + list(np.arange(2*cum_num_bins[i-1], 2*cum_num_bins[i-1] + 2*num_bins_per_chr[i])) 
        seq2 = cell_j_seq + list(np.arange(2*cum_num_bins[i-1] + num_bins_per_chr[i], 2*cum_num_bins[i-1] + 2*num_bins_per_chr[i])) +\
                           list(np.arange(2*cum_num_bins[i-1] , 2*cum_num_bins[i-1] + num_bins_per_chr[i]))
        
        dist1, inter_size1 = get_inter_cell_dist(cell_i_dist[np.ix_(cell_i_seq, cell_i_seq)], cell_j_dist[np.ix_(seq1, seq1)])
        dist2, inter_size2 = get_inter_cell_dist(cell_i_dist[np.ix_(cell_i_seq, cell_i_seq)], cell_j_dist[np.ix_(seq2, seq2)])
        
#         print(seq1, seq2)
        
        if dist1 <= dist2:
            bit_wise_seq[i] = 0
            cell_j_seq = seq1
        elif dist2 < dist1:
            bit_wise_seq[i] = 1
            cell_j_seq = seq2
        else: #dists will be nan when we only have one value in each distance matrix
            cell_j_seq = seq1
            bit_wise_seq[i] = 0
    bit_wise_seq_list = [bit_wise_seq[i] for i in np.arange(1, 20)]
    return get_inter_cell_dist(cell_i_dist[np.ix_(cell_i_seq, cell_i_seq)], cell_j_dist[np.ix_(cell_j_seq, cell_j_seq)]), bit_wise_seq_list, cell_j_seq, cell_i_seq ##############EXTEA OUTPUT
    
    

def main():
    global cum_lens
    global num_chrs
    global data
    global reads_to_include
    
        
    num_chrs = 19
    cum_lens = get_chr_cumulative_lengths()
   

    clustering_method = "pckmeans"
    reads_to_include = "inliers"
    
    
    print("clustering method: ", clustering_method)
    print("including {} reads".format(reads_to_include))
    data = read_data(clustering_method, reads_to_include) #global variables
    data = data.loc[data.stage == "4cell"]
    
    cids_4cell = data.cell_index.unique()
    
    
    for bin_size in [10e6, 5e6]:#200e6, 100e6, 50e6, 
        print("bin size: {}, Number of chromosomes: {}".format(int(bin_size/1e6), num_chrs))
        with Pool(6) as p:
            p.starmap(align_cell_i, zip(cids_4cell, repeat(bin_size)))


    
def consistency_analysis():
    reads_to_inlcude = "inliers" #"all"
    clustering_method = "pckmeans" # "igs"
    num_chrs = 19

    data = read_data(clustering_method, reads_to_inlcude) #cells with less than 150 reads are deleted: 80.,  84., 105., 113.

    cum_lens = get_chr_cumulative_lengths()

    fig, axes = plt.subplots(4,4, figsize = (20,20)) 
    for i, bin_size in tqdm(enumerate([200e6, 100e6, 50e6, 25e6])):
        for j, num_samples in tqdm(enumerate([5, 10, 25, 50])):
            print("\n bin size: ", bin_size)
            print("\n num samples: ", num_samples)
            
            proportion_matching = []
            variances = []
            cell_i_index = 74
            cell_j_index = 75
#             for cell_i_index in tqdm(data.loc[data.stage == '4cell', 'cell_index'].unique()[0:2]):
#                 cids_after_i = data.loc[data.cell_index >= cell_i_index, 'cell_index'].unique()[1:3]
#                 for cell_j_index in cids_after_i:

            cell_i = data.loc[(data.cell_index==cell_i_index) & (data.chr < 20)].copy()
            cell_i['abs_pos'] = -1
            cell_i['abs_pos'] = cell_i.pos.copy() + [cum_lens[ch-1] for ch in cell_i.chr] #encodes the absolute position of the reads along the linear genome
            cell_j = data.loc[(data.cell_index==cell_j_index) & (data.chr < 20)].copy()
            cell_j['abs_pos'] = -1
            cell_j['abs_pos'] = cell_j.pos.copy() + [cum_lens[ch-1] for ch in cell_j.chr] #encodes the absolute position of the reads along the linear genome


            bins, num_bins_per_chr = get_bins(bin_size, cum_lens, num_chrs)

            cell_i_dist = pckmeans_get_dist_mat_binned(cell_i, bins, num_bins_per_chr)
            cell_j_dist = pckmeans_get_dist_mat_binned(cell_j, bins, num_bins_per_chr)
    #         print("intra cell distance matrix shape: ", cell_i_dist.shape)




            min_dists = []
            num_trials = 75
            for trial in range(num_trials):
                dists = []
                for sample in range(num_samples):
                    if sample == 0:
                        order = np.arange(1,20)
                    elif sample == 1:
                        order = np.arange(19,0,-1)
                    else:
                        order = np.arange(1,20)
                        np.random.shuffle(order)
                    d, bit_seq, bin_seq = get_aligned_inter_cell_dist(cell_i_dist, cell_j_dist, num_bins_per_chr, chr_seq = order) #np.arange(19,0,-1)
                    dists.append(d[0])
                min_dists.append(np.round(np.min(dists), 4))

#                     proportion_matching.append(np.mean(dists < np.min(dists) +0.05))
#                     variances.append(np.var(dists))

            print(min_dists)
            axes[j,i].hist(min_dists, bins = 8)
            axes[j,i].set_title("bin size {}".format(bin_size/1e6))
            axes[j,i].set_ylabel("sample size: {}".format(num_samples))
#             axes[1,i].hist(variances, bins = 20)
#             axes[1,i].set_xlabel("variances")
    plt.suptitle("cell indeces {} and {}".format(cell_i_index, cell_j_index))
    plt.savefig("figures/sequential_algorithm_consistency_min_distance_distribution_cells{}_{}.png".format(cell_i_index, cell_j_index))


if __name__ == "__main__":
    main()
#     consistency_analysis()

 
