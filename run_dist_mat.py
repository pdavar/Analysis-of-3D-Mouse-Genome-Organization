import numpy as np
import pandas as pd
import scipy as sp
from scipy import spatial
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import collections

"""Attention: when running on IGS data vs. pckemans data
DEFINING DIFFERENT FUNCTIONS FOR IGS AND PCKMEANS TO ACCOUNT FOR THESE NAMING DIFFERENCES:
clusters [1,2] --> [0,1] (and also chance cl-1 to cl in assigning i)
cluster_hap_imputed --> pckmeans_cluster_hap
cluster --> pckmeans_cluster
"""

##Nan will corresppond to a missing chromosome, or trisomy chromosome, or when only one copy is available, etc...
def pckmeans_get_chr_average_dist_mat(cell_data_mat): #returns a 19 x 19 distance matrix: THE DIAGONALS WILL NOT BE 0
    centers = np.empty((19*2,3)) #row index i = 2*(chr_n-1)+(cluster)
    centers[:] = np.nan
    
    chr_nums = np.arange(1,20) #all autosome chromosome numbers [1,19]
   
    #make array to hold the xyz positions of the centers of the bins
    centers = np.empty((len(chr_nums)*2,3))
    centers[:] = np.nan
    
    #get bin centers
    for chr_n in chr_nums:
        chrom = cell_data_mat.loc[cell_data_mat.chr==chr_n]
        clusters = chrom.pckmeans_cluster.unique()
        
        #if the chromosome is not present in the data, continue
        if len(clusters)==0:continue
                
        if np.all(np.isin(chrom.pckmeans_cluster.unique(),[0,1])):
            for cl in [0,1]:
                i = 2*(chr_n-1)+cl
                cluster = chrom.loc[chrom.pckmeans_cluster==cl]
                centers[i,0] = cluster.x_um_abs.mean()
                centers[i,1] = cluster.y_um_abs.mean()
                centers[i,2] = cluster.z_um_abs.mean()
        elif np.all(np.isin(chrom.pckmeans_cluster.unique(),[-1])):
            cluster = chrom.loc[chrom.pckmeans_cluster==-1]
            i = 2*(chr_n-1)
            # we fill rows i and i+1 with the mean position of the chromosome
            centers[i:i+2,0] = cluster.x_um_abs.mean()
            centers[i:i+2,1] = cluster.y_um_abs.mean()
            centers[i:i+2,2] = cluster.z_um_abs.mean()
    
    #calculate matrix of pairwise distances beteween centers of chromosome territories
    m=sp.spatial.distance.cdist(centers,centers)
    
    #aggregate distances for both homologs to be haplotype-agnostic
    evens=np.arange(0,m.shape[0],2,dtype=int)
    odds=np.arange(1,m.shape[0],2,dtype=int)
    m=m[:,evens]+m[:,odds]
    m=m[evens,:]+m[odds,:]
    
    #double the diagonal, because off-diagonal entries have been created through the sum of four distances,
    #while the diagonals have been created through the sum of two distances and two zeros
    diag_i=np.diag_indices(m.shape[0])
    m[diag_i]=m[diag_i]*2
    
    #divide the matrix by 4, since 4 measurements have been added to produce each entry
    m=m/4
    
    return m


"""
bins is an array containing the bounds of each bin. 
It is constructed in such a way that each bin has a maximum of bin_size
base pairs, and each bin contains reads from no more than 1 chromosome.
returns a num_bins x num_bins matrix
This function is used in the chromosome_alignment.py script
rows of the returned data frame:
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
def pckmeans_get_dist_mat_binned(cell_data_mat, bins, num_bins_per_chr, sample_from_bin = 'mean'): #returns a 19 x 19 distance matrix: THE DIAGONALS WILL NOT BE 0
    num_bins = len(bins)- 1
    cum_lens = get_chr_cumulative_lengths()
    bin_to_chr = dict.fromkeys(bins[1:]) #dictionary mapping the bin values to chromosome. used to impute the chr value of the missing rows introduced after the groupby operation
    for bin_end in bins[1:]:
        bin_to_chr[int(bin_end)] = np.argmin(cum_lens < bin_end)
   
    
    cell_data_mat = cell_data_mat.loc[cell_data_mat.chr<20] #using only autosomes
    assert np.max(cell_data_mat.chr.unique()) < 20
    centers = np.empty((num_bins*2,3)) #row index i = 2*(chr_n-1)+(cluster)
    centers[:] = np.nan
    
    chr_nums = np.arange(1,20) #all autosome chromosome numbers [1,19]
   
    #make array to hold the xyz positions of the centers of the bins
    centers = np.empty((num_bins*2,3))
    centers[:] = np.nan
    
    ###multi-index dataframe: bin, cluster, chromosome
    groups = cell_data_mat.groupby([pd.cut(cell_data_mat.abs_pos, bins), pd.cut(cell_data_mat.pckmeans_cluster, [-0.1,0.9,2])])   
    
    if sample_from_bin == "mean":
        groups = groups.mean()
    elif sample_from_bin == 'first':
        groups = groups.nth(n = 0)
    elif sample_from_bin == "last":
        groups = groups.nth(n = -1)
    else:
        raise Exception('Specify the correct sampling method')
        
        
    groups = groups.reindex(pd.MultiIndex.from_product([ bins[1:], [0,1]], names = ['bin', 'cluster']), fill_value = np.nan)
    groups.chr.fillna(pd.Series(groups.index.get_level_values(0).map(bin_to_chr), groups.index), inplace = True) #this imputes the chr value of each missing value row based on the index value of the bin
    groups = groups.set_index(keys = "chr", append = True)
    groups.sort_index(level = ['chr', 'cluster'], inplace = True) #if we otherwise sort by chromosome without imputation, the rows with NAs will be thrown to either the end or the beginning, so it's crucial to impute the chr value of the missing rows
    assert np.all(np.array(groups.groupby('chr').size()) == 2*np.fromiter(num_bins_per_chr.values(), dtype=float)[1:])
    
    
    assert groups.shape[0] == centers.shape[0], print(groups.shape[0],centers.shape[0])
    centers[:,0] = groups.x_um_abs
    centers[:,1] = groups.y_um_abs
    centers[:,2] = groups.z_um_abs

    #rows are: bin1 cluster1, bin1 cluster2, bin2 cluster1, bin2 cluster 2, ...
    m=sp.spatial.distance.cdist(centers,centers)
    
    
    return m, groups

def pckmeans_get_dist_mat_binned_first(cell_data_mat, bins, num_bins_per_chr): #returns a 19 x 19 distance matrix: THE DIAGONALS WILL NOT BE 0
    num_bins = len(bins)- 1
    cum_lens = get_chr_cumulative_lengths()
    bin_to_chr = dict.fromkeys(bins[1:]) #dictionary mapping the bin values to chromosome. used to impute the chr value of the missing rows introduced after the groupby operation
    for bin_end in bins[1:]:
        bin_to_chr[int(bin_end)] = np.argmin(cum_lens < bin_end)
   
    
    cell_data_mat = cell_data_mat.loc[cell_data_mat.chr<20] #using only autosomes
    assert np.max(cell_data_mat.chr.unique()) < 20
    centers = np.empty((num_bins*2,3)) #row index i = 2*(chr_n-1)+(cluster)
    centers[:] = np.nan
    
    chr_nums = np.arange(1,20) #all autosome chromosome numbers [1,19]
   
    #make array to hold the xyz positions of the centers of the bins
    centers = np.empty((num_bins*2,3))
    centers[:] = np.nan
    
    ###multi-index dataframe: bin, cluster, chromosome
    groups = cell_data_mat.groupby([pd.cut(cell_data_mat.abs_pos, bins), pd.cut(cell_data_mat.pckmeans_cluster, [-0.1,0.9,2])]).nth(n = 0)                          
    groups = groups.reindex(pd.MultiIndex.from_product([ bins[1:], [0,1]], names = ['bin', 'cluster']), fill_value = np.nan)
    groups.chr.fillna(pd.Series(groups.index.get_level_values(0).map(bin_to_chr), groups.index), inplace = True) #this imputes the chr value of each missing value row based on the index value of the bin
    groups = groups.set_index(keys = "chr", append = True)
    groups.sort_index(level = ['chr', 'cluster'], inplace = True) #if we otherwise sort by chromosome without imputation, the rows with NAs will be thrown to either the end or the beginning, so it's crucial to impute the chr value of the missing rows
    assert np.all(np.array(groups.groupby('chr').size()) == 2*np.fromiter(num_bins_per_chr.values(), dtype=float)[1:])
    
    
    assert groups.shape[0] == centers.shape[0], print(groups.shape[0],centers.shape[0])
    centers[:,0] = groups.x_um_abs
    centers[:,1] = groups.y_um_abs
    centers[:,2] = groups.z_um_abs

    #rows are: bin1 cluster1, bin1 cluster2, bin2 cluster1, bin2 cluster 2, ...
    m=sp.spatial.distance.cdist(centers,centers)
    
    
    return m, groups


"""
bins is an array containing the bounds of each bin. 
It is constructed in such a way that each bin has a maximum of bin_size
base pairs, and each bin contains reads from no more than 1 chromosome.
returns a num_bins x num_bins matrix.
HERE, instread of taking the average position of each bin, we randomly sample from that bin. 
This function is used in the chromosome_alignment.py script
rows of the returned data frame:
chr 1 cluster 0 bin 1
chr 1 cluster 0 bin 2 
chr 1 cluster 1 bin 1
chr 1 cluster 1 bin 2 
...
chr 19 cluster 0 bin 1
chr 19 cluster 0 bin 2 
chr 19 cluster 1 bin 1
chr 19 cluster 1 bin 2 """
def pckmeans_get_dist_mat_binned_resample(cell_data_mat, bins,num_bins_per_chr, random_state = None): #returns a 19 x 19 distance matrix: THE DIAGONALS WILL NOT BE 0
    num_bins = len(bins)- 1
    cum_lens = get_chr_cumulative_lengths()
    bin_to_chr = dict.fromkeys(bins[1:]) #dictionary mapping the bin values to chromosome. used to impute the chr value of the missing rows introduced after the groupby operation
    for bin_end in bins[1:]:
        bin_to_chr[int(bin_end)] = np.argmin(cum_lens < bin_end)
  
    
    cell_data_mat = cell_data_mat.loc[cell_data_mat.chr<20] #using only autosomes
    assert np.max(cell_data_mat.chr.unique()) < 20
    centers = np.empty((num_bins*2,3)) #row index i = 2*(chr_n-1)+(cluster)
    centers[:] = np.nan
    
    chr_nums = np.arange(1,20) #all autosome chromosome numbers [1,19]
   
    #make array to hold the xyz positions of the centers of the bins
    centers = np.empty((num_bins*2,3))
    centers[:] = np.nan
    
    ###multi-index dataframe: bin, cluster, chromosome
    groups = cell_data_mat.groupby([pd.cut(cell_data_mat.abs_pos, bins), pd.cut(cell_data_mat.pckmeans_cluster, [-0.1,0.9,2])]).apply(lambda x: x.sample(1, random_state = random_state)).reset_index(level = 2, drop=True) #sampling introduces an arbitrary index level so we just remove that index level after sampling                           
    groups = groups.reindex(pd.MultiIndex.from_product([ bins[1:], [0,1]], names = ['bin', 'cluster']), fill_value = np.nan)
    groups.chr.fillna(pd.Series(groups.index.get_level_values(0).map(bin_to_chr), groups.index), inplace = True) #this imputes the chr value of each missing value row based on the index value of the bin
    groups = groups.set_index(keys = "chr", append = True)
    groups.sort_index(level = ['chr', 'cluster'], inplace = True) #if we otherwise sort by chromosome without imputation, the rows with NAs will be thrown to either the end or the beginning, so it's crucial to impute the chr value of the missing rows
    assert np.all(np.array(groups.groupby('chr').size()) == 2*np.fromiter(num_bins_per_chr.values(), dtype=float)[1:])
    
    
    assert groups.shape[0] == centers.shape[0], print(groups.shape[0],centers.shape[0])
    centers[:,0] = groups.x_um_abs
    centers[:,1] = groups.y_um_abs
    centers[:,2] = groups.z_um_abs

    #rows are: bin1 cluster1, bin1 cluster2, bin2 cluster1, bin2 cluster 2, ...
    m=sp.spatial.distance.cdist(centers,centers)
    
    
    return m, groups

"""
bins is an array containing the bounds of each bin. 
It is constructed in such a way that each bin has a maximum of bin_size
base pairs, and each bin contains reads from no more than 1 chromosome.
returns a num_bins x num_bins matrix"""
def pckmeans_get_chr_average_dist_mat_binned(cell_data_mat, bins): #returns a 19 x 19 distance matrix: THE DIAGONALS WILL NOT BE 0
    num_bins = len(bins)- 1
    
    cell_data_mat = cell_data_mat.loc[cell_data_mat.chr<20] #using only autosomes
    assert np.max(cell_data_mat.chr.unique()) < 20
    centers = np.empty((num_bins*2,3)) #row index i = 2*(chr_n-1)+(cluster)
    centers[:] = np.nan
    
    chr_nums = np.arange(1,20) #all autosome chromosome numbers [1,19]
   
    #make array to hold the xyz positions of the centers of the bins
    centers = np.empty((num_bins*2,3))
    centers[:] = np.nan
    
    ###multi-index dataframe: first idx = abs position bin, second idx = cluster
    groups = cell_data_mat.groupby([pd.cut(cell_data_mat.abs_pos, bins),pd.cut(cell_data_mat.pckmeans_cluster, [-0.1,0.9,2])]).mean().reindex(pd.MultiIndex.from_product([bins[1:], [0,1]]), fill_value = np.nan)

    assert groups.shape[0] == centers.shape[0], print(groups.shape[0],centers.shape[0])
    centers[:,0] = groups.x_um_abs
    centers[:,1] = groups.y_um_abs
    centers[:,2] = groups.z_um_abs

    m=sp.spatial.distance.cdist(centers,centers)
    
    #aggregate distances for both homologs to be haplotype-agnostic
    evens=np.arange(0,m.shape[0],2,dtype=int)
    odds=np.arange(1,m.shape[0],2,dtype=int)
    m=m[:,evens]+m[:,odds]
    m=m[evens,:]+m[odds,:]
    
    #double the diagonal, because off-diagonal entries have been created through the sum of four distances,
    #while the diagonals have been created through the sum of two distances and two zeros
    diag_i=np.diag_indices(m.shape[0])
    m[diag_i]=m[diag_i]*2
    
    #divide the matrix by 4, since 4 measurements have been added to produce each entry
    m=m/4
    return m




def pckmeans_get_dist_matrix(cell_data, method: str, bins = None, random_state = None):
    assert np.isin(method,['chr_average', 'chr_average_binned', 'chr_average_binned_resample'])
    if method == 'chr_average':
        return pckmeans_get_chr_average_dist_mat(cell_data) #get_bigger_chr_average()
    elif method == 'chr_average_binned':
        assert bins is not None
        return pckmeans_get_chr_average_dist_mat_binned(cell_data, bins)
    else:
        print("typo")

    
 ##Nan will corresppond to a missing chromosome, or trisomy chromosome, or when only one copy is available, etc...
def igs_get_chr_average_dist_mat(cell_data_mat): #returns a 19 x 19 distance matrix: THE DIAGONALS WILL NOT BE 0
    centers = np.empty((19*2,3)) #row index i = 2*(chr_n-1)+(cluster)
    centers[:] = np.nan
    
    chr_nums = np.arange(1,20) #all autosome chromosome numbers [1,19]
   
    #make array to hold the xyz positions of the centers of the bins
    centers = np.empty((len(chr_nums)*2,3))
    centers[:] = np.nan
    
    #get bin centers
    for chr_n in chr_nums:
        chrom = cell_data_mat.loc[cell_data_mat.chr==chr_n]
        clusters = chrom.cluster.unique()
        
        #if the chromosome is not present in the data, continue
        if len(clusters)==0:continue
                
        if np.all(np.isin(chrom.cluster.unique(),[2,1])):
            for cl in [2,1]:
                i = 2*(chr_n-1)+ cl - 1
                cluster = chrom.loc[chrom.cluster==cl]
                centers[i,0] = cluster.x_um_abs.mean()
                centers[i,1] = cluster.y_um_abs.mean()
                centers[i,2] = cluster.z_um_abs.mean()
        elif np.all(np.isin(chrom.cluster.unique(),[-1])):
            cluster = chrom.loc[chrom.cluster==-1]
            i = 2*(chr_n-1)
            # we fill rows i and i+1 with the mean position of the chromosome
            centers[i:i+2,0] = cluster.x_um_abs.mean()
            centers[i:i+2,1] = cluster.y_um_abs.mean()
            centers[i:i+2,2] = cluster.z_um_abs.mean()
    
    #calculate matrix of pairwise distances beteween centers of chromosome territories
    m=sp.spatial.distance.cdist(centers,centers)
    
    #aggregate distances for both homologs to be haplotype-agnostic
    evens=np.arange(0,m.shape[0],2,dtype=int)
    odds=np.arange(1,m.shape[0],2,dtype=int)
    m=m[:,evens]+m[:,odds]
    m=m[evens,:]+m[odds,:]
    
    #double the diagonal, because off-diagonal entries have been created through the sum of four distances,
    #while the diagonals have been created through the sum of two distances and two zeros
    diag_i=np.diag_indices(m.shape[0])
    m[diag_i]=m[diag_i]*2
    
    #divide the matrix by 4, since 4 measurements have been added to produce each entry
    m=m/4
    
    return m
"""
bins is an array containing the bounds of each bin. 
It is constructed in such a way that each bin has a maximum of bin_size
base pairs, and each bin contains reads from no more than 1 chromosome.
returns a num_bins x num_bins matrix"""
def igs_get_chr_average_dist_mat_binned(cell_data_mat, bins): #returns a 19 x 19 distance matrix: THE DIAGONALS WILL NOT BE 0
    num_bins = len(bins)- 1
    
    cell_data_mat = cell_data_mat.loc[cell_data_mat.chr<20] #using only autosomes
    assert np.max(cell_data_mat.chr.unique()) < 20
    centers = np.empty((num_bins*2,3)) #row index i = 2*(chr_n-1)+(cluster)
    centers[:] = np.nan
    
    chr_nums = np.arange(1,20) #all autosome chromosome numbers [1,19]
   
    #make array to hold the xyz positions of the centers of the bins
    centers = np.empty((num_bins*2,3))
    centers[:] = np.nan
    
    ###multi-index dataframe: first idx = abs position bin, second idx = cluster
    groups = cell_data_mat.groupby([pd.cut(cell_data_mat.abs_pos, bins),pd.cut(cell_data_mat.cluster, [0.9,1.1,2.1])]).mean().reindex(pd.MultiIndex.from_product([bins[1:], [1,2]]), fill_value = np.nan)

    assert groups.shape[0] == centers.shape[0], print(groups.shape[0],centers.shape[0])
    centers[:,0] = groups.x_um_abs
    centers[:,1] = groups.y_um_abs
    centers[:,2] = groups.z_um_abs

    m=sp.spatial.distance.cdist(centers,centers)
    
    #aggregate distances for both homologs to be haplotype-agnostic
    evens=np.arange(0,m.shape[0],2,dtype=int)
    odds=np.arange(1,m.shape[0],2,dtype=int)
    m=m[:,evens]+m[:,odds]
    m=m[evens,:]+m[odds,:]
    
    #double the diagonal, because off-diagonal entries have been created through the sum of four distances,
    #while the diagonals have been created through the sum of two distances and two zeros
    diag_i=np.diag_indices(m.shape[0])
    m[diag_i]=m[diag_i]*2
    
    #divide the matrix by 4, since 4 measurements have been added to produce each entry
    m=m/4
    return m

def visualize_dist_mat(dist_mat, 
                       inter_cell_vmax = 20,
                       inter_cell= False,
                       title = "",
                       fig_size = (5,5),
                      fig = None):
    # TO DO: set axes labels...
    if fig == None:
        fig = plt.figure()
    sns.set(rc = {'figure.figsize':fig_size})
    ax = sns.heatmap(dist_mat, vmin = 0, vmax = inter_cell_vmax)
    if inter_cell: 
        
        ax.vlines([24,64], *ax.get_xlim()) #drawing vertical lines separating the stages
        ax.hlines([24,64], *ax.get_xlim())
     
    plt.title(title)
    plt.show
    return 
    

    
"""turns a 19x19 distance matrix into a 38*38 by copying everything (so we get the same distance matrix just at a higher resolution"""
def get_bigger_chr_average(dist): #dist is a 19x19 intra cell distance matrix
    m = np.zeros((2*dist.shape[0], 2*dist.shape[0]))
    evens=np.arange(0,2*dist.shape[0],2,dtype=int)
    odds=np.arange(1,2*dist.shape[0],2,dtype=int)
    m[np.ix_(evens,evens)] = dist
    m[np.ix_(odds,odds)]= dist
    m[np.ix_(odds,evens)]= dist
    m[np.ix_(evens,odds)]= dist
    return m
    

def get_matrix_corr(m0,m1, include_diagonal, num_to_n_dict):
    
    n = m0.shape[0]
    
    #unravel the upper triangular, including the diagonal, of the two matrices 
    if include_diagonal ==True:
        k = 0
        ut_ind = np.triu_indices(n, k)
        assert ut_ind[0].shape[0] == n*(n+1)/2
    elif include_diagonal == False:
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
   
    return r, num_to_n_dict[np.sum(filt)] #r is the correlation, len(filt) is the size of the intersection

        
def igs_get_dist_matrix(cell_data, method: str, bins = None):
    assert np.isin(method,['chr_average', 'chr_average_binned'])
    if method == 'chr_average':
        return igs_get_chr_average_dist_mat(cell_data) #get_bigger_chr_average()
    elif method == 'chr_average_binned':
        assert bins is not None
        return igs_get_chr_average_dist_mat_binned(cell_data, bins)
    else:
        print("typo")
    
def get_dist_matrix(cell_data, clustering_method, embedding_method, bins):
    if clustering_method == 'igs':
        return igs_get_dist_matrix(cell_data, embedding_method, bins)
    elif clustering_method == 'pckmeans':
        return pckmeans_get_dist_matrix(cell_data, embedding_method, bins)
    

        

def get_inter_cell_corr_matrix(data: np.array ,
                               clustering_method: str, # 'igs', 'pckmeans'
                               method: str, # 'haplotype','homolog','chr_average', 'haplotype_and_chr_average'
                               bin_size #resolution at which we bin the reads along the genome
                              ):#returns a num_cells x num_cells matrix
    
    
    bins, num_bins_per_chr = get_bins(bin_size, cum_lens, num_chrs = 19)
    print("number of bins: ",len(bins)-1)
   
     #used for determining the number of missing values
    num_to_n_dict = {}
    
    if (method == "chr_average") or (method == "chr_average_binned"):
        include_diagonal = True
        for n in range(0,len(bins)):
            num_to_n_dict[n*(n+1)/2] = n
        print("including the diagonal of the intra-cell representation")
    else:
        include_diagonal = False
        for n in range(0,len(bins)):
            num_to_n_dict[n*(n-1)/2] = n
        print("NOT including the diagonal of the intra-cell representation")
    
    cell_indeces = data.cell_index.unique()
    num_cells = len(cell_indeces)
    # mapping from cell_id [1,112] to matrix index [0,108]
    cell_id_to_id = {}
    id_to_cell_id = {}
    cell_indeces = data.cell_index.unique()
    for i,cell_id in enumerate(cell_indeces):
        cell_id_to_id[cell_id] = i
        id_to_cell_id[i] = cell_id

    corrs_mat = np.empty((num_cells,num_cells))
    corrs_mat[:] = np.nan
    
    intersection_size_mat = np.empty((num_cells,num_cells))
    intersection_size_mat[:] = np.nan
    
    upper_triangular_ind = np.triu_indices(corrs_mat.shape[0])
    corrs_arr= np.array([])#a one dimensional array corresponding to the upper triangular of corrs_mat
    
    for i in range(num_cells):
        cell_i = data.loc[(data.cell_index==id_to_cell_id[i]) & (data.chr < 20)].copy()
        #because cell_index starts from 1
        cell_i['abs_pos'] = -1
        cell_i['abs_pos'] = cell_i.pos.copy() + [cum_lens[ch-1] for ch in cell_i.chr] #encodes the absolute position of the reads along the linear genome
        cell_i_dist_mat = get_dist_matrix(cell_i,clustering_method, method, bins)
        
        for j in range(i, num_cells):
            cell_j = data.loc[(data.cell_index==id_to_cell_id[j]) &  (data.chr < 20)].copy()
            cell_j['abs_pos'] = -1
            cell_j['abs_pos'] = cell_j.pos.copy() + [cum_lens[ch-1] for ch in cell_j.chr] 
            cell_j_dist_mat = get_dist_matrix(cell_j,clustering_method, method, bins)
            r, intersection_size = get_matrix_corr(cell_i_dist_mat,cell_j_dist_mat, include_diagonal, num_to_n_dict)
            corrs_arr = np.append(corrs_arr, r)
            intersection_size_mat[i,j] = intersection_size
    assert corrs_arr.shape[0] == num_cells * (num_cells+1)/2, print(corrs_arr.shape[0], num_cells)
    corrs_mat[upper_triangular_ind] = corrs_arr
    
#     corrs_mat = corrs_mat 

    return corrs_mat, intersection_size_mat
 
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
    
    
    
def get_bins(bin_size, cum_lens, num_chrs = 19):
    bins = [] # the base_pair absolute position cut-offs for forming the bins
    num_bins_per_chr = {0:0} # holds the number of bins for each chromosome
    cum_len_extended = np.append(cum_lens[:num_chrs+1].copy(), cum_lens[num_chrs]+bin_size)
    for ch in range(0,num_chrs+1):
        num_bins_per_chr[ch+1] = np.arange(cum_len_extended[ch],cum_len_extended[ch+1], bin_size).shape[0]
        bins.extend(np.arange(cum_len_extended[ch],cum_len_extended[ch+1], bin_size))
    num_bins = len(bins)- 1
    del num_bins_per_chr[num_chrs+1] #this number doesn't mean anything so we delete it
#     print("number of bins: ",2*num_bins)

    assert bins[-1] == cum_lens[num_chrs], "binning went wrong"
    return bins, num_bins_per_chr
    
def get_chr_cumulative_lengths():
    #building a dictionary that maps each chromosome number (int for autosomes, 'X' or 'Y' for allosomes) to the length
    chrom_size_dict = {0:0}
    with open("mm10.chrom.sizes") as f:
        for line in f:
            (k, v) = line.split()
            k = k[3:]
            try:
                k = int(k)
                chrom_size_dict[k] = int(v)
            except:
                if k=='X':
                    chrom_size_dict[20] = int(v)
                elif k=='Y':
                    chrom_size_dict[21] = int(v)
                else:
                    continue
                
    chrom_size_dict = collections.OrderedDict(sorted(chrom_size_dict.items()))
    lens = [val for key,val in chrom_size_dict.items()] #[0, len_chr1, len_chr2,...]
    cum_lens = np.cumsum(lens) #[0,len_chr1, len_chr1 + len_chr2,...]
    
    return cum_lens


    
    
def main():
    
    global cum_lens
       
    cum_lens = get_chr_cumulative_lengths() #[0,len_chr1, len_chr1 + len_chr2,...]
    
    
    
    for clustering_method in ["pckmeans"]: #pckmeans
        for reads_to_include in ["inliers", "all"]: #"inliers", 
            for method in [ "chr_average_binned" ]:#"haplotype", "haplotype_and_chr_average", "chr_average",
                for bin_size in [200e6, 100e6, 50e6, 25e6]:#
                    print(method,clustering_method,reads_to_include, bin_size)
                    data = read_data(clustering_method, reads_to_include)
                    inter_cell, intersection_size_mat = get_inter_cell_corr_matrix(data, clustering_method, method, bin_size)

                    np.save("data/inter_cell_corr_{}_{}_clusters_{}_bin_size_{}Mbp.npy".format(\
                              method,clustering_method,reads_to_include, int(bin_size/1e6)), inter_cell)
 
                    np.save("data/inter_cell_intersection_{}_{}_clusters_{}_bin_size_{}Mbp.npy".format(\
                              method,clustering_method,reads_to_include, int(bin_size/1e6)), intersection_size_mat)
        
if __name__ == "__main__":
    main()

 