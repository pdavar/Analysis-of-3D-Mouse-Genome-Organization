
"""This script is very similar to the genome_mixing_snp_level.py script
While the other file is based on SNPs only (no inference), this one uses clustering information and imputes 
the haplotype assignments whenever possible, then only uses the chromosomes that have been haplotyped. randomizing is also done at the chromosome level (not at the SNP level)"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import collections
import time
from sklearn.neighbors import NearestNeighbors
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import scipy as sp
from tqdm import tqdm
from sklearn.manifold import MDS
# from run_dist_mat import *
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.neighbors import NearestNeighbors
# from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score, silhouette_samples 
pd.set_option('display.max_columns', None)




#vecs is nxd
def get_norm(vecs):
    return np.sqrt(np.sum(np.square(vecs), axis = 1)) #shape is n
    
"""finds the within-cluster sum-of-squares for each kind of chromosome"""
def get_ss_and_sil(cell):
    cell = cell.loc[cell.chr < 20]
    cell = cell.loc[(cell.pckmeans_cluster_hap == 0) | (cell.pckmeans_cluster_hap == 1)]
    mat = np.array(cell.loc[cell.pckmeans_cluster_hap == 0, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    pat = np.array(cell.loc[cell.pckmeans_cluster_hap == 1, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    mat_or_pat = np.vstack([mat,pat])
    
    if len(mat) == 0 or len(pat) == 0:
        return 1,len(mat), len(pat), 0.5
    else:
        mat_mean = np.mean(mat, axis = 0)
        pat_mean = np.mean(pat, axis = 0)
        mat_or_pat_mean = np.mean(mat_or_pat, axis = 0)

        mat_ss = np.mean(get_norm(mat - mat_mean))
        pat_ss = np.mean(get_norm(pat - pat_mean))
        mat_or_pat_ss = np.mean(get_norm(mat_or_pat - mat_or_pat_mean))

        silhouette = silhouette_score(np.vstack([mat,pat]), labels = np.append(np.ones(mat.shape[0]), np.zeros(pat.shape[0])) )

        return (mat_ss+ pat_ss)/mat_or_pat_ss, len(mat), len(pat), silhouette

def chr_nn(cell, num_neighbors = 5):
    cell = cell.loc[cell.chr < 20]
    cell = cell.loc[(cell.pckmeans_cluster_hap == 0) | (cell.pckmeans_cluster_hap == 1)]
    mat = np.array(cell.loc[cell.pckmeans_cluster_hap == 0, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    pat = np.array(cell.loc[cell.pckmeans_cluster_hap == 1, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    mat_or_pat = cell[['x_um_abs', 'y_um_abs', 'z_um_abs', 'pckmeans_cluster_hap']]
    
    
    X = np.array(mat_or_pat[['x_um_abs', 'y_um_abs', 'z_um_abs']])
    num_chrs = X.shape[0]
    
    nbrs = NearestNeighbors(n_neighbors=num_neighbors+1,  algorithm='ball_tree').fit(X) #+1 b/c the read itself will be included as a neighbor
    
    # shape is N x num_neighbors where N is the number of SNPs
    distances, indices = nbrs.kneighbors(X) ## WARNING indices doesn't necessarily match the index col of the table, so don't use df.loc
    fractions = [chr_proportion_matching(mat_or_pat.iloc[indices[i]]) for i in range(num_chrs)]
    
    return fractions


#get the proporiton of SNPs that match the first row
def chr_proportion_matching(read_neighbors):
    if read_neighbors.iloc[0, -1] == 0: #pckmeans_cluster_hap == 0 --> maternal
        return (read_neighbors.loc[read_neighbors.pckmeans_cluster_hap==0].shape[0]-1) /  (read_neighbors.shape[0]-1)
    else: # -1 b/c we don't want to include the read itself
        return (read_neighbors.loc[read_neighbors.pckmeans_cluster_hap==1].shape[0]-1 )/  (read_neighbors.shape[0]-1)

    
    
def acc_of_logreg(cell):
    
    cell = cell.loc[cell.chr < 22]
    cell = cell.loc[(cell.pckmeans_cluster_hap == 0) | (cell.pckmeans_cluster_hap == 1)]
    mat = np.array(cell.loc[cell.pckmeans_cluster_hap == 0, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    pat = np.array(cell.loc[cell.pckmeans_cluster_hap == 1, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
   
    if mat.shape[0] <4  or pat.shape[0] < 4:
        return 1
    
    else:
        labels = np.append(np.ones(mat.shape[0]), np.zeros(pat.shape[0]))
        mat_or_pat = np.vstack([mat,pat])

        clf = LogisticRegression(random_state=0, penalty = 'none') #when there's no penalty, we don't need a regulaizer
        clf.fit(mat_or_pat, labels)

        return clf.score(mat_or_pat, labels)
    
    
    
def bhattacharyya_dist(mu1, mu2, sigma1, sigma2):
    sigma = (sigma1+sigma2)/2
    d = 1/8 * (mu1 - mu2).T @ np.linalg.pinv(sigma) @ (mu1 -  mu2) + 0.5 * np.log(np.linalg.det(sigma) / np.sqrt(np.linalg.det(sigma1) * np.linalg.det(sigma2)))
    return d



def bhattacharyya_dist_in_cell(cell):
    cell = cell.loc[cell.chr < 22]
    cell = cell.loc[(cell.pckmeans_cluster_hap == 0) | (cell.pckmeans_cluster_hap == 1)]
    mat = np.array(cell.loc[cell.pckmeans_cluster_hap == 0, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    pat = np.array(cell.loc[cell.pckmeans_cluster_hap == 1, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
   
    if mat.shape[0] <4  or pat.shape[0] < 4:
        return 1
    else:
        mu1 = np.mean(mat, axis = 0)
        mu2 = np.mean(pat, axis = 0)

        sigma1 = np.cov(mat.T)
        sigma2 = np.cov(pat.T)
        
        assert np.linalg.det(sigma1) >=0, print(cell.cell_index.unique())
        assert np.linalg.det(sigma2) >=0, print(cell.cell_index.unique())

        return bhattacharyya_dist(mu1, mu2, sigma1, sigma2)
    
    
    
def randomize_parental_assignment_of_chromosomes(cell):
    pd.options.mode.chained_assignment = None 
    cell = cell.loc[cell.chr < 20]
    cell = cell.loc[(cell.pckmeans_cluster_hap == 0) | (cell.pckmeans_cluster_hap == 1), ['chr','pckmeans_cluster_hap','x_um_abs', 'y_um_abs', 'z_um_abs']]

    
    territories_mat = cell.loc[cell.pckmeans_cluster_hap==0, ['chr', 'pckmeans_cluster_hap']].groupby('chr').mean()
    territories_pat = cell.loc[cell.pckmeans_cluster_hap==1, ['chr', 'pckmeans_cluster_hap']].groupby('chr').mean()

    territories_mat["randomized_hap_assignment"] = np.random.binomial(size = territories_mat.shape[0], n = 1, p = 0.5)
    territories_pat["randomized_hap_assignment"] = 1 - territories_mat.randomized_hap_assignment.copy()
    territories_all = pd.concat([territories_pat, territories_mat])
    

    
    randomized_cell = cell.merge(territories_all, on = ['chr', 'pckmeans_cluster_hap'], how ='left')
    del randomized_cell['pckmeans_cluster_hap']
    randomized_cell = randomized_cell.rename(columns = {'randomized_hap_assignment': 'pckmeans_cluster_hap' })
    
    return randomized_cell
    
    
#drawing confidence intervals
def get_error_bar(cell, n_samples, num_neighbors):
    
    randomized_runs = np.zeros((5,n_samples))
    for i in range(n_samples):
        randomized_cell = randomize_parental_assignment_of_chromosomes(cell)

        randomized_runs[0,i] = get_ss_and_sil(randomized_cell)[0]
        randomized_runs[1,i] = get_ss_and_sil(randomized_cell)[3]
        randomized_runs[2,i] = np.mean(chr_nn(randomized_cell, num_neighbors))
        randomized_runs[3,i] = acc_of_logreg(randomized_cell)
        randomized_runs[4,i] = bhattacharyya_dist_in_cell(randomized_cell)
    return np.mean(randomized_runs, axis = 1), np.std(randomized_runs, axis = 1)


def get_number_of_haplotyped_chromosomes(cell):
    cell = cell.loc[cell.chr < 20]
    cell = cell.loc[(cell.pckmeans_cluster_hap == 0) | (cell.pckmeans_cluster_hap == 1)]
    
    return cell.chr.nunique()
 
def plot_all_cells(data):
        #all cells
    fig, axes = plt.subplots(5,1, figsize = (30,25))

    num_neighbors = 10


    pos = 0 # used for cells that are too close together
    for i, cid in tqdm(enumerate(data.cell_index.unique())): # .loc[data.stage == '4cell', 'cell_index']

        cell = data.loc[(data.cell_index==cid)].copy()
        emb_id = cell.embryo_id.unique()[0]
        color = emb_id


        chr_ss, num_chr_mat, num_chr_pat, sil = get_ss_and_sil(cell)
        lin_model_acc = acc_of_logreg(cell)
        chr_fraction_nn = np.mean(chr_nn(cell, num_neighbors))
        bha_dist = bhattacharyya_dist_in_cell(cell)


        axes[0].scatter(emb_id, chr_ss, c = color)
        axes[0].text(emb_id + 0.1, chr_ss+0.02 , str(cid)) #
        axes[0].text(emb_id + 0.05, chr_ss-0.03 , '[' + str(num_chr_mat + num_chr_pat) + ']', c = 'red')
        axes[0].set_xticks(np.arange(1, 45+14))
        axes[0].set_xlabel("embryo id")
        axes[0].set_ylabel("normalized sum of squares", fontsize = 15)

        axes[1].scatter(emb_id, sil, c = color)
        axes[1].text(emb_id + 0.1 , sil+0.02 , str(cid)) #
        axes[1].text(emb_id + 0.05 , sil-0.03 , '[' + str(num_chr_mat + num_chr_pat) + ']', c = 'red')
        axes[1].set_xticks(np.arange(1, 45+14))
        axes[1].set_xlabel("embryo id")
        axes[1].set_ylabel("silhouette", fontsize = 15)

        axes[2].scatter(emb_id, chr_fraction_nn, c = color)
        axes[2].text(emb_id + 0.1, chr_fraction_nn+0.02 , str(cid)) #
        axes[2].text(emb_id + 0.05, chr_fraction_nn-0.03 , '[' + str(num_chr_mat + num_chr_pat) + ']', c = 'red')
        axes[2].set_xticks(np.arange(1, 45+14))
        axes[2].set_xlabel("embryo id")
        axes[2].set_ylabel("{} nearest neighbor \n matching proportion".format(num_neighbors), fontsize = 15)


        axes[3].scatter(emb_id, lin_model_acc, c = color)
        axes[3].text(emb_id + 0.1 + 0.2 * pos, lin_model_acc+0.02 , str(cid)) #
        axes[3].text(emb_id + 0.05, lin_model_acc-0.03 , '[' + str(num_chr_mat + num_chr_pat) + ']', c = 'red')
        axes[3].set_xticks(np.arange(1, 45+14))
        axes[3].set_xlabel("embryo id")
        axes[3].set_ylabel("Linear classifier acc", fontsize = 15)

        axes[4].scatter(emb_id, bha_dist, c = color)
        axes[4].text(emb_id + 0.1 + 0.2 * pos, bha_dist+0.02 , str(cid)) #
        axes[4].text(emb_id + 0.05, bha_dist-0.05 , '[' + str(num_chr_mat + num_chr_pat) + ']', c = 'red')
        axes[4].set_xticks(np.arange(1, 45+14))
        axes[4].set_xlabel("embryo id")
        axes[4].set_ylabel("Bhattacharyya distance \n b/w chromatin distributions", fontsize = 15)


        means, stds = get_error_bar(cell, n_samples = 50, num_neighbors=num_neighbors)
        axes[0].errorbar(emb_id, means[0], yerr = 2*stds[0],  c = 'green', alpha = 0.5, fmt = 'o', linewidth = 3)
        axes[1].errorbar(emb_id, means[1], yerr = 2*stds[1], c = 'green', alpha = 0.5, fmt = 'o', linewidth = 3)
        axes[2].errorbar(emb_id, means[2], yerr = 2*stds[2],  c = 'green', alpha = 0.5, fmt = 'o', linewidth = 3)
        axes[3].errorbar(emb_id, means[3], yerr = 2*stds[3], c = 'green', alpha = 0.5, fmt = 'o', linewidth = 3)
        axes[4].errorbar(emb_id, means[4], yerr = 2*stds[4], c = 'green', alpha = 0.5, fmt = 'o', linewidth = 3)




    axes[0].vlines([24,45],color = 'gray', *axes[0].get_ylim(), alpha = 0.5) #drawing vertical lines separating the stages
    axes[1].vlines([24,45],color = 'gray', *axes[1].get_ylim(), alpha = 0.5) #drawing vertical lines separating the stages
    axes[2].vlines([24,45],color = 'gray', *axes[2].get_ylim(), alpha = 0.5) #drawing vertical lines separating the stages
    axes[3].vlines([24,45],color = 'gray', *axes[3].get_ylim(), alpha = 0.5) #drawing vertical lines separating the stages
    axes[4].vlines([24,45],color = 'gray', *axes[4].get_ylim(), alpha = 0.5) #drawing vertical lines separating the stages

    axes[0].text(0.01, 0.9, "Cell index", c = 'black', transform=axes[0].transAxes, 
                           fontsize=14, verticalalignment='top')
    axes[0].text(0.01, 0.8, "Number of SNPs", c = 'red', transform=axes[0].transAxes, 
                           fontsize=14, verticalalignment='top')


    plt.savefig("figures/genome_mixing_chr_level_all_confidence_interval.png",bbox_inches='tight')
    plt.show()
    
    
    
def plot_4cell_cells(data):
    cids = data.loc[data.stage == '4cell', 'cell_index'].unique()
    ids_embryo_change = np.array([cids[i] for i in range(len(cids)) if \
                          data.loc[data.cell_index == cids[i], 'embryo_id'].unique()\
                          != data.loc[data.cell_index == cids[i-1], 'embryo_id'].unique()]) #ids at which we have a new embryo 
    eids = data.loc[data.stage == '4cell', 'embryo_id'].unique()
    mean_id_in_embryo = np.array(data.loc[data.stage == '4cell'].groupby('embryo_id').mean()['cell_index'])
        #all cells
    
    fig, axes = plt.subplots(5,1, figsize = (30,25))

    num_neighbors = 10


    pos = 0 # used for cells that are too close together
    for i, cid in tqdm(enumerate(cids)): # .loc[data.stage == '4cell', 'cell_index']

        cell = data.loc[(data.cell_index==cid)].copy()
        n_haplotyped = get_number_of_haplotyped_chromosomes(cell)
        emb_id = cell.embryo_id.unique()[0]
        color = emb_id


        chr_ss, num_chr_mat, num_chr_pat, sil = get_ss_and_sil(cell)
        lin_model_acc = acc_of_logreg(cell)
        chr_fraction_nn = np.mean(chr_nn(cell, num_neighbors))
        bha_dist = bhattacharyya_dist_in_cell(cell)


        axes[0].scatter(cid, chr_ss, c = color)
        axes[0].text(cid, chr_ss+0.02,'[' + str(n_haplotyped) + ']', c = 'red')
        axes[0].set_ylabel("normalized sum of squares", fontsize = 15)

        axes[1].scatter(cid, sil, c = color)
        axes[1].text(cid, sil+0.02,'[' + str(n_haplotyped) + ']', c = 'red')
        axes[1].set_ylabel("silhouette", fontsize = 15)

        axes[2].scatter(cid, chr_fraction_nn, c = color)
        axes[2].text(cid, chr_fraction_nn+0.02,'[' + str(n_haplotyped) + ']', c = 'red')
        axes[2].set_ylabel("{} nearest neighbor \n matching proportion".format(num_neighbors), fontsize = 15)


        axes[3].scatter(cid, lin_model_acc, c = color)
        axes[3].text(cid, lin_model_acc+0.02,'[' + str(n_haplotyped) + ']', c = 'red')
        axes[3].set_ylabel("Linear classifier acc", fontsize = 15)

        axes[4].scatter(cid, bha_dist, c = color)
        axes[4].text(cid, bha_dist+0.02,'[' + str(n_haplotyped) + ']', c = 'red')
        axes[4].set_ylabel("Bhattacharyya distance \n b/w chromatin distributions", fontsize = 15)


        means, stds = get_error_bar(cell, n_samples = 70, num_neighbors=num_neighbors)
        axes[0].errorbar(cid, means[0], yerr = 2*stds[0],  c = 'green', alpha = 0.5, fmt = 'o', linewidth = 3)
        axes[1].errorbar(cid, means[1], yerr = 2*stds[1], c = 'green', alpha = 0.5, fmt = 'o', linewidth = 3)
        axes[2].errorbar(cid, means[2], yerr = 2*stds[2],  c = 'green', alpha = 0.5, fmt = 'o', linewidth = 3)
        axes[3].errorbar(cid, means[3], yerr = 2*stds[3], c = 'green', alpha = 0.5, fmt = 'o', linewidth = 3)
        axes[4].errorbar(cid, means[4], yerr = 2*stds[4], c = 'green', alpha = 0.5, fmt = 'o', linewidth = 3)




    
    axes[0].text(0.01, 0.9, "Cell index", c = 'black', transform=axes[0].transAxes, 
                           fontsize=14, verticalalignment='top')
    axes[0].text(0.01, 0.8, "Number of SNPs", c = 'red', transform=axes[0].transAxes, 
                           fontsize=14, verticalalignment='top')
    
    for i in range(5):
        axes[i].vlines(ids_embryo_change-0.5,color = 'gray', *axes[i].get_ylim(), alpha = 0.5) #drawing vertical lines separating the stages
        axes[i].set_xticks(cids)
        axes[i].set_xlabel("cell id")
        for j in range(len(mean_id_in_embryo)):
            axes[i].text(mean_id_in_embryo[j], axes[i].get_ylim()[1], s = str(eids[j]), bbox=dict(facecolor='blue', alpha=0.2), fontsize = 12)



    plt.savefig("figures/genome_mixing_chr_level_4cell_confidence_interval.png",bbox_inches='tight')
    plt.show()


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


def main():
    reads_to_inlcude = "inliers" #"all"
    clustering_method = "pckmeans" # "igs"
    data = read_data(clustering_method, reads_to_inlcude) #cells with less than 150 reads are deleted: 80.,  84., 105., 113.

    cell_id_to_id = {}
    id_to_cell_id = {}
    cids = np.array(data.cell_index.unique())
    for i,cell_id in enumerate(cids):
        cell_id_to_id[cell_id] = i
        id_to_cell_id[i] = cell_id 
        
#     plot_all_cells(data)
    plot_4cell_cells(data)
    
    
if __name__ == "__main__":
    main()

    