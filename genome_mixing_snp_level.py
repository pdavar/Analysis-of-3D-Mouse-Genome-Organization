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
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.neighbors import NearestNeighbors
# from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score, silhouette_samples 
pd.set_option('display.max_columns', None)




def get_norm(vecs):
    return np.sqrt(np.sum(np.square(vecs), axis = 1)) #shape is n
    
"""finds the within-cluster sum-of-squares for each kind of snp"""
def get_snp_ss_and_sil(cell):
    cell = cell.loc[cell.chr < 22]
    mat = np.array(cell.loc[cell.hap1_reads > cell.hap2_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    pat = np.array(cell.loc[cell.hap2_reads > cell.hap1_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    mat_or_pat = np.vstack([mat,pat])
    

    mat_mean = np.mean(mat, axis = 0)
    pat_mean = np.mean(pat, axis = 0)
    mat_or_pat_mean = np.mean(mat_or_pat, axis = 0)

    mat_ss = np.mean(get_norm(mat - mat_mean))
    pat_ss = np.mean(get_norm(pat - pat_mean))
    mat_or_pat_ss = np.mean(get_norm(mat_or_pat - mat_or_pat_mean))

    try:
        silhouette = silhouette_score(np.vstack([mat,pat]), labels = np.append(np.ones(mat.shape[0]), np.zeros(pat.shape[0])) )
    except:
        silhouette = 0.5
    return np.nansum([mat_ss, pat_ss])/mat_or_pat_ss, len(mat), len(pat), silhouette

def snp_nn(num_neighbors, cell):
    cell = cell.loc[cell.chr < 22]
    mat = np.array(cell.loc[cell.hap1_reads > cell.hap2_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    pat = np.array(cell.loc[cell.hap2_reads > cell.hap1_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    mat_or_pat = cell.loc[(cell.hap2_reads > cell.hap1_reads) | (cell.hap1_reads > cell.hap2_reads), ['x_um_abs', 'y_um_abs', 'z_um_abs', 'hap1_reads', 'hap2_reads']]
    
    X = np.array(mat_or_pat[['x_um_abs', 'y_um_abs', 'z_um_abs']])
    num_snps = X.shape[0]
    
    num_neighbors = min(num_neighbors+1, num_snps)#+1 b/c the read itself will be included as a neighbor
    nbrs = NearestNeighbors(n_neighbors=num_neighbors ,  algorithm='ball_tree').fit(X) 
    
    # shape is N x num_neighbors where N is the number of SNPs
    _, indices = nbrs.kneighbors(X) ## WARNING indices doesn't necessarily match the index col of the table, so don't use df.loc
    fractions = [snp_proportion_matching(mat_or_pat.iloc[indices[i]]) for i in range(num_snps)]
    
    return fractions

# get the proporiton of SNPs that match the first row
def snp_proportion_matching(snp_neighbors):
    if snp_neighbors.iloc[0, -1] > 0: #hap2_reads > 0 --> paternal
        return (snp_neighbors.loc[snp_neighbors.hap2_reads>0].shape[0]-1) /  (snp_neighbors.shape[0]-1)
    else: # -1 b/c we don't want to include the read itself
        return (snp_neighbors.loc[snp_neighbors.hap1_reads>0].shape[0]-1 )/  (snp_neighbors.shape[0]-1)

    
def acc_of_linear_svm(cell, C = 1e5):
    
    cell = cell.loc[cell.chr < 22]
    mat = np.array(cell.loc[cell.hap1_reads > cell.hap2_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    pat = np.array(cell.loc[cell.hap2_reads > cell.hap1_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
   
    if mat.shape[0] <4  or pat.shape[0] < 4:
        return 1
    
    else:
        labels = np.append(np.ones(mat.shape[0]), np.zeros(pat.shape[0]))
        mat_or_pat = np.vstack([mat,pat])

        clf = SVC(C = C, kernel = 'linear')
        clf.fit(mat_or_pat, labels)
        return clf.score(mat_or_pat, labels)


def acc_of_logreg(cell):
    
    cell = cell.loc[cell.chr < 22]
    mat = np.array(cell.loc[cell.hap1_reads > cell.hap2_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    pat = np.array(cell.loc[cell.hap2_reads > cell.hap1_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
   
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
    mat = np.array(cell.loc[cell.hap1_reads > cell.hap2_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    pat = np.array(cell.loc[cell.hap2_reads > cell.hap1_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
   
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
    
   
    
    
    
#Given the reads with a SNP, create another copy of those reads where the snp assignments are randomized
def get_randomized_snps_for_cell(cell):
    pd.options.mode.chained_assignment = None 
    cell = cell.loc[cell.chr < 22]
    cell = cell.loc[(cell.hap2_reads > cell.hap1_reads) | (cell.hap1_reads > cell.hap2_reads), ['chr','x_um_abs', 'y_um_abs', 'z_um_abs', 'hap1_reads', 'hap2_reads']]
    num_mat = cell.loc[cell.hap1_reads > cell.hap2_reads].shape[0]
    num_pat = cell.loc[cell.hap2_reads > cell.hap1_reads].shape[0]
    
    
    copied_cell = cell[['chr','x_um_abs', 'y_um_abs', 'z_um_abs']].copy().reset_index()
    copied_cell['hap1_reads'] = 0
    copied_cell['hap2_reads'] = 0
    
    assert copied_cell.shape[0] == num_mat+num_pat
    mat_ind = np.random.choice(num_mat+num_pat, size = num_mat, replace = False)
    copied_cell.hap1_reads.iloc[mat_ind] = 1
    copied_cell.hap2_reads =  1 - copied_cell.hap1_reads
    
    assert copied_cell.loc[copied_cell.hap1_reads > 0].shape[0] == num_mat
    assert copied_cell.loc[copied_cell.hap2_reads > 0].shape[0] == num_pat
    assert copied_cell.shape[0] == cell.shape[0]
    
    return copied_cell


#drawing confidence intervals
def get_error_bar(cell, n_samples,num_neighbors):
    
    randomized_runs = np.zeros((5,n_samples))
    for i in range(n_samples):
        randomized_cell = get_randomized_snps_for_cell(cell)

        randomized_runs[0,i] = get_snp_ss_and_sil(randomized_cell)[0]
        randomized_runs[1,i] = get_snp_ss_and_sil(randomized_cell)[3]
        randomized_runs[2,i] = np.mean(snp_nn(num_neighbors,randomized_cell))
        randomized_runs[3,i] = acc_of_logreg(randomized_cell)
        randomized_runs[4,i] = bhattacharyya_dist_in_cell(randomized_cell)
    return np.mean(randomized_runs, axis = 1), np.std(randomized_runs, axis = 1)


def plot_4cell_cells(data):
    print("hello")
    cids = data.loc[data.stage == '4cell', 'cell_index'].unique()
    ids_embryo_change = np.array([cids[i] for i in range(len(cids)) if \
                          data.loc[data.cell_index == cids[i], 'embryo_id'].unique()\
                          != data.loc[data.cell_index == cids[i-1], 'embryo_id'].unique()]) #ids at which we have a new embryo 
    eids = data.loc[data.stage == '4cell', 'embryo_id'].unique()
    mean_id_in_embryo = np.array(data.loc[data.stage == '4cell'].groupby('embryo_id').mean()['cell_index'])
        #all cells
    
    fig, axes = plt.subplots(5,1, figsize = (30,25))

    num_neighbors = 5


    pos = 0 # used for cells that are too close together
    for i, cid in tqdm(enumerate(cids)): # .loc[data.stage == '4cell', 'cell_index']

        cell = data.loc[(data.cell_index==cid)].copy()
        emb_id = cell.embryo_id.unique()[0]
        color = emb_id


        chr_ss, num_snp_mat, num_snp_pat, sil = get_snp_ss_and_sil(cell.copy())
        n_snp = num_snp_mat + num_snp_pat 
        lin_model_acc = acc_of_logreg(cell)
        chr_fraction_nn = np.mean(snp_nn(num_neighbors, cell))
        bha_dist = bhattacharyya_dist_in_cell(cell)


        axes[0].scatter(cid, chr_ss, c = color)
        axes[0].text(cid, chr_ss+0.02,'[' + str(n_snp) + ']', c = 'red')
        axes[0].set_ylabel("normalized sum of squares", fontsize = 15)

        axes[1].scatter(cid, sil, c = color)
        axes[1].text(cid, sil+0.02,'[' + str(n_snp) + ']', c = 'red')
        axes[1].set_ylabel("silhouette", fontsize = 15)

        axes[2].scatter(cid, chr_fraction_nn, c = color)
        axes[2].text(cid, chr_fraction_nn+0.02,'[' + str(n_snp) + ']', c = 'red')
        axes[2].set_ylabel("{} nearest neighbor \n matching proportion".format(num_neighbors), fontsize = 15)


        axes[3].scatter(cid, lin_model_acc, c = color)
        axes[3].text(cid, lin_model_acc+0.02,'[' + str(n_snp) + ']', c = 'red')
        axes[3].set_ylabel("Linear classifier acc", fontsize = 15)

        axes[4].scatter(cid, bha_dist, c = color)
        axes[4].text(cid, bha_dist+0.02,'[' + str(n_snp) + ']', c = 'red')
        axes[4].set_ylabel("Bhattacharyya distance \n b/w chromatin distributions", fontsize = 15)


        means, stds = get_error_bar(cell, n_samples = 100, num_neighbors=num_neighbors)
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



    plt.savefig("figures/genome_mixing_snp_level_4cell_confidence_interval.png",bbox_inches='tight')
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

def save_mixing_levels(data):
    all_methods = {"Sum_of_Squares":[], "Silhouette":[], "Linear Classifier":[], "nearest neighbors":[], 'Bhattacharyya_dist':[]} #np.zeros((5,45)) #each row is a method
    num_snps = []

    for i, cid in enumerate(data.loc[data.stage == '4cell', 'cell_index'].unique()): #

        cell = data.loc[(data.cell_index==cid)].copy()
        emb_id = cell.embryo_id.unique()[0]


        chr_ss, num_snp_mat, num_snp_pat, sil = get_snp_ss_and_sil(cell)
        lin_model_acc = acc_of_logreg(cell)
        chr_fraction_nn = np.mean(snp_nn(5, cell))
        bha_dist = bhattacharyya_dist_in_cell(cell)

        all_methods['Sum_of_Squares'].append(chr_ss)
        all_methods["Silhouette"].append(sil)
        all_methods["Linear Classifier"].append(lin_model_acc)
        all_methods["nearest neighbors"].append(chr_fraction_nn)
        all_methods['Bhattacharyya_dist'].append(bha_dist)


        num_snps.append(num_snp_mat + num_snp_pat)
        
        
    with open('genome_mixing_metrics_dict_snp_level.pkl', 'wb') as f:
        pickle.dump(all_methods, f)
    np.save("genome_mixing_num_snps.npy", num_snps)

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
#     plot_4cell_cells(data)
    save_mixing_levels(data)
    
    
if __name__ == "__main__":
    main()

    