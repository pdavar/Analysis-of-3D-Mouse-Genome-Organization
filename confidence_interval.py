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
    
"""finds the within-cluster sum-of-squares for each kind of snp"""
def get_snp_ss_and_sil(cell):
    cell = cell.loc[cell.chr < 22]
    mat = np.array(cell.loc[cell.hap1_reads > cell.hap2_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    pat = np.array(cell.loc[cell.hap2_reads > cell.hap1_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
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

def snp_nn(num_neighbors, cell):
    cell = cell.loc[cell.chr < 22]
    mat = np.array(cell.loc[cell.hap1_reads > cell.hap2_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    pat = np.array(cell.loc[cell.hap2_reads > cell.hap1_reads, ['x_um_abs', 'y_um_abs', 'z_um_abs']])
    mat_or_pat = cell.loc[(cell.hap2_reads > cell.hap1_reads) | (cell.hap1_reads > cell.hap2_reads), ['x_um_abs', 'y_um_abs', 'z_um_abs', 'hap1_reads', 'hap2_reads']]
    
    X = np.array(mat_or_pat[['x_um_abs', 'y_um_abs', 'z_um_abs']])
    num_snps = X.shape[0]
    
    num_neighbors = min(num_neighbors+1, num_snps)
    nbrs = NearestNeighbors(n_neighbors=num_neighbors ,  algorithm='ball_tree').fit(X) #+1 b/c the read itself will be included as a neighbor
    
    # shape is N x num_neighbors where N is the number of SNPs
    distances, indices = nbrs.kneighbors(X) ## WARNING indices doesn't necessarily match the index col of the table, so don't use df.loc
    fractions = [snp_proportion_matching(mat_or_pat.iloc[indices[i]]) for i in range(num_snps)]
    
    return fractions

#get the proporiton of SNPs that match the first row
def snp_proportion_matching(snp_neighbors):
    if snp_neighbors.iloc[0, -1] > 0: #hap2_reads > 0 --> paternal
        return (snp_neighbors.loc[snp_neighbors.hap2_reads>0].shape[0]-1) /  (snp_neighbors.shape[0]-1)
    else: # -1 b/c we don't want to include the read itself
        return (snp_neighbors.loc[snp_neighbors.hap1_reads>0].shape[0]-1 )/  (snp_neighbors.shape[0]-1)

    
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
    
    
def visualize_snps(cell):
    # creating figure
    fig = plt.figure(figsize = (5,5))
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    
    cell = cell.loc[cell.chr < 22]
    mat = np.array(cell.loc[cell.hap1_reads > cell.hap2_reads])
    pat = np.array(cell.loc[cell.hap2_reads > cell.hap1_reads])
   
    ax.scatter(cell.x_um_abs, cell.y_um_abs, cell.z_um_abs, c = 'gray', s = 10, alpha = 0.2, cmap = 'tab20')
    ax.scatter(mat.x_um_abs, mat.y_um_abs, mat.z_um_abs, c = 'red', s = 10, alpha = 1, cmap = 'tab20')
    ax.scatter(pat.x_um_abs, pat.y_um_abs, pat.z_um_abs, c = 'blue', s = 10, alpha = 1, cmap = 'tab20')

    # setting title and labels
    ax.set_title("cell {}".format(str(cell.cell_index.unique()[0])))
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')

    # displaying the plot
    plt.show()
    
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
def get_error_bar(cell, n_samples, num_neighbors):
    
    randomized_runs = np.zeros((5,n_samples))
    for i in range(n_samples):
        randomized_cell = get_randomized_snps_for_cell(cell)

        randomized_runs[0,i] = get_snp_ss_and_sil(randomized_cell)[0]
        randomized_runs[1,i] = get_snp_ss_and_sil(randomized_cell)[3]
        randomized_runs[2,i] = np.mean(snp_nn(num_neighbors,randomized_cell))
        randomized_runs[3,i] = acc_of_logreg(randomized_cell)
        randomized_runs[4,i] = bhattacharyya_dist_in_cell(randomized_cell)
    return np.mean(randomized_runs, axis = 1), np.std(randomized_runs, axis = 1)

    
    
    
    
def plot_all_cells(data):
        #all cells
    fig, axes = plt.subplots(5,1, figsize = (30,25))

    num_neighbors = 5


    pos = 0 # used for cells that are too close together
    for i, cid in tqdm(enumerate(data.cell_index.unique())): # .loc[data.stage == '4cell', 'cell_index']

        cell = data.loc[(data.cell_index==cid)].copy()
        emb_id = cell.embryo_id.unique()[0]
        color = emb_id


        snp_ss, num_snp_mat, num_snp_pat, sil = get_snp_ss_and_sil(cell)
        lin_model_acc = acc_of_logreg(cell)
        snp_fraction_nn = np.mean(snp_nn(num_neighbors,cell))
        bha_dist = bhattacharyya_dist_in_cell(cell)


        if num_snp_mat < 4 or num_snp_pat < 4:
            color = 'magenta'

        axes[0].scatter(emb_id, snp_ss, c = color)
        axes[0].text(emb_id + 0.1, snp_ss+0.02 , str(cid)) #
        axes[0].text(emb_id + 0.05, snp_ss-0.03 , '[' + str(num_snp_mat + num_snp_pat) + ']', c = 'red')
        axes[0].set_xticks(np.arange(1, 45+14))
        axes[0].set_xlabel("embryo id")
        axes[0].set_ylabel("normalized sum of squares", fontsize = 15)

        axes[1].scatter(emb_id, sil, c = color)
        axes[1].text(emb_id + 0.1 , sil+0.02 , str(cid)) #
        axes[1].text(emb_id + 0.05 , sil-0.03 , '[' + str(num_snp_mat + num_snp_pat) + ']', c = 'red')
        axes[1].set_xticks(np.arange(1, 45+14))
        axes[1].set_xlabel("embryo id")
        axes[1].set_ylabel("silhouette", fontsize = 15)

        axes[2].scatter(emb_id, snp_fraction_nn, c = color)
        axes[2].text(emb_id + 0.1, snp_fraction_nn+0.02 , str(cid)) #
        axes[2].text(emb_id + 0.05, snp_fraction_nn-0.03 , '[' + str(num_snp_mat + num_snp_pat) + ']', c = 'red')
        axes[2].set_xticks(np.arange(1, 45+14))
        axes[2].set_xlabel("embryo id")
        axes[2].set_ylabel("{} nearest neighbor \n matching proportion".format(num_neighbors), fontsize = 15)


        axes[3].scatter(emb_id, lin_model_acc, c = color)
        axes[3].text(emb_id + 0.1 + 0.2 * pos, lin_model_acc+0.02 , str(cid)) #
        axes[3].text(emb_id + 0.05, lin_model_acc-0.03 , '[' + str(num_snp_mat + num_snp_pat) + ']', c = 'red')
        axes[3].set_xticks(np.arange(1, 45+14))
        axes[3].set_xlabel("embryo id")
        axes[3].set_ylabel("Linear classifier acc", fontsize = 15)

        axes[4].scatter(emb_id, bha_dist, c = color)
        axes[4].text(emb_id + 0.1 + 0.2 * pos, bha_dist+0.02 , str(cid)) #
        axes[4].text(emb_id + 0.05, bha_dist-0.05 , '[' + str(num_snp_mat + num_snp_pat) + ']', c = 'red')
        axes[4].set_xticks(np.arange(1, 45+14))
        axes[4].set_xlabel("embryo id")
        axes[4].set_ylabel("Bhattacharyya distance \n b/w SNP distributions", fontsize = 15)


        means, stds = get_error_bar(cell, n_samples = 100, num_neighbors=num_neighbors
                                   )
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


    plt.savefig("figures/genome_mixing_all_confidence_interval.png",bbox_inches='tight')
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
        
    plot_all_cells(data)
    
    
if __name__ == "__main__":
    main()
    