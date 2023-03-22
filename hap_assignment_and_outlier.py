import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from sklearn.mixture import GaussianMixture




def detect_outliers():
    embryos = pd.read_csv('data/pckmeans_embryo_data.csv')
    embryos['outlier'] = -1
    cell_ids = embryos.cell_index.unique()
    for cell_id in tqdm(cell_ids):
        for chr_id in range(1,22): #not doing it for allosomes
            chr_df = embryos.loc[(embryos.cell_index==cell_id)&(embryos.chr==chr_id)]
            for cl in chr_df.pckmeans_cluster.unique(): #doing it even for territories for which we don't have two clusters

                x = chr_df.loc[chr_df.pckmeans_cluster == cl]
                x = np.array(x[['x_um_abs', 'y_um_abs', 'z_um_abs']])
                outliers = np.zeros(x.shape[0])
                try:
                    # first finding the nearest neighbor of each read to get the mean distances used by DBSCAN
                    nbrs = NearestNeighbors(n_neighbors=2,  algorithm='ball_tree').fit(x) #2, b/c it includes the point itself
                    distances, indices = nbrs.kneighbors(x)

                    eps = 2*np.mean(distances[:,1])
                    labels = DBSCAN(eps=eps, min_samples=2).fit(x).labels_
                    outliers[np.where(labels == -1)] = 1
                    embryos.loc[(embryos.cell_index==cell_id)&(embryos.chr==chr_id)&(embryos.pckmeans_cluster == cl), 'outlier'] = outliers
                except:
                    print("couldn't detect outliers in cell {}  chromosome {} cluster {}".format(cell_id, chr_id, cl))
                    continue
    embryos.to_csv('data/pckmeans_embryo_data.csv', index = False)
    
    
    
    
"""uses the fact that the paternal pronuclei is larger
returns a binary vector of length num_reads in a zygote, where 0 is maternal and 1 is paternal"""
def assign_parent_of_origin_to_zygote(zygote):
    assert zygote.stage.unique()[0] =='zygote', "cell _id is not a zygoe"
    X = np.array(zygote[['x_um_abs','y_um_abs','z_um_abs']])   
    gmm = GaussianMixture(n_components=2, random_state=0, covariance_type = 'spherical').fit(X)
    labels = gmm.predict(X) 
    if np.sum(labels==0) > np.sum(labels==1): 
        labels = 1- labels
    return labels

"""returns a vector of length num_reads_in_chr where 0 means maternal and 1 means paternal"""
def assign_parent_of_origin_to_chr_in_zygote(zygote, chr_num):
    labels = assign_parent_of_origin_to_zygote(zygote)
    chr_index = np.where(zygote.chr == chr_num)
    chr_parents = labels[chr_index]
    
    if np.mean(zygote.loc[zygote.chr == chr_num, 'pckmeans_cluster'] == chr_parents)>0.8:
        return zygote.loc[zygote.chr == chr_num, 'pckmeans_cluster']
    else: 
        return 1 - zygote.loc[zygote.chr == chr_num, 'pckmeans_cluster']
    
        
    
    
    
def assign_parent_of_origin():
    data = pd.read_csv('data/pckmeans_embryo_data.csv')
    parent_dict = {0: "mat", 1: "pat", -1: "unassigned"}
    data["pckmeans_cluster_hap"] = -1
    cell_ids = data.cell_index.unique()
    for cell_id in tqdm(cell_ids):
        cell = data.loc[data.cell_index == cell_id]
        chrs = cell.chr.unique()
        for chr_num in chrs:
            chr_df = cell.loc[cell.chr == chr_num]
            if chr_df.stage.unique()[0] == 'zygote':
                data.loc[(data.cell_index == cell_id)&(data.chr == chr_num), "pckmeans_cluster_hap"] = assign_parent_of_origin_to_chr_in_zygote(cell, chr_num)
            else:
                assignment = assign_parental_to_homologs(chr_df)
                if type(assignment)==int: #the case where we only had one cluster (sex chromosomes and cluster values of -1)
                    data.loc[(data.cell_index == cell_id)&(data.chr == chr_num), "pckmeans_cluster_hap"] = assignment
                else:
                    data.loc[(data.cell_index == cell_id)&(data.chr == chr_num)&(data.pckmeans_cluster == 0), "pckmeans_cluster_hap"] = assignment[0]
                    data.loc[(data.cell_index == cell_id)&(data.chr == chr_num)&(data.pckmeans_cluster == 1), "pckmeans_cluster_hap"] = assignment[1]

    data.to_csv('data/pckmeans_embryo_data.csv', index = False)
    
    
    
    
    
"""This function returns a binary tuple (cluster_0_assignment, cluster_1_assignment)
(0: maternal, 1: paternal)"""
def assign_parental_to_homologs(chromosome):
    num_clusters = len(chromosome.pckmeans_cluster.unique())
    
    ## handling special cases
    if num_clusters == 1 and chromosome.chr.unique()[0] == 20:
        return 0 #the x chromosome will have an assignment of 0 (meternal) in male embryos
    elif chromosome.chr.unique()[0] == 21:
        return 1  #the y chromosome will have an assignment of 1 (peternal)
    elif num_clusters == 1:
        return -1 #will not assign parent of origin to chromosomes that couldn't be clustered
    else:
        assert num_clusters==2
        cl_0_mat_sum = np.sum(chromosome.loc[chromosome.pckmeans_cluster==0, 'hap1_reads'])
        cl_0_pat_sum = np.sum(chromosome.loc[chromosome.pckmeans_cluster==0, 'hap2_reads'])
        cl_1_mat_sum = np.sum(chromosome.loc[chromosome.pckmeans_cluster==1, 'hap1_reads'])
        cl_1_pat_sum = np.sum(chromosome.loc[chromosome.pckmeans_cluster==1, 'hap2_reads'])
        
        if cl_0_mat_sum >  cl_0_pat_sum: # cluster 0 is more maternal than paternal
            if cl_1_mat_sum <=  cl_1_pat_sum: return 0,1
            else: # both clusters are more maternal than paternal
                if cl_0_mat_sum == 1 and cl_1_mat_sum > 1: return 1, 0
                elif cl_0_mat_sum >1 and cl_1_mat_sum ==1: return 0, 1
                else: return -1, -1
        elif cl_0_mat_sum <  cl_0_pat_sum: # cluster 0 is more paternal than maternal
            if cl_1_mat_sum >=  cl_1_pat_sum: return 1, 0
            else: # both clusters are more paternal than maternal
                if cl_0_pat_sum == 1 and cl_1_pat_sum > 1: return 0, 1
                elif cl_0_pat_sum >1 and cl_1_pat_sum ==1: return 1, 0
                else: return -1, -1
        elif cl_0_mat_sum ==  cl_0_pat_sum: # cluster 0 is equally maternal and paternal, we check if cluster 1 has a clear assignment
            if cl_1_mat_sum <  cl_1_pat_sum: return 0,1
            elif cl_1_mat_sum >  cl_1_pat_sum: return 1,0
            else: return -1, -1
        else:
            print("error: you didn't account for this case")
 
    
    
if __name__ == "__main__":
    print("\n detecting outliers ...")
    detect_outliers()
    print("\n assigning parent of origin to territories ...")
    assign_parent_of_origin()
        
    

    
    