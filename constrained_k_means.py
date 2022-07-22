import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import pandas as pd
import time
import pandas as pd
from sklearn.cluster import KMeans #used for comparison purposes
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import sys

class PCKMeans:
    def __init__(self,
                 K: int, # number of components excluding the chromosomes added for context(for us it's 2)
                 constraints: np.array([]), # upper triangluar constraints matrix
                 data: np.array([]), # Nxd array of data
                 chr_n: int, # the first chr_n rows of the data are being clustered. the latter part is for context (separation)
                 separation_penalty = 2, # how much we penalize cluster separation
                 radius_threshold = 2, #used for separation heuristic
                 verbose = True): 
        
        self.K = K
        self.constraints = constraints + constraints.T #to make it symmetric
        self.data = data
        self.n = data.shape[0]
        self.chr_n = chr_n
        self.max_iter = 40
        self.labels = -1 * np.ones((self.n))
        self.separation_penalty = separation_penalty #something to play around with
        self.radius_threshold = radius_threshold
        self.verbose = verbose
       
        self.cluster_centers = np.zeros((self.K, self.data.shape[1]))
        if verbose:
            print("  ---------  ")
            print("num_components = ", self.K )
            print("cluster separation penalty = ", self.separation_penalty)
            print("cluster separation radius threshold = ", self.radius_threshold)
            print("  ---------  ")
        self.labels[self.chr_n:] = 2 #so the labels of the chromosome of interest are 0 and 1 (for downstream convenience)
        
        
    def how_many_negative_constraints(self):
        return np.sum(self.constraints[:self.chr_n, :self.chr_n] < 0)
    """if there are negative constraints, initialize the means to be the reads belonging to the most negative constraints. otherwise, do random initialization a few times and pick the one with the least objective function value"""
    
    def initialize_means(self, seed):
        if self.how_many_negative_constraints()>2: #this corresponds to having more than 1 constraint (using 2 b/c it's now symmetric and we can only have an even number of negative constraints)
            np.random.seed(seed)
            #we perturb the constraints a bit in case we have a dominant noisy constraint that might lead to a poor initialization
            perturbed_constraints = self.constraints[:self.chr_n, :self.chr_n]+ np.random.normal(0,2,(self.chr_n, self.chr_n))
            (i,j) = np.unravel_index(perturbed_constraints.argmin(), (self.chr_n, self.chr_n)) 
            self.cluster_centers[[0,1],:] = self.data[(i,j),:] #+ np.random.normal(0,2,(3)) #in case the initialization is not good
            if self.verbose: 
                print("given negative constraints: initializing means with seed {}".format(seed))
                print(np.round(self.cluster_centers[[0,1],:],4))
        else:  #if we only have one negative constraint, ignore it and use random initialization
            np.random.seed(seed)
            ind = np.random.choice(self.chr_n, size = self.K)
            self.cluster_centers = self.data[ind,:] #+ np.random.normal(0,1,(self.K,3))
            if self.verbose: 
                print("No negative constraints: initializing means with seed {}".format(seed))
                print(np.round(self.cluster_centers[[0,1],:],4))
    
    def fit(self, seed = 0, id_of_interest = None, visualize = False):
        
        self.initialize_means(seed)
        self.labels[: self.chr_n] = -1   #re-initializing the labels for every run
        
        self.id_of_interest = id_of_interest
        # Repeat until convergence
        for iteration in tqdm(range(self.max_iter)):
            if visualize: self.make_plot(iteration)
            
            # Assign clusters
            self.assign_clusters()
            
            # Estimate means
            prev_cluster_centers = self.cluster_centers
            self.cluster_centers = self.get_cluster_centers()

            # Check for convergence
            difference = (prev_cluster_centers - self.cluster_centers)
            converged = np.allclose(difference, np.zeros(self.cluster_centers.shape), atol=1e-6, rtol=0)
#             print("difference: ", difference)
            if converged: 
                if self.verbose: print("converged")
                break
#         self.assign_clusters() #to make sure that dist_mat is the most up-to-date
        return self.labels
    
    
    
    def fit_and_get_total_loss(self, id_of_interest = None, visualize = False):
      
        num_runs = 3
        seeds = [0,100,1000]
        losses = np.zeros((num_runs))
        diff_run_labels = -1 * np.ones((num_runs, self.n ))
        for run in range(num_runs):
            seed = seeds[run]
            diff_run_labels[run,:] = self.fit(seed, id_of_interest = id_of_interest, \
                                              visualize = visualize).copy()
             
            for i in range(self.chr_n):
                losses[run] += self.loss(i, int(diff_run_labels[run,i]))
                
            #penalizing the clusterings where one cluster has less than or equal to 2 reads
            _, counts_labels = np.unique(diff_run_labels[run,:], return_counts=True)
            if np.any(counts_labels < 3) and self.chr_n>10: 
                losses[run] += 100 
                print("cluster too small")
        if self.verbose: print("losses of the different runs: ", np.round(losses,2))
        best_run = np.nanargmin(losses) #disregarding NANs. B/c if one of the models doesn't converge due to bad initialization, loss is NAN
        self.labels = diff_run_labels[best_run, :]
        self.cluster_centers = self.get_cluster_centers()
        _ = self.get_dist_mat_and_return_sorted_indeces()
        return self.labels
    
    
    """returns the cluster_assignments array where cluster_assignments[i] is a 
    value in [0,K] which is the cluster assignment of data[i]"""
    def assign_clusters(self):  # kxd
        self.get_dist_mat_and_return_sorted_indeces()
        for i in self.sorted_inds: # i want this order to be based on some distance from the means b/c intuitively, once i have a good initialization, the points that are near the mean can be clustered w/o regards for constraints. It's the far away points that need constraints to inform their clustering
            losses = []
            for k in range(self.K):
                losses.append(self.loss(i,k))
            self.labels[i] = np.argmin(losses)
        
    def get_dist_mat_and_return_sorted_indeces(self):
        #construct an nxk matrix that contains the distances b/w data point i and centroid k
        self.dist_mat = np.zeros((self.chr_n, self.K))
        for i in range(self.chr_n):
            for k in range(self.K):
                self.dist_mat[i,k] = self.dist(self.data[i], self.cluster_centers[k])
        min_dist_to_centroid = np.min(self.dist_mat, axis = 1) # array of len n
        self.sorted_inds = [i for _,i  in sorted(zip(min_dist_to_centroid, np.arange(0,self.chr_n)))] 
        return self.sorted_inds

        
    def dist(self, pt1, pt2):
        return np.linalg.norm(pt1-pt2)
   
    def get_cluster_centers(self):
        return np.array([self.data[self.labels == i].mean(axis=0) for i in range(self.K)])
    
    def correlation(self, vec1, vec2):#cosine similarity
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    """this function computes the cost of assigning data[i] to cluster k 
    given the constraints and the cluster assignment of other points"""
    def loss(self, i, k):
        loss =  1/2 * self.dist_mat[i,k] ** 2
#         print("distance loss: ", loss)
        
        #points that have not yet been clustered (label = -1) will not be considered in the loss
        
        do_not_links = np.where(self.constraints[i] < 0)[0] 
        do_links = np.where(self.constraints[i] > 0)[0]
        
        for j in do_not_links:
            if (self.labels[j] != -1) and (self.labels[j] == k): 
                loss = loss + abs(self.constraints[i,j]) ####MAKE SURE i<j#######
              
            
            #cluster separation penalty
            #we must make sure that we have i-----j----mu_k (i.e. j is in the middle) before we incur a loss
            i_j_vec = self.data[j] - self.data[i]
            i_mu_k_vec = self.cluster_centers[k] - self.data[i]
            i_j_pojected = np.dot(i_j_vec, i_mu_k_vec) / (np.linalg.norm(i_mu_k_vec)**2+1e-6) * i_mu_k_vec
            orthogonal_component = np.linalg.norm(i_j_vec - i_j_pojected)
            
            if np.linalg.norm(i_mu_k_vec) > np.linalg.norm(i_j_vec) and\
            self.correlation(i_j_vec, i_mu_k_vec) > 0 and\
            orthogonal_component < self.radius_threshold: 
                loss += self.separation_penalty / (self.dist_mat[i,k]+1e-6) #normalizing by the distance, b/c it's the density of non_chr reads that matters, and not the raw number.
#                 print("loss after separation: ", loss)
        for j in do_links:
            if (self.labels[j] != -1) and (self.labels[j] != k): 
                loss = loss + abs(self.constraints[i,j])
        
        if i in self.id_of_interest: print("i: {}, k: {}, dist: {}, loss: {}".format(i, k, 1/2 * self.dist_mat[i,k] ** 2, loss))
        return loss
    
    def make_plot(self, it):
        sns.scatterplot(x = self.data[:self.chr_n,0], y = self.data[:self.chr_n,1], c = self.labels[:self.chr_n])
        sns.scatterplot(x = self.data[self.chr_n:,0], y = self.data[self.chr_n:,1], color = 'gray', alpha = 0.8)
        sns.scatterplot(x = self.cluster_centers[:,0], y = self.cluster_centers[:,1], marker = "*", s = 200, color = 'red')
        plt.title("iteration {}".format(it))
        plt.show()
        
    def likelihood_correction(self):
        assert np.all([i in [0,1] for i in np.unique(self.labels[0:self.chr_n])])
        self.corrected_labels = self.labels.copy()
        for i in self.sorted_inds:
            current_label = int(self.corrected_labels[i])
            ind = (self.corrected_labels==current_label) &(np.arange(0,self.n) != i)
            mean = np.mean(self.data[ind], axis = 0)
            cov = np.cov(self.data[ind].T)
#             current_dist = np.sqrt((self.data[i] - mean).T @ np.linalg.pinv(cov) @ (self.data[i] - mean))
            current_pdf = multivariate_normal.pdf(self.data[i], mean=mean, cov=cov)
            
            
            alternate_label = 1 - current_label
            ind = (self.corrected_labels == alternate_label) 
            mean = np.mean(self.data[ind], axis = 0)
            cov = np.cov(self.data[ind].T)
#             alternate_dist = np.sqrt((self.data[i] - mean).T @ np.linalg.inv(cov) @ (self.data[i] - mean))
            alternate_pdf = multivariate_normal.pdf(self.data[i], mean=mean, cov=cov)
                                
            if alternate_pdf > 100 * current_pdf:
                self.corrected_labels[i] = alternate_label
                if self.verbose: print(np.round(alternate_pdf,8), np.round(current_pdf,8))
                print(i, ": switched label \n")
        return self.corrected_labels
        
     
            
        

            
          
     
    
    
"""1Kbp can correspond to 0.3 microns max
if the returned ration is greater than 1, then we have a violation and must set a negative constraint"""
def spatial_to_genomic_distance_ratio(read1, read2):
    spatial_dist = np.sqrt(np.sum((read1.loc[['x_um_abs','y_um_abs','z_um_abs']] - read2.loc[['x_um_abs','y_um_abs','z_um_abs']])**2)) 
    if read1.pos > read2.pos:
        genomic_dist = read1.pos - (read2.pos + read2.frag_len)
    elif read1.pos < read2.pos:
        genomic_dist = read2.pos - (read1.pos + read1.frag_len)
    else:
        genomic_dist = 0
    return spatial_dist / (0.3*(genomic_dist/1e3) + 1e-6)

"""crude implementation...needs more thought"""
def set_snp_constraint(read1, read2, angle_threshold = np.pi/6):
    read1_mat_pat = read1.loc[['hap1_reads', 'hap2_reads']]
    read2_mat_pat = read2.loc[['hap1_reads', 'hap2_reads']]
    
    read1_cos_theta = np.dot(read1_mat_pat, [1,0]) / np.linalg.norm(read1_mat_pat)
    read2_cos_theta = np.dot(read2_mat_pat, [1,0]) / np.linalg.norm(read2_mat_pat)
    
    diff_cos_theta = np.dot(read1_mat_pat, read2_mat_pat) / (np.linalg.norm(read1_mat_pat) * np.linalg.norm(read2_mat_pat) )
    
    #if the mat_pat vector of the reads are quite different, return a large negative value
    if diff_cos_theta < np.cos(angle_threshold): #=: if their angle is larger than 30 degrees
        return -1 * (1- diff_cos_theta)
    else: #the two vectors are within angle_threshold of each other
        if ((read1_cos_theta > np.cos(angle_threshold)) and (read2_cos_theta > np.cos(angle_threshold))) or\
            ((read1_cos_theta < np.cos(angle_threshold)) and (read2_cos_theta < np.cos(angle_threshold))):
            return diff_cos_theta
        else:
            return 0

"""Using GMM and SVD, it gives the direction of the normal_vec and its origin (means)"""
def gmm_for_zygote(cell_id):
    assert cell_id <= 24, "cell _id is not a zygoe"
    cell_df = pd.read_csv("data/embryo_data.csv")
    cell_df = cell_df.loc[cell_df.cell_index==cell_id]
    X = np.array(cell_df[['x_um_abs','y_um_abs','z_um_abs']])   
    gmm = GaussianMixture(n_components=2, random_state=0, covariance_type = 'spherical').fit(X)
    return gmm 


#b/c i want the reads with a low probability of belonging to either cluster to get a very low weight
def non_linear_transform(prob):
    if prob > 0.65: return -10
    else: return -(prob**2)
    
def get_constraint(read1, read2, is_zygote, gmm = None):
    if is_zygote: assert gmm is not None
    constraint = 0
    if read1.chr != read2.chr:
        constraint = -10
    elif is_zygote:
        read1_pos = np.array(read1[['x_um_abs','y_um_abs','z_um_abs']]).reshape(1,-1)
        read2_pos = np.array(read2[['x_um_abs','y_um_abs','z_um_abs']]).reshape(1,-1)
        if gmm.predict(read1_pos) != gmm.predict(read2_pos):
            read1_prob = np.max(gmm.predict_proba(read1_pos), axis = 1)[0]
            read2_prob = np.max(gmm.predict_proba(read2_pos), axis = 1)[0]
            constraint = non_linear_transform(np.min([read1_prob, read2_prob]))
    else:
        ratio = spatial_to_genomic_distance_ratio(read1, read2)
        if ratio > 1 :
            constraint = -5
        elif (read1.hap1_reads + read1.hap2_reads !=0) and  (read2.hap1_reads + read2.hap2_reads != 0):
            constraint =  10*set_snp_constraint(read1, read2)
#     print(constraint )
    return constraint 
       


"""given n reads in a chromosome, it returns an nxn upper triangular matrix"""
def get_constraint_matrix(chromosome: np.array([])):
    n = chromosome.shape[0]
    constraints = np.zeros((n, n))
    
    ut_ind = np.triu_indices(n, 1) # the upper triangular indices (above the diagonal)
    x_ind = ut_ind[0]
    y_ind = ut_ind[1]
    ind = [(x_ind[i], y_ind[i]) for i in range(x_ind.shape[0])] #[(0,0), (0,1), ...]
    
    is_zygote = chromosome.stage.unique()[0] == 'zygote'
    gmm = None
    if is_zygote:
        gmm = gmm_for_zygote(chromosome.cell_index.unique()[0])
    constraints[ut_ind] = [get_constraint(chromosome.iloc[i],chromosome.iloc[j], is_zygote, gmm) for (i,j) in ind]
    
    assert ~np.isnan(constraints).any()
    return constraints

def main(from_cell, to_cell):
    print("clustering from cell {} to cell {}".format(from_cell, to_cell))
    t1 = time.time()
    ##### running pckmeans on each chromosome on in each cell
    embryos = pd.read_csv("data/pckmeans_embryo_data.csv")#, usecols = ['embryo_id', 'cell_id', 'cell_index', 'stage', 'amp_ind', 'x_um_abs',
#        'y_um_abs', 'z_um_abs', 'chr','pos', 'rel_chr_pos', 'frag_len','hap1_reads', 'hap2_reads'])
    embryos = embryos.loc[(embryos.cell_index >= from_cell) & (embryos.cell_index <= to_cell)]
#     embryos['pckmeans_before_correction'] = -1
#     embryos['pckmeans_cluster'] = -1
    cell_indices = embryos.cell_index.unique()
    print(cell_indices)
    for cell_index in cell_indices:      
        cell_df = embryos.loc[embryos.cell_index == cell_index]
        chrs = cell_df.chr.unique()
        for chr_ in [20,21]: #chrs:   
            print("\n -----------------------------------------")
            print("Cell {}, Chromosome {}\n".format(cell_index, chr_))
            print("time passed: ", time.time() - t1)

            chr_index = np.where(cell_df.chr==chr_)
            not_chr_index = np.where(cell_df.chr!=chr_)
            np.random.seed(100) #used for the sampling of not_chr reads
            not_chr_index_sampled = np.random.choice(not_chr_index[0], size = max(2*len(chr_index[0]), 100))
            cell_chr_df = cell_df.iloc[chr_index]
            cell_not_chr_df = cell_df.iloc[not_chr_index_sampled]

            #this is the data that is getting clustered
            X=np.array(cell_chr_df[['x_um_abs','y_um_abs','z_um_abs']]) #this is the data that will be clustered into 2 clusters
            not_X =  np.array(cell_not_chr_df[['x_um_abs','y_um_abs','z_um_abs']])
            data = np.vstack((X,not_X))
            chr_n = X.shape[0]
            not_chr_n = not_X.shape[0]
            n = data.shape[0]

            # special case for the clustering of allosomes
            if chr_ > 21: continue 
            elif (chr_ == 21) & (chr_n > 1): n_components = 1; print("male embryo")
            elif (chr_ == 20) & (cell_df.loc[cell_df.chr == 21].shape[0]>1): n_components = 1;  print("male embryo")
            elif cell_chr_df.shape[0] < 3: continue
            else: n_components = 2


            if n_components == 1:
                embryos.loc[(embryos.cell_index == cell_index)&(embryos.chr==chr_), "pckmeans_cluster"] = 0
            else:
                if chr_n < 4: #if the chromosome has less than 4 reads, don't do clustering
                    embryos.loc[(embryos.cell_index == cell_index)&(embryos.chr==chr_), "pckmeans_cluster"] = -1
                    print("too few reads...no clustering")
                else:

                    try:
                        #positive constraints among the "cell_not_chr" reads, and negative constraint b/w different chromosomes
                        constraints = np.zeros((n,n))
                        constraints[-not_chr_n:, -not_chr_n:] = 10
                        constraints[:chr_n, -not_chr_n:] = -10
                        constraints[:chr_n, :chr_n] = get_constraint_matrix(cell_chr_df)


                        #run the clustering algorithm with the constraints
                        pckmeans = PCKMeans(2, constraints, data, chr_n = chr_n, separation_penalty=10, radius_threshold = 1)
                        labels = pckmeans.fit_and_get_total_loss([],visualize = False)
                    except:
                        print("\n  #### encountered clustering error in cell {} chromosome {} #### \n".format(cell_index, chr_))


                        #performing the variance correction if only the reads are not from a zygote (where we have the pronuclei separation
                    is_zygote = cell_df.stage.unique()[0] == 'zygote'
                    if is_zygote: 
                        embryos.loc[(embryos.cell_index == cell_index)&(embryos.chr==chr_), "pckmeans_cluster"] = labels[:chr_n]
                        np.save('data/backup/cell{}_chr{}.npy'.format(cell_index, chr_), labels[:chr_n])
                    else: 
                        embryos.loc[(embryos.cell_index == cell_index)&(embryos.chr==chr_), "pckmeans_before_correction"] = labels[:chr_n]
                        try:
                            corrected_labels = pckmeans.likelihood_correction()[:chr_n]
                            embryos.loc[(embryos.cell_index == cell_index)&(embryos.chr==chr_), "pckmeans_cluster"] = corrected_labels
                            np.save('data/backup/cell{}_chr{}.npy'.format(cell_index, chr_), corrected_labels)
                        except:
                            print("coulnd't perform variance correction") #because there are too few points in a cluster
                            embryos.loc[(embryos.cell_index == cell_index)&(embryos.chr==chr_), "pckmeans_cluster"] =labels[:chr_n]
                            np.save('data/backup/cell{}_chr{}.npy'.format(cell_index, chr_), labels[:chr_n])

                    

        embryos.to_csv("data/pckmeans_embryo_data_{}-{}.csv".format(from_cell, to_cell), index = False)


# python constrained_k_means.py 31 55
# bounds are both inclusive
if __name__ == "__main__":
    if len(sys.argv) == 3:
        from_cell = int(sys.argv[1])
        to_cell = int(sys.argv[2])
    else:
        from_cell = 1
        to_cell = 113
        
    main(from_cell, to_cell)
    
    
    
    
    