import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt


#Read data from file
dataset = scipy.io.loadmat('mat/KSamples.mat')

# Returns an array of cluster numbers for each data point.
# Input parameters: D - Data points, C - Centroids
# Algorithm: For each data point, find the euclidean distance from each centroid. And then
#             assign the cluster number for which the distance is smallest.
def assign(D, C):
  clust = []
  for pt in D:
      x=pt[0]
      y=pt[1]
      dist = []
      for ct in C:
           cx=ct[0]
           cy=ct[1]
           d = np.sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy))
           dist.append(d)
      min = 0
      for i in range(len(dist)):
         if(dist[i]<dist[min]):
              min = i
      clust.append(min)

  return clust

# Return the new centroids of the cluster set.
# Input Parameters:  D - data points, C - cluster numbers.
# Algorithm: Computes the new centroid for each cluster is calculated as the mean of the data points in each cluster. 

def centroid(D, C):
  
  clusterlist=[]
  centroids=[]
  max=C[0]
  for i in range(len(C)):
      if(C[i]>max):
        max=C[i]
  for i in range(max+1):
      clusterlist.append([])
  for i in range(len(D)):
        pt=D[i]
        cn=C[i]
        clusterlist[cn].append(pt)
  for l1 in clusterlist:
      sumx=float(0.0)
      sumy=float(0.0)
      for p in l1:
          x= p[0]
          y= p[1]
          sumx=sumx+x
          sumy=sumy+y
      if(len(l1) != 0):
          cx=sumx/len(l1)
          cy=sumy/len(l1)
      else:
          cx = 0
          cy = 0
      centroid=[cx,cy]
      centroids.append(centroid)
  return centroids

# Return the intial random centriods using strategy 1.
# Input Parameters: D - Data points, k - number of centroids
def get_intial_centriod_1(D,k):
  centriods = []
  for x in range(k):
      y =  random.randint(0,len(D)-1)
      centriods.append([D[y][0], D[y][1]])
  return centriods

# Return the intial random centriods using strategy 2.
# Input Parameters: D - Data points, k - number of centroids
# Algorithm: pick only the first centriod randomly, from that initial centriod rest are calculated 
#             by finding avg distance between the point and each centriod and then choose the next centriod 
#             point such that average distance of this chosen one to all previous centers is maximum
def get_intial_centriod_2(D,k):
    new_centriods_array = []
    for i in range(k):
        if i==0:
             y = random.randint(0,len(D)-1)
             new_centriods_array.append([D[y][0], D[y][1]])
        else:        
            avg_dist_arr = []
            for pt in D:
                sum_dist=0
                for ct in new_centriods_array:
                    d = np.sqrt((pt[0]-ct[0])*(pt[0]-ct[0]) + (pt[1]-ct[1])*(pt[1]-ct[1]))
                    sum_dist=sum_dist+d
                avg_dist_pt=sum_dist/len(ct)
                avg_dist_arr.append(avg_dist_pt)
            max_dist_pt=0
            for i in range(len(avg_dist_arr)):
                if(avg_dist_arr[i]>avg_dist_arr[max_dist_pt]):
                     max_dist_pt=i
            new_centriods_array.append([D[max_dist_pt][0], D[max_dist_pt][1]])
    return new_centriods_array
 
# Returns true if converged
# Input Parameters: Initial centroids, new centroids
# Explanation: In this method we are passing the two parameter initcen and newcen to check the points are 
#              conveged or not
def isconverged(initcen, newcen):
    pt_initcen=[]
    pt_newcen = []
    for i in range(len(initcen)):
        pt_initcen=initcen[i]
        pt_newcen = newcen[i]
        if((pt_initcen[0]!=pt_newcen[0]) or (pt_initcen[1]!= pt_newcen[1])):
            return 0
    return 1     

# Returns Cluster centroids and cluster numbers for each data point.
# Input Parameters: D - Data points, k - number of clusters, Strategy 1 or 2.
# Explanation: This method does K-means algorithm. That is we do multiple 
#              iteranations of finding the centroids and the cluster numbers until it converges.
#              If the points are not yet converged, we repeat the process until it get converged.
       
def kmeans(D,k,strategy):
        if(strategy==1):
             init_centroids = get_intial_centriod_1(D,k)
        else:
             init_centroids = get_intial_centriod_2(D,k)

        while(1):
            clust_number = assign(D, init_centroids)
            centroid_clust=centroid(D,clust_number)
            if(isconverged(init_centroids,centroid_clust)):
                break
            init_centroids=centroid_clust
        return centroid_clust, clust_number
    
# Returns the objective function value using the formula given.
# Input Parameters: D - Data points, Converged centorids, Cluster numbers.
def objective_fuction(D, con_centroid, clust_number):
    sum_object=0
    for i in range(len(D)):
           pt = D[i]
           cn = clust_number[i]
           ct = con_centroid[cn]
           d2 = ((pt[0]-ct[0])*(pt[0]-ct[0]) + (pt[1]-ct[1])*(pt[1]-ct[1]))
           sum_object=sum_object+d2
    return sum_object

# Returns an array of object values for the given k value array.
# Input parameters: K - array of cluster's size, Strategy - 1 or 2.
def kmeans_objective_values(K, strategy):
    object_values=[]
    for k in K:
       convergence_cluster, clust_number= kmeans(D,k,strategy)   
       objectfunction_value=objective_fuction(D,convergence_cluster, clust_number)
       object_values.append(objectfunction_value)
       print "    (", k,", ", objectfunction_value, ")"
    return object_values
        


D = np.array(dataset['AllSamples'])
K = [2,3,4,5,6,7,8,9,10]

print "#### Strategy 1####" 
print " Initialization 1:" 
strategy_01_objective_values_01=kmeans_objective_values(K,1)
print " Initialization 2:" 
strategy_01_objective_values_02=kmeans_objective_values(K,1)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))

ax1.scatter(K, strategy_01_objective_values_01)
ax1.plot(K, strategy_01_objective_values_01)
ax1.scatter(K, strategy_01_objective_values_02)
ax1.plot(K, strategy_01_objective_values_02)
ax1.set_title("Strategy 1 : Objective function value vs k")
ax1.set_xlabel("k values")
ax1.set_ylabel("Objective function value")

print "#### Strategy 2 ####" 
print " Initialization 1:" 
strategy_02_objective_values_01=kmeans_objective_values(K,2)
print " Initialization 2: " 
strategy_02_objective_values_02=kmeans_objective_values(K,2)
    
ax2.scatter(K, strategy_02_objective_values_01)
ax2.plot(K, strategy_02_objective_values_01)
ax2.scatter(K, strategy_02_objective_values_02)
ax2.plot(K, strategy_02_objective_values_02)
ax2.set_title("Strategy 2 : Objective function value vs k")
ax2.set_xlabel("k values")
ax2.set_ylabel("Objective function value ")

plt.show()


