import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

data = (np.array(pd.read_csv('../dataset/jain.txt',sep = '\t')))
print(data)

plt.scatter(data[:,0],data[:,1])
plt.show()

class User_KMeans:
    def __init__(self):
        """
        using the default values for initialization
        
        k : int, default=2
        The number of clusters to form as well as the number of centroids to generate.
        
        max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
        """
        self.k=2
        self.tol = 0.001
        self.max_iter = 300
        self.centroids = {}
        
    def fit(self,data):
        # taking the initial random centroids
        for i in range(self.k):
            self.centroids[i]=data[np.random.choice(len(data), self.k, replace=True)]
        
        for i in range(self.max_iter):
            self.classifications={} #initializing the empty classification dictionary
            
            for i in range(self.k):
                self.classifications[i]=[] #initializing the empty classification for each iteration
            
            for data_set in data:
                distances=[np.linalg.norm(data_set-self.centroids[centroid]) for centroid in self.centroids]
                self.classifications[distances.index(min(distances))].append(data_set)
                
            prev_centroids = dict(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification]=np.average(self.classifications[classification],axis=0)
            is_optimization_completed = True
            
            for centroid in self.centroids:
                if(np.sum((self.centroids[centroid]-prev_centroids[centroid])/prev_centroids[centroid]*100.0)>self.tol):
                    is_optimization_completed=False
            
            if is_optimization_completed:
                   break
    
    def predict(self,data):
        distances=[np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

model = User_KMeans()
model.fit(data)
predicted=[]
for i in data:
    predicted_op = model.predict(i)
    predicted.append(predicted_op)
print("Predicted")    
print(predicted)
print("Actual")  
print(data[:,2])
actual = data[:,2]

centroid1=centroid2=centroid3=centroid4=0
for i,j  in zip(predicted,actual):
      if i==0 and j==2:
        centroid1=centroid1+1
      elif i==1 and j==1:
        centroid2 = centroid2+1
      elif i==1 and j==2:
        centroid3 = centroid3+1
      elif i==0 and j==1:
        centroid4 = centroid4+1

print("Accuracy = ",100*(centroid1+centroid2)/(centroid1+centroid2+centroid3+centroid4))

for each_centroid in model.centroids:
    plt.scatter(model.centroids[each_centroid][0],model.centroids[each_centroid][1],marker="o")

for classification in model.classifications:
    for data_set in model.classifications[classification]:
         plt.scatter(data_set[0],data_set[1],marker="x")
plt.scatter(data[:,0],data[:,1])