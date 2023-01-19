import cvxpy as cp
import numpy as np

class  active_learning():
    def  __init__(self, vectors, k):
            self .vectors = vectors
            self .k = k
            self.number_of_outliers = int (len(vectors) * 0.05)
            self.mip_centers = None

    def greedy_k_center(self): #not two time same center dc
            print("greedy k center" )
            centers = [self .vectors[np.random.randint(self .vectors.shape[0])]]
            vectors = self .vectors
            vectors = np.delete(vectors, np.where(np.all(vectors == centers[0], axis=1)), axis=0)
            for i in range(self.k-1):
                    if i >0 :
                          vectors = np.delete(vectors, np.where(np.all(vectors == centers[i-1], axis=1)), axis=0)
                    dists = np.min(np.sum((self .vectors - centers[-1])**2, axis=1)**0.5)
                    next_center = self .vectors[np.argmax(dists)]
                    centers.append(next_center)
            centers = np.array(centers)
            return centers

    def robust_k_center(self):

        # initialize the centers with greedy k-center
        centers_out = self .greedy_k_center()

        self.mip_centers = centers_out
        vectors = self .vectors
        #remove centers from vector
        for i in range(centers_out.shape[0]):
            vectors = np.delete(vectors, np.where(np.all(self.vectors == centers_out[i], axis=1)), axis=0)

        # find maximal distance tthat a point has to its nearest center, do not consider the centers that are already selected
        dists = np.zeros((vectors.shape[0], centers_out.shape[0]))
        #compute distance between each point and each center
        for i in range(vectors.shape[0]):
            for j in range(centers_out.shape[0]):
                dists[i,j] = np.linalg.norm(vectors[i]-centers_out[j])
        #take minimum for each vector
        mindists = np.min(dists, axis=1)
        #take maximum of the minimums
        ub = np.max(mindists)





        #lower bound
        lb = ub/2

        while ub-lb > 1:

            if self.feasible((lb+ub)/2):
                print("feasible" )
                #the the upper bound is the maximum distance between two points whose distance is less then lb+up/2
                #compute all distances between all the points
                dists = np.zeros((self.vectors.shape[0], self.vectors.shape[0]))
                for i in range(self.vectors.shape[0]):
                    for j in range(self.vectors.shape[0]):
                        dists[i,j] = np.linalg.norm(self.vectors[i]-self.vectors[j])
                # find distances less than lb+ub/2
                lessdists = dists[dists <= (lb+ub)/2]
                ub = np.max(lessdists)


            else:
                print("not feasible" )
                dists = np.zeros((self.vectors.shape[0], self.vectors.shape[0]))
                for i in range(self.vectors.shape[0]):
                    for j in range(self.vectors.shape[0]):
                        dists[i, j] = np.linalg.norm(self.vectors[i] - self.vectors[j])
                # find distances less than lb+ub/2
                lessdists = dists[dists >= (lb + ub) / 2]
                ub = np.min(lessdists)+ np.random.uniform(0, 1)
            print(ub-lb)

        return self.mip_centers


    def feasible(self, distance):

        n, d = self.vectors.shape
        # create decision variables

        #centers = cp.Variable((self.k, d))
        centers = cp.Variable(n, boolean= True)             #1 if the point is a center, 0 otherwise
        w = cp.Variable((n ,n ), boolean = True)         #i,j 1 if point j is the center of point i
        e = cp.Variable((n,n), boolean= True)                 #1 if the point is an outlier, 0 otherwise
        #z = cp.Variable((n, self.k))
        #y = cp.Variable(n)

        #create constraints
        constraints = []
        # at most self.k centers
        constraints.append(cp.sum(centers) == self.k)
        # at most self.number_of_outliers outliers
        constraints.append(cp.sum(e) <= self.number_of_outliers)
        # each point can be just in one cluster, the sum of each row of w is 1
        for i in range(n):
            constraints.append(cp.sum(w[i,:]) == 1)
        # w[i,j] <= centers[j] for all i,j
        for i in range(n):
            for j in range(n):
                constraints.append(w[i,j] <= centers[j])
        # if dist(i,j) > distance then w[i,j] = e[i,j]
        for i in range(n):
            for j in range(n):
                if np.linalg.norm(self.vectors[i]-self.vectors[j]) > distance:
                    constraints.append(w[i, j] == e[i,j])


        # create objective function
        obj = cp.Minimize(cp.sum(e))

        #create problem
        mip = cp.Problem(obj, constraints)
        #solve problem
        mip.solve()

        self.mip_centers = centers.value

        #check if the problem is feasible
        if mip.status == 'optimal':
            return True
        else:
            return False











vectors = np.array([[3,1],[1,5],[2,3],[2.5, 4],[5,5]])
a_l = active_learning(vectors, 2)
centers = a_l.robust_k_center()
print(centers)