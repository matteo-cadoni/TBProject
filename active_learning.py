import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from geomloss import SamplesLoss
import kmedoids

class  active_learning():
    def  __init__(self, vectors, k):
            self .vectors = vectors
            self .k = k
            self.number_of_outliers = int (len(vectors) * 0.05)
            print("number of outliers: " , self.number_of_outliers)
            self.mip_centers = None

    def greedy_k_center(self): #not two time same center dc
            print("greedy k center" )
            centers = [self .vectors[np.random.randint(self .vectors.shape[0])]]

            vectors = self .vectors
            vectors = np.delete(vectors, np.where(np.all(vectors == centers[0], axis=1)), axis=0)

            for i in range(self.k-1):
                print(i)
                dists = np.zeros((vectors.shape[0], len(centers)))
                for k in range(len(vectors)):
                    for j in range(len(centers)):
                        dists[k,j] = np.linalg.norm(vectors[k]-centers[j])
                mindists = np.min(dists, axis=1)
                next_center = vectors[np.argmax(mindists)]
                centers.append(next_center)
                vectors = np.delete(vectors, np.where(np.all(vectors == next_center, axis=1)), axis=0)

            centers = np.array(centers)
            return centers

    def robust_k_center(self):

        # initialize the centers with greedy k-center
        centers_out = self .greedy_k_center()
        # initialize the centers with random points
        #centers_out = self.vectors[np.random.choice(self.vectors.shape[0], self.k, replace=False)]

        self.mip_centers = centers_out
        vectors = self .vectors
        #remove centers from vector
        print("centers_out: " , centers_out.shape)
        for i in range(centers_out.shape[0]):
            print(np.where(np.all(self.vectors == centers_out[i], axis=1)))
            vectors[ np.where(np.all(self.vectors == centers_out[i], axis=1))] = 0
        vectors = vectors[vectors[:,0] != 0]

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

        while ub-lb > 0.1:

            if self.gurobi_feasible((lb+ub)/2):
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
                lb = np.min(lessdists)
            print("ub: " , ub)
            print("lb: " , lb)

        centers = self.vectors[self.mip_centers == 1]
        return (lb + ub) / 2, centers


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



        #check if the problem is feasible
        if mip.status == 'optimal':
            self.mip_centers = centers.value
            return True
        else:
            return False
    def gurobi_feasible(self, distance):
        m = gp.Model("k-center")
        n, d = self.vectors.shape
        # create decision variables
        centers = m.addVars(n, vtype=gp.GRB.BINARY, name="centers")
        w = m.addVars(n, n, vtype=gp.GRB.BINARY, name="w")
        e = m.addVars(n, n, vtype=gp.GRB.BINARY, name="e")
        # create constraints
        # at most self.k centers
        m.addConstr(gp.quicksum(centers[i] for i in range(n)) == self.k)
        # at most self.number_of_outliers outliers
        m.addConstr(gp.quicksum(e[i, j] for i in range(n) for j in range(n)) <= self.number_of_outliers)
        # each point can be just in one cluster, the sum of each row of w is 1
        for i in range(n):
            m.addConstr(gp.quicksum(w[i, j] for j in range(n)) == 1)
        # w[i,j] <= centers[j] for all i,j
        for i in range(n):
            for j in range(n):
                m.addConstr(w[i, j] <= centers[j])
        # if dist(i,j) > distance then w[i,j] = e[i,j]
        for i in range(n):
            for j in range(n):
                if np.linalg.norm(self.vectors[i] - self.vectors[j]) > distance:
                    m.addConstr(w[i, j] == e[i, j])
        # create objective function
        m.setObjective(gp.quicksum(e[i, j] for i in range(n) for j in range(n)), gp.GRB.MINIMIZE)
        # solve problem
        m.optimize()
        # check if the problem is feasible

        if m.status == gp.GRB.OPTIMAL:
            #print the centers
            self.mip_centers = np.zeros(n)
            for i in range(n):
                self.mip_centers[i] = centers[i].x


            return True
        else:
            return False

    def ot_clustering(self):
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        # compute the distance matrix with sampleloss
        dists = np.zeros((self.vectors.shape[0], self.vectors.shape[0]))
        for i in range(self.vectors.shape[0]):
            for j in range(self.vectors.shape[0]):
                if i < j:
                    dists[i, j] = loss(self.vectors[i], self.vectors[j])
                    dists[j, i] = dists[i, j]
        fp = kmedoids.fasterpam(dists, 100)
        return fp











#create points randomly values between 0 and 50
vectors = np.random.randint(150, size=(100, 32))

a_l = active_learning(vectors, 5)
#centers = a_l.robust_k_center()
#centers = a_l.greedy_k_center()
#print(centers)
#plot points
#plt.scatter(vectors[:,0], vectors[:,1])
#plot centers
#plt.scatter(centers[:,0], centers[:,1], c='r')
#plt.show()

#1 hot encoded centers
distance, centers2 = a_l.robust_k_center()
print(centers2)
#plot points
"""
plt.scatter(vectors[:,0], vectors[:,1])
#plot centers
plt.scatter(centers2[:,0], centers2[:,1], c='r')
#plot radius of the centers as a circle
for i in range(centers2.shape[0]):
    circle = plt.Circle((centers2[i,0], centers2[i,1]), distance, color='r', fill=False)
    plt.gcf().gca().add_artist(circle)

plt.show()
"""
"""
#create 3 dimensional points
vectors = np.random.randint(10, size=(100, 3))
a_l = active_learning(vectors, 3)
dist, centers = a_l.robust_k_center()
"""