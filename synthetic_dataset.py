# Create the synthetic dataset with just 5% of the samples labeled
X, y = make_blobs(n_samples=5000, centers=[[0, 0], [10, 10]], cluster_std=[1, 1])

X_labelled,X_unlabelled,y_labelled,y_unlabelled=train_test_split(X,y, test_size=0.95, random_state=0)
#divide X_labelled betwee the two classes
X_labelled_0=X_labelled[np.where(y_labelled==0)]
X_labelled_1=X_labelled[np.where(y_labelled==1)]

#plot the different classes
plt.scatter(X_unlabelled[:, 0], X_unlabelled[:, 1], c='gray', alpha=0.5, label="unlabelled points") 
plt.scatter(X_labelled_0[:, 0], X_labelled_0[:, 1], c='yellow', label="labelled points with y=0")
plt.scatter(X_labelled_1[:, 0], X_labelled_1[:, 1], c='purple', label="labelled points with y=1")    
copy_y_unlabelled=y_unlabelled.copy()
for i in range(len(y_unlabelled)):
  y_unlabelled[i]=-1

#print(X_unlabelled)
# Mostrare il grafico
plt.show()

# Compute the pairwise distance matrix
distance_matrix = pairwise_distances(X)
#print(distance_matrix)

# Compute the weights for the labeled points and the unlabeled points
labelled_indices = np.where(y_labelled != -1)
unlabelled_indices = np.where(y_unlabelled == -1)
gamma = 1 / (2 * np.median(distance_matrix) ** 2)
weights_labelled = rbf_kernel(X_labelled, X_unlabelled, gamma=gamma)
weights_unlabelled = rbf_kernel(distance_matrix[np.ix_(unlabelled_indices[0], labelled_indices[0])], gamma=gamma)

# Compute the weights for the unlabeled-unlabeled points
unlabelled_indices_1, unlabelled_indices_2 = np.meshgrid(unlabelled_indices, unlabelled_indices)
unlabelled_distances = distance_matrix[unlabelled_indices_1, unlabelled_indices_2]
unlabelled_weights = rbf_kernel(unlabelled_distances, gamma=gamma)

mat = np.copy(-weights_unlabelled)
for i in range(len(y_unlabelled)):
  mat[i][i] = 2 * np.sum(weights_labelled[:,i]) + np.sum(weights_unlabelled[:,i]) - weights_unlabelled[i][i]

eigvals = np.linalg.eigvals(mat) 
L=max(eigvals)
print(L)
alpha=1/L
print(alpha)

import time
def objective(y_un):
  result = 0
  for i in range(len(y_labelled)):
    for j in range(len(y_un)):
      result += weights_labelled[i][j] * (y_labelled[i] - y_un[j]) ** 2
  
  for i in range(len(y_un)):
    for j in range(len(y_un)):
      result+=(1/2)*( weights_unlabelled[i][j]*(y_un[i]-y_un[j])**2)
  return result


# define gradient of objective function
def gradient(y_un):
  result = np.zeros(len(y_un))
  for i in range(len(y_labelled)):
    for j in range(len(y_un)):
      result[j] += 2* weights_labelled[i][j] * (y_un[j]-y_labelled[i])
  
  for i in range(len(y_un)):
    for j in range(len(y_un)):
      result[j] += weights_unlabelled[i][j]*(y_un[i]-y_un[j])
  return result

#update all the gradient
label_unlabel = np.sum(weights_labelled, axis=0).reshape((-1,1))
unlabel_unlabel = np.sum(weights_unlabelled, axis=0).reshape((-1,1))
coeff_vector = (2 * label_unlabel + unlabel_unlabel)

def update_full_gradient(prev_grad):
  result = np.copy(prev_grad)
  for i in range(len(result)):
    result += weights_unlabelled[i] * alpha*prev_grad[i]   
    result[i] -= coeff_vector[i] * alpha*prev_grad[i]
  return result

#update just one component
def update_gradient(prev_grad,i):
  result = np.copy(prev_grad)
  result += weights_unlabelled[i] * alpha*prev_grad[i]
  result[i] -= coeff_vector[i] * alpha*prev_grad[i]
  return result

def objective_np(y_labelled,y_un):
  global weights_labelled, weights_unlabelled
  result=0
  y_labelled = np.array(y_labelled)  
  weights_labelled = np.array(weights_labelled)
  weights_unlabelled = np.array(weights_unlabelled)

  diff_labelled = y_labelled[:, np.newaxis] - y_un
  result += np.sum(weights_labelled * diff_labelled**2)

  diff_unlabelled = y_un[:, np.newaxis] - y_un
  result += (1/2) * np.sum(weights_unlabelled * diff_unlabelled**2)
  return result

#define a maximum number of iteration and epsilon such that if the norm of the gradient is smaller
#than epsilon we stop the method
max_iter=100
EPSILON=1e-4
# as learning rate we took the value that we found before

# define initial guess
y_unl = np.zeros(len(y_unlabelled))+.5
grad=gradient(y_unl)
for iter in range(max_iter):
    y_unl = y_unl - alpha * grad
    grad=update_full_gradient(grad)
    if EPSILON >= np.linalg.norm(grad):
        break
y_unl=np.clip(y_unl,0,1)
y_unl=np.round(y_unl)
end_time = time.perf_counter()
# print solution and number of ierations
print("Minimum point:", y_unl)
print("Number of iteration",iter+1)

#plot of the label assigned and the labelled points

#separate the labelled points between the two classes
X_labelled_0=X_labelled[np.where(y_labelled==0)]
X_labelled_1=X_labelled[np.where(y_labelled==1)]

#separate the unlabelled points between the two predicted classes
X_unlabelled_0=X_unlabelled[np.where(y_unl==0)]
X_unlabelled_1=X_unlabelled[np.where(y_unl==1)]

#plot of the labelled points
plt.scatter(X_labelled_0[:, 0], X_labelled_0[:, 1], c='yellow')
plt.scatter(X_labelled_1[:, 0], X_labelled_1[:, 1], c='purple')

#plot of the unlabelled points
plt.scatter(X_unlabelled_0[:, 0], X_unlabelled_0[:, 1], c='yellow', alpha=0.5)
plt.scatter(X_unlabelled_1[:, 0], X_unlabelled_1[:, 1], c='purple',alpha=0.5)

plt.show()

#accuracy score
accuracy = accuracy_score(copy_y_unlabelled, y_unl)
print("Accuracy:", accuracy)

#plot of just the misclassified points
misclassified_points=[]
for i in range(len(y_unl)):
  if y_unl[i]!=copy_y_unlabelled[i]:
    misclassified_points.append(X_unlabelled[i])
misclassified_points_matrix=np.zeros((len(misclassified_points),2))
for i in range(len(misclassified_points)):
  misclassified_points_matrix[i,:]=misclassified_points[i]
plt.scatter(X_labelled[:, 0], X_labelled[:, 1], c='grey')  # Tracciare i punti labelled
plt.scatter(X_unlabelled[:, 0], X_unlabelled[:, 1], c='grey')  # Tracciare i punti unlabelled
plt.scatter(misclassified_points_matrix[:,0],misclassified_points_matrix[:,1],c='red')
plt.title("Misclassified points")
plt.show()

y_unl_bcgd=np.zeros(len(y_unlabelled))+.5
previous_grad=gradient(y_unl_bcgd)
for iter in range(max_iter*len(y_unl_bcgd)):
  i=random.randint(0,len(y_unl_bcgd)-1)
  y_unl_bcgd[i]= y_unl_bcgd[i] - alpha * previous_grad[i]
  previous_grad=update_gradient(previous_grad,i)
  if EPSILON >= np.linalg.norm(previous_grad):
        break
        
y_unl_bcgd=np.clip(y_unl_bcgd,0,1)
y_unl_bcgd=y_unl_bcgd.round()

#plot of the label assigned and the labelled points

#separate the labelled points between the two classes
X_labelled_0=X_labelled[np.where(y_labelled==0)]
X_labelled_1=X_labelled[np.where(y_labelled==1)]

#separate the unlabelled points between the two predicted classes
X_unlabelled_0=X_unlabelled[np.where(y_unl_bcgd==0)]
X_unlabelled_1=X_unlabelled[np.where(y_unl_bcgd==1)]

#plot of the labelled points
plt.scatter(X_labelled_0[:, 0], X_labelled_0[:, 1], c='yellow')
plt.scatter(X_labelled_1[:, 0], X_labelled_1[:, 1], c='purple')

#plot of the unlabelled points
plt.scatter(X_unlabelled_0[:, 0], X_unlabelled_0[:, 1], c='yellow', alpha=0.5)
plt.scatter(X_unlabelled_1[:, 0], X_unlabelled_1[:, 1], c='purple',alpha=0.5)

plt.show()

#accuracy score
accuracy = accuracy_score(copy_y_unlabelled, y_unl_bcgd)
print("Accuracy:", accuracy)

#plot of just the misclassified points
misclassified_points=[]
for i in range(len(y_unl_bcgd)):
  if y_unl_bcgd[i]!=copy_y_unlabelled[i]:
    misclassified_points.append(X_unlabelled[i])
misclassified_points_matrix=np.zeros((len(misclassified_points),2))
for i in range(len(misclassified_points)):
  misclassified_points_matrix[i,:]=misclassified_points[i]
plt.scatter(X_labelled[:, 0], X_labelled[:, 1], c='grey')  # Tracciare i punti labelled
plt.scatter(X_unlabelled[:, 0], X_unlabelled[:, 1], c='grey')  # Tracciare i punti unlabelled
plt.scatter(misclassified_points_matrix[:,0],misclassified_points_matrix[:,1],c='red')
plt.title("Misclassified points")
plt.show()

y_unl_gs=np.zeros(len(y_unlabelled))+.5
previous_grad=gradient(y_unl_gs)
for iter in range(max_iter*len(y_unl_gs)):
  abs_previous_grad=np.abs(previous_grad)
  i=np.argmax(abs_previous_grad)  
  y_unl_gs[i]= y_unl_gs[i] - alpha * previous_grad[i]
  #y_unl_gs=np.clip(y_unl_gs,0,1)
  previous_grad=update_gradient(previous_grad,i)
  if EPSILON >= np.linalg.norm(previous_grad):
    break
y_unl_gs=np.clip(y_unl_gs,0,1)
y_unl_gs=np.round(y_unl_gs)

#plot of the label assigned and the labelled points

#separate the labelled points between the two classes
X_labelled_0=X_labelled[np.where(y_labelled==0)]
X_labelled_1=X_labelled[np.where(y_labelled==1)]

#separate the unlabelled points between the two predicted classes
X_unlabelled_0=X_unlabelled[np.where(y_unl_gs==0)]
X_unlabelled_1=X_unlabelled[np.where(y_unl_gs==1)]

#plot of the labelled points
plt.scatter(X_labelled_0[:, 0], X_labelled_0[:, 1], c='yellow')
plt.scatter(X_labelled_1[:, 0], X_labelled_1[:, 1], c='purple')

#plot of the unlabelled points
plt.scatter(X_unlabelled_0[:, 0], X_unlabelled_0[:, 1], c='yellow', alpha=0.5)
plt.scatter(X_unlabelled_1[:, 0], X_unlabelled_1[:, 1], c='purple',alpha=0.5)

plt.show()

#accuracy score
accuracy = accuracy_score(copy_y_unlabelled, y_unl_gs)
print("Accuracy:", accuracy)

#plot of just the misclassified points
misclassified_points=[]
for i in range(len(y_unl_gs)):
  if y_unl_gs[i]!=copy_y_unlabelled[i]:
    misclassified_points.append(X_unlabelled[i])
misclassified_points_matrix=np.zeros((len(misclassified_points),2))
for i in range(len(misclassified_points)):
  misclassified_points_matrix[i,:]=misclassified_points[i]
plt.scatter(X_labelled[:, 0], X_labelled[:, 1], c='grey')  # Tracciare i punti labelled
plt.scatter(X_unlabelled[:, 0], X_unlabelled[:, 1], c='grey')  # Tracciare i punti unlabelled
plt.scatter(misclassified_points_matrix[:,0],misclassified_points_matrix[:,1],c='red')
plt.title("Misclassified points")
plt.show()
