# Script for opening the predownloaded dataset on google colab:
from google.colab import files
files.upload()

from sklearn.cluster import KMeans
data=pd.read_csv("Training set.csv")
#removing the outliers in order to have a clear plot
data=data[data['Height']<=1000]
data=data[data['Weight']<200]
data=data[data['Height']>90]


X = np.column_stack((data.Height, data.Weight))
y=data.Sex.values
#changing the value of y to obtain 0/1 instead of "Male"/"Female"
for i in range(len(y)):
  if y[i]=='Female':
    y[i]=0
  else:
    y[i]=1
y = np.array(y).astype(float)

#plot of the label assigned and the labelled points

#separate the labelled points between the two classes
X_0=X[np.where(y==0)]
X_1=X[np.where(y==1)]

#plot of the labelled points
plt.scatter(X_0[:, 0], X_0[:, 1], c='yellow',label="Female")
plt.scatter(X_1[:, 0], X_1[:, 1], c='purple',label="Male")

plt.legend(loc="upper left")
plt.show()

# In order to test our model on 2 separate clusters we remove the point in the middle by taking out the point too close to the center of the other cluster
from sklearn.model_selection import train_test_split
X_labelled,X_unlabelled,y_labelled,y_unlabelled=train_test_split(X,y, test_size=0.9, random_state=0)
#divide X_labelled betwee the two classes
X_labelled_0=X_labelled[np.where(y_labelled==0)]
X_labelled_1=X_labelled[np.where(y_labelled==1)]

#plot the different classes
plt.scatter(X_unlabelled[:, 0], X_unlabelled[:, 1], c='gray', alpha=0.5, label="unlabelled points") 
plt.scatter(X_labelled_0[:, 0], X_labelled_0[:, 1], c='yellow', label="labelled points Female class")
plt.scatter(X_labelled_1[:, 0], X_labelled_1[:, 1], c='purple', label="labelled points Male class")   

copy_y_unlabelled=y_unlabelled.copy()
for i in range(len(y_unlabelled)):
  y_unlabelled[i]=-1

plt.legend(loc="upper left")
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

# define objective function
def objective(y_un):
  global y_labelled,weights_labelled,weights_unlabelled
  result = 0
  for i in range(len(y_labelled)):
    for j in range(len(y_un)):
      result += weights_labelled[i][j] *((y_labelled[i] - y_un[j]) ** 2)
  
  for i in range(len(y_un)):
    for j in range(len(y_un)):
      result+=(1/2)*( weights_unlabelled[i][j]*((y_un[i]-y_un[j])**2))
  return result


# define gradient of objective function
def gradient(y_un):
  global y_labelled,weights_labelled,weights_unlabelled
  result = np.zeros(len(y_un))
  for i in range(len(y_labelled)):
    for j in range(len(y_un)):
      result[j] += 2* weights_labelled[i][j] * (y_un[j]-y_labelled[i])
  
  for i in range(len(y_un)):
    for j in range(len(y_un)):
      result[j] +=2* weights_unlabelled[i][j]*(y_un[i]-y_un[j])
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

# Finding the optimal learning rate
mat = np.copy(-weights_unlabelled)
for i in range(len(y_unlabelled)):
  mat[i][i] = 2 * np.sum(weights_labelled[:,i]) + np.sum(weights_unlabelled[:,i]) - weights_unlabelled[i][i]

eigvals = np.linalg.eigvals(mat) 
L=max(eigvals)
print(L)
alpha=1/L
print(alpha)

#set the parameters (the value of alpha was already set using the hessian matrix)
max_iter=100
EPSILON=1e-4

# CLASSIC GRADIENT DESCENT
# define initial guess
y_unl=np.zeros(len(y_unlabelled))+1
# run gradient descent algorithm
start_time = time.perf_counter()
times=[]
accuracies=[]
grad=gradient(y_unl)
for iter in range(max_iter):
    y_unl = y_unl - alpha * grad
    current_time = time.perf_counter()
    times.append(current_time-start_time)
    accuracies.append(accuracy_score(copy_y_unlabelled, np.round(y_unl)))
    grad=update_full_gradient(grad)
    if EPSILON >= np.linalg.norm(grad):
        break
y_unl=np.clip(y_unl,0,1)
y_unl=np.round(y_unl)
end_time = time.perf_counter()
# print solution and number of ierations
print("Minimum point:", y_unl)
print("Number of iteration",iter+1)

plt.plot(times,accuracies)
plt.xlabel("time")
plt.ylabel("accuracy")
plt.title("Accuracy classic GD")

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
plt.scatter(X_unlabelled[:, 0], X_unlabelled[:, 1], c='grey', label="unlabelled points")
plt.scatter(misclassified_points_matrix[:,0],misclassified_points_matrix[:,1],c='red', label="misclassified points")
plt.legend(loc="upper left")
plt.title("Misclassified points")
plt.show()

#BCGD WITH RANDOMIZED RULE
#define initial value
y_unl_bcgd=np.zeros(len(y_unlabelled))+.5
previous_grad=gradient(y_unl_bcgd)
times2=[]
accuracies2=[]
max_iter=max_iter*len(y_unl_bcgd)
start_time=time.perf_counter()
for iter in range(max_iter):
  i=random.randint(0,len(y_unl_bcgd)-1)
  prev=y_unl_bcgd.copy()
  y_unl_bcgd[i]= y_unl_bcgd[i] - alpha * previous_grad[i]
  previous_grad=update_gradient(previous_grad,i)
  current_time=time.perf_counter()
  times2.append(current_time-start_time)
  accuracies2.append(accuracy_score(copy_y_unlabelled, np.round(y_unl_bcgd)))
  if EPSILON >= np.linalg.norm(previous_grad):
        break

y_unl_bcgd=np.clip(y_unl_bcgd,0,1)
y_unl_bcgd=np.round(y_unl_bcgd)

#print solution and number of iterations
print("Minimum point:", y_unl_bcgd)
print("Number of iteration",iter+1)

plt.plot(times2,accuracies2)
plt.xlabel("time")
plt.ylabel("accuracy")
plt.title("Accuracy BCGD with randomized rule")

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
plt.scatter(X_unlabelled[:, 0], X_unlabelled[:, 1], c='grey', label="unlabelled points")
plt.scatter(misclassified_points_matrix[:,0],misclassified_points_matrix[:,1],c='red', label="misclassified points")
plt.title("Misclassified points")
plt.show()

#BCGD WITH GAUSS-SOUTHWELL RULE
y_unl_gs=np.zeros(len(y_unlabelled))+.5
previous_grad=gradient(y_unl_gs)
times3=[]
start_time=time.perf_counter()
accuracies3=[]
for iter in range(max_iter*len(y_unl_gs)):
  abs_previous_grad=np.abs(previous_grad)
  i=np.argmax(abs_previous_grad)  
  y_unl_gs[i]= y_unl_gs[i] - alpha * previous_grad[i]
  previous_grad=update_gradient(previous_grad,i)
  current_time=time.perf_counter()
  times3.append(current_time-start_time)
  accuracies3.append(accuracy_score(copy_y_unlabelled,np.round( y_unl_gs)))
  if EPSILON >= np.linalg.norm(previous_grad):
    break
y_unl_gs=np.clip(y_unl_gs,0,1)
y_unl_gs=np.round(y_unl_gs)

#print solution and number of iterations
print("Minimum point:", y_unl_gs)
print("Number of iteration",iter+1)

plt.plot(times3,accuracies3)
plt.xlabel("time")
plt.ylabel("accuracy")
plt.title("Accuracy BCGD with GS rule")

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
plt.show()



