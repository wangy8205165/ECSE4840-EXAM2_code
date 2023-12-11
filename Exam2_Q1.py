"""
ECSE4840 Exam Question 1
Author: Yixiang Wang
"""

import numpy as np

def loss_function(theta,X,y,lambda_reg):
    N = len(y)
    prediction = (X @ theta)
    loss1=np.sum((prediction - y)**2)/(2*N)
    loss2=(lambda_reg/2)*np.sum(theta**2)
    return loss1+loss2

def gradient_descent(X,y,theta_init,alpha,lambda_reg,num_iters):
    m = y.size
    theta = theta_init.copy()
    J_history = []
    
    for i in range(num_iters):
        theta -= alpha*((1/m)*(X.T @ (X @ theta - y))+lambda_reg*theta)
        J_history.append(loss_function(theta,X,y,lambda_reg))
    return theta,J_history

#convert the data in Table 1 into numpy arrays:
X = np.array([[0,1],[-1,1]])
y = np.array([-1,1])
#set up all the parameters needed:
alpha = 0.1 # learning rate
lambda_reg = 1 # Regularization coefficient
num_iters = 2 # Number of iterations
theta_init = np.array([0.0,0.0]) #Initial theta

theta, J_history = gradient_descent(X,y,theta_init,alpha,lambda_reg,num_iters)


I = np.array([[1,0],[0,1]])
theta_optimal = np.array([-4/11,-1/11])
Hessian = (1/2)*X.T@X+lambda_reg* I
eigenvalues = np.linalg.eigvals(Hessian)
beta = max(eigenvalues)

T = num_iters
norm_optimal=np.linalg.norm(theta_init-theta_optimal)**2
RHS = (2*beta*norm_optimal)/T
loss_optimal=loss_function(theta_optimal, X, y, lambda_reg)
loss_init=loss_function(theta_init,X,y,lambda_reg)
actual_error=J_history[-1]-loss_optimal
loss_2_0=J_history[-1]-loss_init

print("The theta at 2nd iteration is{}".format(theta))
print("Loss with theta 0 is {}".format(loss_init))
print("Loss with tehta 2 is {}".format(J_history[-1]))
print("The Right side of the inequality is {}".format(RHS))
print("The error between theta0 and theta2 is {}".format(loss_2_0))
print("The actual error between theta 2 and optimal theta is {}".format(actual_error)) 
print("Does the ineuqlaity hold True? {}".format(actual_error<RHS))
   
    
    
    
    
    
    
    
    
    
    
    