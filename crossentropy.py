import numpy as np

#Example and explantion based on:
#https://stackoverflow.com/questions/41990250/what-is-cross-entropy

#Cross-entropy
#Is commonly used to quantify the difference between two probability distributions. 
#Usually the "true" distribution (the one that your machine learning algorithm is trying to match) is expressed in terms of a one-hot distribution.

#For example, suppose for a specific training instance, the label is B (out of the possible labels A, B, and C). 
#The one-hot distribution for this training instance is therefore:

#Pr(Class A)  Pr(Class B)  Pr(Class C)
#        0.0          1.0          0.0

p = np.array([0,1,0])

print("\np:=",p)

#You can interpret the above "true" distribution to mean that the training instance has 0% probability of being class A, 100% probability of being class B, and 0% probability of being class C.

#Now, suppose your machine learning algorithm predicts the following probability distribution:

#Pr(Class A)  Pr(Class B)  Pr(Class C)
#      0.228        0.619        0.153

q = np.array([0.228,0.619,0.153])

print("\nq:=",q)

#How close is the predicted distribution to the true distribution? That is what the cross-entropy loss determines. Use this formula:

#Cross entropy loss formula: H(p,q) = - Sum p(x) log q(x)
#										 x

#The sum is over the three classes A, B, and C.
#If you complete the calculation, you will find that the loss is 0.479. So that is how "wrong" or "far away" your prediction is from the true distribution.

loss = - np.sum(p * np.log(q))
loss2 = - p.dot(np.log(q))
print("\nloss:=",loss)
print("\nloss2:=",loss2)

# Lets say we predicted almost corrected, since we cannot take log of zeros, and the output function in our model commonly being softmax function to avoid this problem.

q = np.array([0.1,.8,0.1])

print("\nq:=",q)

loss = - np.sum(p * np.log(q))
loss2 = - p.dot(np.log(q))
print("\nloss:=",loss)
print("\nloss2:=",loss2)

# And we go even better

q = np.array([0.001,0.998,0.001])

print("\nq:=",q)

loss = - np.sum(p * np.log(q))
loss2 = - p.dot(np.log(q))
print("\nloss:=",loss)
print("\nloss2:=",loss2)