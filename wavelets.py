
# coding: utf-8

# In[106]:

import numpy as np
import pylab
import seaborn as sns
from scipy.integrate import quad
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format = 'retina';")


# Wavelets are one type of adaptive, nonparametric estimation method.
# Here we explore the basic ideas.

# We first define the Father and Mother Haar wavelets:
# $$\phi(x) = 
# \begin{cases}
# 1 \mathrm{~if~} x \in [0,1] \\
# 0 \mathrm{~otherwise~} \\
# \end{cases}
# $$
# and
# $$\psi(x) = 
# \begin{cases}
# -1 \mathrm{~if~} x \in [0,1/2] \\
# 1 \mathrm{~if~} x \in (1/2,1] \\
# 0 \mathrm{~otherwise~} \\
# \end{cases}
# $$

# In[43]:

phi = lambda x: 1 if (x >= 0 and x <= 1) else 0
psi = lambda x: -1 if (x >= 0 and x <= 0.5)     else 1 if (x > 0.5 and x <= 1) else 0


# Here's what the two functions look like:

# In[92]:

x = np.linspace(-2,2,500)
y1 = np.array([phi(i) for i in x])

pylab.step(x, y1)
pylab.xlim(-.5,1.5); pylab.ylim(-1.5,1.5); 
pylab.ylabel(r"$\phi(x)$"); pylab.xlabel(r"$x$")


# In[93]:

y2 = np.array([psi(i) for i in x])
pylab.step(x, y2)

pylab.xlim(-.5,1.5); pylab.ylim(-1.5,1.5); 
pylab.ylabel(r"$\psi(x)$"); pylab.xlabel(r"$x$")


# Now we define the wavelets as shifted and rescaled versions of the
# Father and Mother wavelets:
# $$\phi_{jk}(x) = 2^{j/2} \phi(2^j x - k)$$
# and
# $$\psi_{jk}(x) = 2^{j/2} \psi(2^j x - k).$$
# 
# Below we plot some examples: $\phi_{2,2}, \psi_{2,2}$.

# In[62]:

phi_jk = lambda x, j, k: 2**(j/2.) * phi(2**j * x - k)
psi_jk = lambda x, j, k: 2**(j/2.) * psi(2**j * x - k)


# In[90]:

y1 = np.array([phi_jk(i,2,2) for i in x])

pylab.step(x, y1)
pylab.xlim(0.,1.2); pylab.ylim(-.5,2.5); pylab.ylabel(r"$\phi_{2,2}(x)$");
pylab.xlabel(r"$x$")


# In[91]:

y1 = np.array([psi_jk(i,2,2) for i in x])

pylab.step(x, y1)
pylab.xlim(0.,1.2); pylab.ylim(-2.5,2.5); pylab.ylabel(r"$\psi_{2,2}(x)$");
pylab.xlabel(r"$x$")


# The set of rescaled and shifted mother wavelets at resolution $j$
# is defined as:
# $$W_j = \{\psi_{jk}, k=0,1,\ldots,2^{j-1}\}.$$
# 
# We plot an example where $j=3$:

# In[105]:

for k in np.arange(0,5):
    y = np.array([psi_jk(i,3,k) for i in x])
    pylab.step(x, y)

pylab.xlim(-0.4,1.); pylab.ylim(-3.5,3.5); pylab.ylabel(r"$\psi_{j,k}(x)$");
pylab.xlabel(r"$x$")
pylab.title(r"Set of rescaled and shifted mother wavelets at resolution $j=3$")
pylab.legend([r"$k=0$",r"$k=1$",r"$k=2$",r"$k=3$",r"$k=4$"])


# **Theorem**: The set of functions $$\{\phi, W_0, W_2, W_2,\ldots\}$$
# is an orthonormal basis for $L_2(0,1)$, i.e., the set of real-valued functions on $[0,1]$ where $\int_0^1 f^2(x) dx < \infty$.
# 
# As a result, we can expand any function $f \in L_2(0,1)$ in this basis:
# $$
# f(x) = \alpha \phi(x) + \sum_{j=0}^{\infty} \sum_{k=0}^{2^j-1} 
# \beta_{jk} \phi_{jk}(x),
# $$
# where $\alpha = \int_0^1 \phi(x) dx$ is the scaling coefficient, and
# $\beta_{jk} = \int_0^1 f(x) \psi_{jk}(x) dx$ are the detail coefficients.
# 
# So to approximate a function $f \in L_2(0,1)$, we can take the finite
# sum
# $$
# f_J(x) = \alpha \phi(x) + \sum_{j=0}^{J-1} \sum_{k=0}^{2^j-1} 
# \beta_{jk} \phi_{jk}(x).
# $$
# This is called the resolution $J$ approximation, and has $2^J$ terms.
# 
# We consider an example below. Suppose we are interested in approximating
# the Doppler function:

# In[111]:

doppler = lambda x: np.sqrt(x*(1-x)) * np.sin(2.1*np.pi/(x+.05))
x = np.linspace(0,1,500)
pylab.plot(x, doppler(x))
pylab.title("Doppler function")


# In[154]:

def compute_sum(J, x):
    a_int = lambda x: doppler(x) * phi(x) 
    beta_int_jk = lambda x, j, k: doppler(x) * psi_jk(x,j,k) 
    
    total = alpha * phi(x)
    for j in range(0,J):
        for k in range(0, 2**j):
            beta_jk = quad(beta_int_jk, 0, 1, args=(j,k))[0]
            total += beta_jk * psi_jk(x, j, k)
    return total
    
finite_3 = lambda x: compute_sum(3, x)
finite_5 = lambda x: compute_sum(5, x)
finite_8 = lambda x: compute_sum(8, x)
finite_10 = lambda x: compute_sum(10, x)
finite_20 = lambda x: compute_sum(20, x)


# In[130]:

compute_sum(3, 0.5)
finite_3(0.5)


# In[147]:

x = np.linspace(0,1,500)
y = [finite_3(i) for i in x]
pylab.plot(x, y)
pylab.title(r"Doppler function approx at resolution $J=3$")


# In[150]:

x = np.linspace(0,1,500)
y = [finite_5(i) for i in x]
pylab.plot(x, y)
pylab.title(r"Doppler function approx at resolution $J=5$")


# In[151]:

x = np.linspace(0,1,500)
y = [finite_8(i) for i in x]
pylab.plot(x, y)
pylab.title(r"Doppler function approx at resolution $J=8$")


# In[153]:

x = np.linspace(0,1,500)
y = [finite_10(i) for i in x]
pylab.plot(x, y)
pylab.title(r"Doppler function approx at resolution $J=10$")


# In[183]:

_= """
def get_coefs(J, xvals):
    father_coefs = []
    for x in xvals:
        father_coefs.append(phi(x))
        mother_coefs = [[] for i in range(J)]
        for j in range(0,J):
            for k in range(0, 2**j):
                mother_coefs[j].append(psi_jk(x, j, k))
    return father_coefs, mother_coefs
x = np.linspace(0,1,200)
father_coefs, mother_coefs = get_coefs(3, x)
y = [finite_3(i) for i in x]
pylab.plot(x, y)
pylab.title(r"Doppler function approx at resolution $J=3$")
"""


# In[174]:




# In[176]:




# In[ ]:



