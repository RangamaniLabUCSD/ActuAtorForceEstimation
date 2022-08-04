#Import JAX
from jax.config import config
config.update('jax_enable_x64', True)

import jax
import jax.numpy as np

# Create simple data
x = np.arange(10,dtype=np.float64)
xi = np.ones(3)

A = np.ones((10,3))

# Create three functions, increasing in complexity
f = lambda x: np.array([x,x**2,x**3]).T     # N x 3
f2 = lambda x,xi: x*np.dot(f(x),xi)
f3 = lambda x,xi: x*np.dot(f(x),xi)+np.dot(A,xi)

def egrad(g):
  def wrapped(x, *rest):
    y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
    x_bar, = g_vjp(np.ones_like(y))
    return x_bar
  return wrapped

# Print results
print("Function f")
print(egrad(f)(x))
y, vjp = jax.vjp(f, x)
print("VJP", vjp(y))
print("jacobian", jax.jacrev(f)(x).shape)

print("")
print("Function f2")
print(egrad(f2)(x,xi))
print("")
print("Function f3")
print(egrad(f3)(x,xi))
print("")
y, vjp = jax.vjp(f3, x, xi)
print(y) ## answer

print(type(vjp))
print(vjp(np.ones_like(y)))
print("")
print(jax.jacrev(f3)(x,xi))
print("")
print(f3(x,xi))

print(np.ones_like(y))
