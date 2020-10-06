import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from arspy.hull import compute_hulls, Hull
import random

import matplotlib

def f(x, a=2.0, b=5.0):
  """
  Log beta distribution
  """
  #print(type(x))
  #print(np.log(x))
  #x = x.numpy()
  return ((a-1.) * tf.math.log(x) + (b-1.) * tf.math.log(1.-x))


def fprima(x, a=1.3, b=2.7):
  """
  Derivative of Log beta distribution
  """
  return (a-1)/x-(b-1)/(1-x)

if __name__ == '__main__':

  #S = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0])
  #fS = f(S)
  S_hi = np.load('conv_time_array_2.npy')
  fS_hi = 1.0 * np.load('conv_test_array_2.npy')
  print(S_hi.shape)
  sample = np.sort(random.sample(list(np.arange(0, S_hi.size)), 10))
  sample = np.hstack([[0], sample])
  #sample = np.arange(0, S_hi.size, 100)
  print(sample)
  S = S_hi[sample]
  fS = fS_hi[sample]
  print(S)
  print(fS)
  lower_hull, upper_hull = compute_hulls(S, fS, domain=[0.0, 1.0])
    #lower_hull, upper_hull = compute_hulls((-2.0, -1.996, -0.998, 0.0, 0.998, 1.996, 2.0), (-4.0, -3.984016, -0.996004, 0.0, -0.996004, -3.984016, -4.0), (float("-inf"), float("inf")))
  #print(upper_hull)
  plt.figure()
  plt.plot(S_hi, fS_hi, c='b', alpha=0.5, linewidth=3, label='target')
  plt.scatter(S, fS, alpha=0.25, c='b', label='samples', s=150)
  for node in upper_hull[:-1]:
    plt.plot([node.left, node.right],
             [node.left * node.m + node.b, node.right * node.m + node.b],
             alpha=1.0, c='r', linestyle='--')
    plt.scatter(np.array([node.left, node.right]),
                np.array([node.left * node.m + node.b, node.right * node.m + node.b]),
                alpha=0.5, c='r')
  plt.plot([upper_hull[-1].left, upper_hull[-1].right],
           [upper_hull[-1].left * upper_hull[-1].m + upper_hull[-1].b,
            upper_hull[-1].right * upper_hull[-1].m + upper_hull[-1].b],
           c='r', alpha=1.0, linestyle='--', label='envelope')
  # plt.scatter(np.array([upper_hull[-1].right]),
  #             np.array([upper_hull[-1].right * upper_hull[-1].m + upper_hull[-1].b]),
  #             c='r', alpha=0.5)

  plt.legend(loc=0)
  plt.savefig('arspy_test.png')
  plt.savefig('arspy_test.pdf')

  hull = Hull(upper_hull)
  inv_time, inverse = hull.eval_inverse_integrated()
  time, integrated = hull.eval_integrated()#time=inv_time)
  plt.figure()
  plt.plot(inv_time, inverse, label='inverse')
  plt.plot(time, integrated, label='integrated')
  plt.legend()
  plt.savefig('inverse_test.png')

  inv_time, inverse = hull.eval_inverse_integrated(time=integrated)
  #time, integrated = hull.eval_integrated(time=inv_time)
  plt.figure()
  plt.plot(inv_time, inverse, label='inverse')
  plt.plot(time, integrated, label='integrated')
  plt.legend()
  plt.savefig('inverse_test.png')

  # hull_sample = []
  # time = np.linspace(hull.hull_list[0].inverse_int_domain_lower,
  #                    hull.hull_list[-1].inverse_int_domain_upper, 100)
  # for t in time:
  #   hull_sample = h

  #upper_hull, upper_hull = compute_hulls(S, fS, [-np.inf, np.inf])
  #ars = ARS(f, fprima, xi = [0.1, 0.6], lb=0, ub=1, a=1.3, b=2.7)
  #samples = ars.draw(10000)
  #plt.hist(samples, bins=100, density=True)
  # plt.plot(ars.z, f(ars.z))
  # plt.plot(ars.z, ars.u + ars.offset)
  # plt.savefig('test.png')
