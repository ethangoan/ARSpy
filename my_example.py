import numpy as np
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import tensorflow as tf

from arspy.hull import compute_hulls, Hull
import random

from tbnn.pdmp.poisson_process import SBPSampler

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
  for index in range(0, 12):
    #S = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0])
    #fS = f(S)
    #S_hi = np.linspace(-4, 4, 1000)
    #fS_hi = s_hi ** 3.0
    #S_hi = np.load('conv_time_array_0.npy')
    #fS_hi = 1.0 * np.load('conv_test_array_0.npy')
    # S_hi = np.load('../tbnn/tbnn/pdmp/event_test/conv_time_array_9.npy')
    # fS_hi = np.load('../tbnn/tbnn/pdmp/event_test/conv_test_array_9.npy')
    #S_hi = np.load('../tbnn/tbnn/pdmp/log_test/time_array_{}.npy'.format(index))
    #fS_hi = np.load('../tbnn/tbnn/pdmp/log_test/test_array_{}.npy'.format(index))
    S_hi = np.load('../tbnn/tbnn/pdmp/event_test/conv_time_array_{}.npy'.format(index))
    fS_hi = np.load('../tbnn/tbnn/pdmp/event_test/conv_test_array_{}.npy'.format(index))
    epsilon = 0.000001
    fS_hi_orig = np.copy(fS_hi)
    fS_hi[fS_hi < epsilon] = epsilon
    print(S_hi.shape)
    sample = np.sort(random.sample(list(np.arange(0, S_hi.size)), 10))
    sample = np.hstack([[0], sample])
    #sample = np.arange(0, S_hi.size, 100)
    print(sample)
    S = S_hi[sample]
    fS = fS_hi[sample]
    fS_orig = fS_hi_orig[sample]
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



    sbps = SBPSampler()
    G = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    X = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    for i in range(0, S_hi.size):
      X.write(i, np.array([S_hi[i], 1.0]))
      G.write(i, fS_hi_orig[i])
    print('G = {}'.format(G.stack()))
    print('G = {}'.format(G.stack()))
    x_time = tf.reshape(X.stack(), [-1, 2])
    G_vector = tf.reshape(G.stack(), shape=[-1, 1])
    beta = tf.linalg.inv(tf.transpose(x_time) @ x_time + 0.00001 * tf.eye(2)) @ tf.transpose(x_time) @ G_vector
    #beta = tf.transpose(x_time) @ G_vector
    beta_mean, beta_cov = sbps.sbps_beta_posterior([], [], G, X, beta, S.size)
    print(beta_mean)
    print(beta_cov)
    #sbps_bound = tf.maximum(0.0, tf.reshape(X.stack() @ tf.reshape(beta_mean, [2, 1]), -1))
    sbps_bound = tf.reshape(X.stack() @ tf.reshape(beta_mean, [2, 1]), -1).numpy()
    sbps_bound[sbps_bound < 0.0] = 0.0
    sbps_linear = tf.reshape(X.stack() @ tf.reshape(beta, [2, 1]), -1).numpy()

    print('sbps_bound = {}'.format(sbps_bound))
    #sbps_bound[sbps_bound < 0.0] = 0.0
    plt.plot(S_hi, sbps_bound, label='sbps')
    plt.plot(S_hi, sbps_linear, label='sbps_linear')
    plt.legend(loc=0)
    plt.savefig('arspy_test_{}.png'.format(index))
    plt.savefig('arspy_test_{}.pdf'.format(index))

    hull = Hull(upper_hull)
    inv_time, inverse = hull.eval_inverse_integrated()
    time, integrated = hull.eval_integrated()#time=inv_time)
    plt.figure()
    plt.plot(inv_time, inverse, label='inverse')
    plt.plot(time, integrated, label='integrated')
    plt.legend()
    plt.savefig('inverse_test_{}.png'.format(index))
    plt.savefig('inverse_test_{}.pdf'.format(index))

    inv_time, inverse = hull.eval_inverse_integrated(time=integrated)
    #time, integrated = hull.eval_integrated(time=inv_time)
    # plt.figure()
    # plt.plot(inv_time, inverse, label='inverse $H^{-1}(t)')
    # plt.plot(time, integrated, label='integrated')
    # plt.legend()
    # plt.savefig('inverse_test_{}.pdf'.format(index))


    # plt.savefig('sbps.pdf')
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
    # plt.savefig('test_{}.png'.format(index))
