#This code is subjected to GNU LESSER GENERAL PUBLIC LICENSE
#This code is linked to the work 
#"Deep learning for gradient flows using the Brezis-Ekeland principle"


import numpy as np
import math as m
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.util import nest
import tensorflow_probability as tfp
import os
import time

start = time.time()
#parameter initialzation
pi = tf.constant(m.pi)
#lamda for penalty term
lamb= 100   #variable \lambda in the paper
a = [2., 2., 1. , 2. , 3.]
# space domain
x_min = 0.    
x_max = pi  
#time domain
t_min = 0.        
t_max =  0.0001 
dt = 0.0001
time_step = 0.0001 #variable \Delta t in the paper
# n. of samples 
n_sample = 100000  #variable N_i in the paper
n_sampleBc = 100   #variable N_b in the paper
# n. of dim
n_dim = 5     # variable d in the paper
#number of iterations
time_iter = 10     # variable N in the paper
inner_iter = 200   # variable K in the paper

string = 'model0'


class PDEsolver(object):
  #this class represent the pde solver

  # here we define all the variables needed
  def __init__(self, pi, x_min, x_max, t_min, t_max, N, N_Bc, n , lamb, dt, time_step, a ):
  
    self.pi = pi
    self.lamb = lamb
    self.nDim = n 
    self.nSample = N
    self.nSampleBc = N_Bc 
    self.totSample = self.nSample +  2 * self.nDim * self.nSampleBc  #variable N_s in the paper
    self.xMax = x_max
    self.xMin = x_min
    self.tMax = t_max
    self.tMin = t_min  
    self.dt = dt
    self.timeStep = time_step
    self.a = tf.Variable(a) #a must be a vector of dim nDim, variable a in the paper
    self.c = 1/(tf.reduce_sum(self.a**2))  # variable \kappa in the paper

    # x is a tensor created by using the random distribution from [xMin, xMax). It has nSample rows and nDim columns.
    #It contains the internal points of the domain.
    self.x = tf.Variable(tf.random.uniform([self.nSample, self.nDim ], self.xMin + 0.001 , self.xMax))

    # xBc is a tensor created by using the random distribution from [xMin, xMax). It has nSampleBc rows and 2*nDim^2 columns.
    #It contains the points of the boundaries. The idea is that, according to nDim, every nDim columns define the points of one boundary of the domain.
    #Starting from this random tensor, self.xBc is modified to constrain points on the 0 boundary and on the pi boundary
    self.xBc = tf.Variable(tf.random.uniform([self.nSampleBc, 2 * self.nDim**2], self.xMin, self.xMax))   
    i = 0
    while i < self.nDim: 
      self.xBc[: , i*(self.nDim*2) + i ].assign( tf.Variable(tf.zeros([self.nSampleBc,  ])))
      self.xBc[: , i*(self.nDim*2) +i + self.nDim].assign( tf.Variable(tf.transpose(tf.repeat([self.pi], self.nSampleBc))))
      i = i + 1


    # finally the whole dataset of points in the domain and at th eboundary is concatenated in the following loop. 
    i = 0
    while i < 2*self.nDim**2:

      self.x = tf.concat([self.x, self.xBc[:, i:(self.nDim+i)]], 0)
      i= i+self.nDim
    
    #the tensor x is shuffled
    self.x = tf.random.shuffle(self.x)

    #yStep is initialized; will later be used to save models at different time step
    self.yStep =  tf.Variable(tf.zeros([self.totSample, 1 ]))

    #main model definition
    self.model = keras.Sequential([
        layers.Dense(60,activation = 'linear',  name="layer1"),
        layers.Dense(60,activation=tf.keras.layers.LeakyReLU(alpha=0.03),  name="layer2"),
        layers.Dense(60,activation=tf.keras.layers.LeakyReLU(alpha=0.03),    name="layer3"),
        layers.Dense(60,activation=tf.keras.layers.LeakyReLU(alpha=0.03),    name="layer4"),
        layers.Dense(1, activation ='linear', name="layer5"),
           
     ])
    
    self.model_t0 = tf.keras.models.clone_model(self.model)
    
    #auxiliary model definition
    self.auxiliaryModel = keras.Sequential([
        layers.Dense(30,activation = 'linear', name="layer1"),
        layers.Dense(30,activation=tf.keras.layers.LeakyReLU(alpha=0.03),  name="layer2"),
        layers.Dense(30,activation=tf.keras.layers.LeakyReLU(alpha=0.03),   name="layer3"),
        layers.Dense(30,activation=tf.keras.layers.LeakyReLU(alpha=0.03), name="layer4"),
        layers.Dense(1, activation = 'linear', name="layer5"), 
     ])

    
  def f_model_t0(self):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(self.x)
      y = self.model_t0(self.x)
    dy_dx = tape.gradient(y, self.x)
    return y, dy_dx

  # creation of the y vector togheter with its spatial derivatives
  def f_model(self):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(self.x)
      y = self.model(self.x)
    dy_dx = tape.gradient(y, self.x)
    return y, dy_dx 

  # creation of the v vector togheter with its spatial derivatives
  # v vector and its derivative is used to compute H_1 norm
  def f_auxiliaryModel(self):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(self.x)
      v = self.auxiliaryModel(self.x)
    dv_dx = tape.gradient(v, self.x)
    return v, dv_dx

  # evaluation of the true initial condition of the pde problem
  #this method will be used for computation of the IcLoss that can be found below
  def init_con(self):
      y0 = 1
      for j in range(self.nDim):
        y0 *= tf.sin(self.a[j] *self.x[:,j])
      y0_true = y0 
     return tf.reshape(y0_true, [self.totSample, 1])

  # evaluation of the true solution of the pde problem at time step dt (dt is updated during running of the code)
  def true_sol(self):
    sol = 1
    for j in range(self.nDim):
         sol *= tf.sin(self.a[j] *self.x[:,j])
    sol_true  = sol * tf.exp( -self.dt)
    return tf.reshape(sol_true, [self.totSample, 1])


  # training function of the neural network where the loss function is optimized
  def fit(self,  loss_fun, model, max_iter, lr, b1, b2): 
    opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1 = b1, beta_2 = b2)
    i = 1
    while True:       
        opt.minimize( loss_fun , var_list = model.trainable_weights)
        if opt.iterations > max_iter:
            print(f'stopping at max_iter={max_iter}')
            print('beta 1:', opt.beta_1.numpy(), 'beta 2:', opt.beta_2.numpy(), 'learning rate:', opt.learning_rate.numpy())
            return model.trainable_weights, loss_fun


  # evaluation of y
  def y(self):
    return self.f_model()[0]
  # evaluation of dy_dx
  def dy_dx(self):
    return self.f_model()[1]

  # evaluation of v
  def v(self):
    return self.f_auxiliaryModel()[0]
  # evaluation of dv_dx
  def dv_dx(self):
    return self.f_auxiliaryModel()[1]

  #evaluation of dy_dt : 
  def dy_dt(self):            
    return (self.y() - self.yStep) / self.timeStep 

  # evaluation of L2 product by use of MonteCarlo Integration method; returns l2 product or squared norm if u_var = v_var
  def L2_prod(self, u_var , v_var):
    prod = u_var * v_var 
    integrand = (tf.reduce_sum(prod, 0)* ((self.xMax-self.xMin)**self.nDim)) / (self.totSample)
    integrand2 = tf.reduce_sum(integrand)  #this is needed just in higer dimensions and the tf.sqrt can create problem during the optimization
    return integrand2

  def L2_prodBC(self, u_var , v_var):
    prod = u_var * v_var 
    integrand = (tf.reduce_sum(prod, 0)* (self.xMax-self.xMin)**(self.nDim-1)) / (self.nSampleBc)
    integrand2 =tf.reduce_sum(integrand)
    return integrand2

  # computation of Brezis Ekeland functional
  def BE_phi(self): 
    i = 0 
    self.BC = 0
    while i < 2* self.nDim**2 : 
      y_bc = self.model(self.xBc[:, i:self.nDim+i])
      self.BC = self.BC + self.L2_prodBC(y_bc , y_bc)
      i = i + self.nDim
      
    self.BC = self.lamb * self.BC
    integral =  self.tMax * ( self.c * 0.5 * self.L2_prod(self.dy_dx(), self.dy_dx())    # norm of the gradient
                            + self.c *  0.5 * self.H_1Norm() **2    #H-1 norm                                   
                            + self.L2_prod(self.dy_dt(), self.y()))  #product bewteen time derivative and y

    loss = integral   + self.BC
    return loss 

  def H_1Norm(self): 
      i = 0 
      BC = 0
      while i < 2* self.nDim**2 : 
        y_bc = self.auxiliaryModel(self.xBc[:, i:self.nDim+i])
        BC = BC + self.L2_prod(y_bc , y_bc)
        i = i + self.nDim
       
      BC = self.lamb * BC
      H_1 =   self.L2_prod( self.v(),  -self.dy_dt()/self.c ) /tf.sqrt(self.L2_prod(self.dv_dx(), self.dv_dx()) + BC)                                                                                                                 
      return - H_1

 # loss function to get the initial condition satisfied
  def IcLoss(self):
    IC_loss =  tf.reduce_sum((self.init_con() - self.y()) **2)/ (self.totSample)
    return IC_loss

solver = PDEsolver(pi, x_min, x_max, t_min, t_max, n_sample, n_sampleBc, n_dim, lamb, dt, time_step, a)
solver.f_model();
solver.f_model_t0();
solver.f_auxiliaryModel();

iterIC = 50000
learn_rateIC = 0.001

my_file = open( str(solver.nDim) + 'Dim.txt', 'w+')
my_file.write('nDim '+ str(solver.nDim)+ '; ' +' \n')
my_file.write('Initial Condition iter and learnRate: ' + str(iterIC) + ';  ' + str(learn_rateIC) + ' \n')
my_file.write('Initial condition optimization: ' +' \n')

optimization = solver.fit( solver.IcLoss , solver.model, iterIC, learn_rateIC , 0.9, 0.99)

MSE = tf.reduce_sum((solver.y()-solver.init_con())**2) /solver.totSample
Infinity_norm = tf.reduce_max(tf.abs(solver.init_con()- solver.y()))
abs_err_lInf = tf.reduce_max(tf.abs(solver.y()- solver.init_con()))
abs_err_l2 = solver.L2_prod(solver.y() - solver.init_con(), solver.y() - solver.init_con())
re_err_l2 = solver.L2_prod(solver.init_con() - solver.y()/ solver.init_con(), solver.init_con() - solver.y()/ solver.init_con() )
re_err_lInf = tf.reduce_max( tf.abs(solver.init_con() - solver.y()/ solver.init_con()))

my_file.write('MSE ' + str(MSE.numpy()) + '; ')
my_file.write('Infinity Norm ' + str(Infinity_norm.numpy())+'\n')
my_file.write('BE_phi: ' + str(solver.BE_phi().numpy()) +'; ' + 'H-1: ' + str(solver.H_1Norm().numpy()) +'\n' )
my_file.write('Relative error (squared): ' + str(solver.L2_prod(solver.init_con() - solver.y(), solver.init_con() - solver.y())/ solver.L2_prod(solver.init_con(), (solver.init_con() )))+'\n')
my_file.write('Abs_err_lInf:' + str(abs_err_lInf) + '; Abs_err:_l2: ' + str(abs_err_l2) +';' + '\n')
my_file.write('re_err_lInf:' + str(re_err_lInf) + '; re_err:_l2: ' + str(re_err_l2) +';' + '\n')

solver.f_model_t0()
solver.model_t0.set_weights(solver.model.get_weights())

# updating the values of yStep with model at time 0
solver.yStep = solver.f_model_t0()[0]

accuracy = tf.Variable(tf.transpose(tf.zeros([inner_iter, time_iter])))
accuracy_inner = tf.Variable(tf.transpose(tf.zeros([inner_iter, 1])))
BE_loss =  tf.Variable(tf.transpose(tf.zeros([inner_iter, time_iter])))
BE_loss_inner =  tf.Variable(tf.transpose(tf.zeros([inner_iter, 1])))
H1_loss =  tf.Variable(tf.transpose(tf.zeros([inner_iter, time_iter])))
H1_loss_inner =  tf.Variable(tf.transpose(tf.zeros([inner_iter, 1])))

abs_err_l2_inner = tf.Variable(tf.transpose(tf.zeros([inner_iter, 1])))
abs_err_lInf_inner = tf.Variable(tf.transpose(tf.zeros([inner_iter, 1])))
re_err_l2_inner = tf.Variable(tf.transpose(tf.zeros([inner_iter, 1])))
re_err_lInf_inner = tf.Variable(tf.transpose(tf.zeros([inner_iter, 1])))

cont1 = tf.Variable(tf.transpose(tf.zeros([inner_iter, 1])))
cont2 = tf.Variable(tf.transpose(tf.zeros([inner_iter, 1])))
cont3 = tf.Variable(tf.transpose(tf.zeros([inner_iter, 1])))
cont4 = tf.Variable(tf.transpose(tf.zeros([inner_iter, 1])))

abs_err_l2 =  tf.Variable(tf.transpose(tf.zeros([inner_iter, time_iter])))
abs_err_lInf = tf.Variable(tf.transpose(tf.zeros([inner_iter, time_iter])))
re_err_l2 =  tf.Variable(tf.transpose(tf.zeros([inner_iter, time_iter])))
re_err_lInf =  tf.Variable(tf.transpose(tf.zeros([inner_iter, time_iter])))

epoch = tf.Variable(tf.transpose([tf.linspace( 1, inner_iter*time_iter, time_iter * inner_iter)]))

my_file.write('Brezis_Ekeland optimization: '+ '\n')
my_file.write('Total Inner iter: ' +str(inner_iter)+'; total Time iter: ' + str(time_iter) +' \n')
#loop for updating time step 
solver.dt = 0.0001
for j in range(time_iter):
  #loop for getting better alternation of h-1 norm and of BE
  for i in range(inner_iter):
    my_file.write('Time: ' + str(solver.dt) +'\n' + 'Iteration n.:  ' + str(i) + '\n')    

    if i <= 5:
      learn_rate = 0.00001
    elif 5 < i <= 50:
      learn_rate = 0.000001
    elif 50 < i <= 120:
      learn_rate = 0.0000001
    elif 120 < i <= 140:
      learn_rate = 0.00000001
    elif 140< i <= 180 :
      learn_rate = 0.000000001
    elif 180 < i :
      learn_rate = 0.0000000001
  
    solver.fit(solver.BE_phi, solver.model, 50, learn_rate, 0.9, 0.99)
    solver.fit(solver.H_1Norm, solver.auxiliaryModel, 500, 0.00001, 0.9, 0.99)
   
    RMSE = tf.sqrt(tf.reduce_sum((solver.true_sol() - solver.y()).numpy())**2 / (solver.totSample))
    MSE = tf.reduce_sum(((solver.true_sol() - solver.y()).numpy())**2 ) / (solver.totSample )  
    MAE = tf.reduce_sum(tf.abs((solver.true_sol() - solver.y()).numpy())) / (solver.totSample)  
    
    my_file.write('RMSE: '+ str(RMSE.numpy()) +'; ' + 'MAE: ' +  str(MAE.numpy()) + '; '+  'MSE: ' + str(MSE.numpy()) + '\n')
    my_file.write('BE_phi: ' + str(solver.BE_phi().numpy()) +'; ' + 'H-1: ' + str(solver.H_1Norm().numpy()) +'\n' )

    accuracy[j,i].assign( MSE.numpy())
    accuracy_inner[0,i].assign(MSE.numpy())
    BE_loss[j,i].assign( solver.BE_phi().numpy())
    BE_loss_inner[0,i].assign( solver.BE_phi().numpy())
    H1_loss[j,i].assign( solver.H_1Norm().numpy())
    H1_loss_inner[0,i].assign( solver.H_1Norm().numpy())

    abs_err_lInf_inner[0,i].assign( tf.reduce_max(tf.abs(solver.y()- solver.true_sol())))
    abs_err_l2_inner[0,i].assign(solver.L2_prod(solver.y() - solver.true_sol(), solver.y() - solver.true_sol()))
    re_err_l2_inner[0,i].assign((solver.L2_prod(solver.true_sol() - solver.y(), solver.true_sol() - solver.y())/ solver.L2_prod(solver.true_sol(), solver.true_sol() )))
    re_err_lInf_inner[0,i].assign(tf.reduce_max( tf.abs(solver.true_sol() - solver.y()/ solver.true_sol())))

    abs_err_lInf[j,i].assign( tf.reduce_max(tf.abs(solver.y()- solver.true_sol())))
    abs_err_l2[j,i].assign(solver.L2_prod(solver.y() - solver.true_sol() , solver.y() - solver.true_sol()))
    re_err_l2[j,i].assign(solver.L2_prod(solver.true_sol() - solver.y(), solver.true_sol() - solver.y())/ solver.L2_prod(solver.true_sol(), (solver.true_sol() )))
    re_err_lInf[j,i].assign(tf.reduce_max( tf.abs(solver.true_sol() - solver.y()/ solver.true_sol())))
    
    cont1[0,i].assign( solver.L2_prod(solver.dy_dx(), solver.dy_dx()))
    cont2[0,i].assign( solver.H_1Norm() **2    )
    cont3[0,i].assign( solver.L2_prod(solver.dy_dt(), solver.y()))
    cont4[0,i].assign( solver.BC)

    my_file.write('Abs_err_lInf:' + str(abs_err_lInf_inner[0,i]) + '; Abs_err:_l2: ' + str(abs_err_l2_inner[0, i]) +';' + '\n')
    my_file.write('re_err_lInf:' + str(re_err_lInf_inner[0,i]) + '; re_err:_l2: ' + str(re_err_l2_inner[0, i]) +';' + '\n')
    my_file.write('Relative error (squared): ' + str(solver.L2_prod(solver.true_sol() - solver.y(), solver.true_sol() - solver.y())/ solver.L2_prod(solver.true_sol(), (solver.true_sol() ))))


    i= 0
    while i < 2* solver.nDim**2 : 
      y_bc = solver.model(solver.xBc[:, i:solver.nDim+i])
      norm = tf.reduce_sum(y_bc **2 ) / solver.nSampleBc
      print('MSE product of boundary of main model' , norm.numpy())
      my_file.write('MSE product of boundary of main model: '+ str(norm.numpy()) +'; '  + '\n')
      i = i + solver.nDim

    i= 0
    while i < 2* solver.nDim**2 : 
      y_bc = solver.auxiliaryModel(solver.xBc[:, i:solver.nDim+i])
      norm = tf.reduce_sum(y_bc **2 ) / solver.nSampleBc
      print('MSE of boundary of auxiliary model ' , norm.numpy())
      my_file.write('MSE product of boundary of auxiliary model: '+ str(norm.numpy()) +'; '  + '\n')
      i = i + solver.nDim

  

  

  np.savetxt( 'cont1' + str(solver.dt), cont1)  
  np.savetxt( 'cont2' + str(solver.dt), cont2)
  np.savetxt( 'cont3' + str(solver.dt), cont3) 
  np.savetxt( 'cont4' + str(solver.dt), cont4)  
    
  
  new_model = string + str(j)
  globals()[new_model] = tf.keras.models.clone_model(solver.model)   
  globals()[new_model].set_weights(solver.model.get_weights())
  solver.yStep = tf.reshape(globals()[new_model](solver.x), [solver.totSample,1 ])
  np.savetxt(new_model, globals()[new_model](solver.x))
 
  solver.dt = solver.dt + 0.0001

tfinale= solver.dt -0.0001
  
np.savetxt('accuracy', accuracy )
np.savetxt('BE', BE_loss)
np.savetxt('H-1 norm', H1_loss)
np.savetxt('abs err l2', abs_err_l2 )
np.savetxt('abs err lInf', abs_err_lInf)
np.savetxt('re err l2', re_err_l2 )
np.savetxt('re err lInf', re_err_lInf)
np.savetxt('x', solver.x)

elapsed_time_fl = (time.time() - start) 
print('time: ', elapsed_time_fl)
my_file.write('Total time: ' + str(elapsed_time_fl) + '\n') 
my_file.close()

