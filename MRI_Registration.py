import numpy as np
import os
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.externals import joblib
import time
from sklearn.ensemble import RandomForestRegressor

def delete_NaN():
  img_=nib.load(os.getcwd() + '/CTRL_alc_M1_t2_1.nii')
  img= img_.get_fdata()
  copy=img
  C=0 #number of NaN values in image.
  
  for i in range(165):
    for j in range(230):
      for k in range(135):
        if np.isnan(img[i,j,k]) :
          C=C+1
          copy[i,j,k]=0
          
  joblib.dump(copy, os.getcwd() + '/CTRL_alc_M1_t2_1_sans_NaN.saved')
  return(C)

def rotate(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2) #width then height.
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
  
def translation(image,tx,ty):
  rows,cols= image.shape
  M = np.float32([[1,0,tx],[0,1,ty]])
  translated = cv2.warpAffine(image,M,(cols,rows))
  return translated

def get_values(image): ##retrieve the data contained in the middle column and row of the image.
  rows,cols= image.shape
  T=np.zeros((1,rows+cols))
  for i in range(rows):
    T[0,i]=image[i,cols//2]
  for j in range(cols):
    T[0,rows+j]=image[rows//2,j]
  return T

def add_noise(vect1D,mag,size):
  vect1D+=np.random.uniform(-1*mag, 1*mag,size)
  return(vect1D)

def create_dataset(path_img,path_dataset,txmin,txmax,Ntx,tymin,tymax,Nty,thetamin,thetamax,Ntheta):
  ## definition of translation and rotation step
  if Ntx!=0:
    pas_tx=float(txmax-txmin)/float(Ntx)
  else:
    pas_tx=0
  if Nty!=0:
    pas_ty=float(tymax-tymin)/float(Nty)
  else :
    pas_ty=0
  if Ntheta!=0:
    pas_theta=float(thetamax-thetamin)/float(Ntheta)
  else:
    pas_theta=0   
  
  ##loads a '.nii' image
  img_=joblib.load(path_img)
  img=img_.T[120]

  ## creation of dataset ( matrice containing the 2 extracted lines of pixels ) and labels ( associated translations and rotations).
  set_size=(Ntx+1)*(Nty+1)*(Ntheta+1)
  print('setsize='+str(set_size))
  rows,cols=img.shape
  set=np.zeros((set_size,rows+cols))
  labels=np.zeros((set_size,3))

  compteur=0
  ##Combinations of all rotation and translation to form the database.
  for X in range(0,Ntx+1):
    for Y in range(0,Nty+1):
      for O in range(0,Ntheta+1):

        copy=img #copies the input image to avoid modifying it.

        temporaire=translation(rotate(copy,thetamin+O*pas_theta),txmin+X*pas_tx,tymin+Y*pas_ty) #returns transformed image.
        set[compteur,:]=get_values(temporaire) #the middle column and row of the transformed image is fed to the dataset.
        
        #Label matrix is populated.
        labels[compteur,0]=txmin+X*pas_tx
        labels[compteur,1]=tymin+Y*pas_ty
        labels[compteur,2]=thetamin+O*pas_theta
        
        compteur+=1

  #The two matrices are stored in disk
  joblib.dump(set,path_dataset+'/dataset606060CRTL.saved')
  joblib.dump(labels,path_dataset+'/labels606060CRTL.saved')
  
#Example on how to call the function: 
#create_dataset(os.getcwd() + '/dataset/CTRL_alc_M1_t2_1_sans_NaN.saved',os.getcwd() + '/dataset',-30,30,60,-30,30,60,-30,30,60)

def get_fit(path_dataset,path_labels): #Generates the model from the the data and labels

  dataset=joblib.load(path_dataset)
  labels=joblib.load(path_labels)

  ### ### ### PART TO CHANGE IN THE EVENT OF A CHANGE IN REGRESSION METHOD. NOW RANDOM FOREST IS USED. FOR OTHER METHODS, REFER TO SK-LEARN
  t1=time.time()
  regr = RandomForestRegressor(n_estimators=20, random_state=0)
  regr.fit(dataset,labels)
  t2=time.time()
  
  print('R^2=')
  print(regr.score(dataset,labels))
  ### ### ###

  #The model is stored in disk.
  joblib.dump(regr, os.getcwd() + '/random_forest/dataset/fit_CRTL.saved')
  print(t2-t1)

#Example on how to call the function: 
#get_fit(os.getcwd() + '/random_forest/dataset/dataset606060.saved', os.getcwd() + '/random_forest/dataset/labels606060.saved')

def predict(path_regr,path_img_test,N_tests): #Perfoms the predictions based on the provided model.

  x,y,theta=0,0,0
  V=0 #number of good predictions.

  temps_moy=0
  erreur_moy_x=0
  erreur_moy_y=0
  erreur_moy_theta=0

  #An image that will be randomnly transformed is loaded.
  img_=joblib.load(path_img_test)
  img=img_.T[120]

  #The model is loaded.
  regr=joblib.load(path_regr)
  
  prediction=np.zeros((N_tests,3)) 
  
  for i in range(N_tests):

    copy=img

    x=random.uniform(-30,30) 
    y=random.uniform(-30,30)
    theta=random.uniform(-30,30)
    
    #A prediction of the transformation is made and stored
    t1=time.time()
    prediction[i,:]=regr.predict(get_values(translation(rotate(copy,theta),x,y)))
    t2=time.time()
    
    deltax=abs(x-prediction[i,0])
    deltay=abs(y-prediction[i,1])
    deltatheta=abs(theta-prediction[i,2])

    erreur_moy_x+=deltax
    erreur_moy_y+=deltay
    erreur_moy_theta+=deltatheta
    temps_moy+=t2-t1
      
    if (deltatheta)<1 and np.absolute(deltax + deltay*1j)<np.sqrt(2):
      V+=1
    
    #print(deltax)
    #print(deltay)
    #print(deltatheta)
  erreur_moy_x=erreur_moy_x/N_tests
  erreur_moy_y=erreur_moy_y/N_tests
  erreur_moy_theta=erreur_moy_theta/N_tests
  temps_moy=temps_moy/N_tests
  print(V,N_tests,float(V)/float(N_tests),'erreurs en x, y et theta '+str(erreur_moy_x)+', '+str(erreur_moy_y)+', '+str(erreur_moy_theta),'temps moyen de prediction '+str(temps_moy))
  
#Example on how to call the function: 
#predict(os.getcwd() + 'random_forest/dataset/fit.saved','os.getcwd() + '/random_forest/dataset/CTRL_alc_M1_t2_1.nii_sans_NaN.saved',1)    

def noisy_predict(path_regr,path_img_test,N_tests,mag): #realise des predictions a partir du modele fourni en ajoutant du bruit uniforme d'amplitude mag

  x,y,theta=0,0,0
  V=0 #number of good predictions.
  
  temps_moy=0
  erreur_moy_x=0
  erreur_moy_y=0
  erreur_moy_theta=0

  #An image that will be randomnly transformed is loaded.
  img_=joblib.load(path_img_test)
  img=img_.T[120]
  rows,cols=img.shape
  
  #The model is loaded.
  regr=joblib.load(path_regr)
  
  prediction=np.zeros((N_tests,3)) 
  
  for i in range(N_tests):

    copy=img

    x=random.uniform(-30,30) 
    y=random.uniform(-30,30)
    theta=random.uniform(-30,30)
    
    #A prediction is made and stored. The addnoise function is used.
    t1=time.time()
    prediction[i,:]=regr.predict(add_noise(get_values(translation(rotate(copy,theta),x,y)),mag,rows+cols))
    t2=time.time()

    deltax=abs(x-prediction[i,0])
    deltay=abs(y-prediction[i,1])
    deltatheta=abs(theta-prediction[i,2])

    erreur_moy_x+=deltax
    erreur_moy_y+=deltay
    erreur_moy_theta+=deltatheta
    temps_moy+=t2-t1
      
    if (deltatheta)<1 and np.absolute(deltax + deltay*1j)<np.sqrt(2):
      V+=1
    
    #print(deltax)
    #print(deltay)
    #print(deltatheta)
  erreur_moy_x=erreur_moy_x/N_tests
  erreur_moy_y=erreur_moy_y/N_tests
  erreur_moy_theta=erreur_moy_theta/N_tests
  temps_moy=temps_moy/N_tests
  return([float(V)/float(N_tests),erreur_moy_x,erreur_moy_y,erreur_moy_theta])
  print(V,N_tests,float(V)/float(N_tests),'erreurs en x, y et theta '+str(erreur_moy_x)+', '+str(erreur_moy_y)+', '+str(erreur_moy_theta),'temps moyen de prediction '+str(temps_moy))

def draw_plots(): #print curves which displays error and accuracy as a function of the amplitude of the noise added
  X=np.zeros((1,21))
  Y=np.zeros((1,21))
  T=np.zeros((1,21))
  V=np.zeros((1,21))  #Right predictions
  A=np.zeros((1,21))  #Noise 
  for k in range(0,1050,50):
    #with suppress_stdout():
      res=noisy_predict(os.getcwd() + '/random_forest/dataset/fit_CRTL.saved', os.getcwd() + '/dataset/CTRL_alc_M1_t2_1_sans_NaN.saved',1000,k)
      V[0,k//50]=res[0]
      X[0,k//50]=res[1]
      Y[0,k//50]=res[2]
      T[0,k//50]=res[3]
      A[0,k//50]=k

  plt.figure(1)
  plt.plot(A[0,:],V[0,:], 'g')
  plt.xlabel('Amplitude of noise')
  plt.ylabel('Sucess rate (for a 1000 tests)')
  plt.title('Sucess rate in function of noise')
  plt.show()

  plt.figure(2)
  plt.plot(A[0,:],X[0,:], 'g')
  plt.plot(A[0,:],Y[0,:], 'r')
  plt.plot(A[0,:],T[0,:], 'b')
  plt.xlabel('Amplitude of noise')
  plt.ylabel('Error in x(green), y(red) and theta(blue) (for a 1000 tests)')
  plt.title('Errors in function of noise')
  plt.show()
    




                
        





