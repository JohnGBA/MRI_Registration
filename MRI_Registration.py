import numpy as np
import os
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import pickle
import random
import scipy.misc
from scipy import *
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
import re
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import sys



def delete_NaN():
  #img_=nib.load('C:/Projet_2A/bdd'+'/allenBrainAtlas.nii')
  #/media/johnathan/Windows/Users/abcd/Desktop/2020/codes for Github/Python ML
  img_=nib.load('/home/johnathan/Desktop/2020/Python ML/bdd'+'/CTRL_alc_M1_t2_1.nii')
  img= img_.get_fdata()
  copy=img
  C=0
  
  for i in range(165):
    for j in range(230):
      for k in range(135):
        if np.isnan(img[i,j,k]) :
          C=C+1
          copy[i,j,k]=0
  #joblib.dump(copy,'C:/Projet_2A/bdd'+'/atlas_sans_NaN.saved')
  joblib.dump(copy,'/home/johnathan/Desktop/2020/Python ML/bdd'+'/CTRL_alc_M1_t2_1_sans_NaN.saved')
  return(C)

def rotate(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2) #largeur puis hauteur
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
  
def translation(image,tx,ty):
  rows,cols= image.shape
  M = np.float32([[1,0,tx],[0,1,ty]])
  translatee = cv2.warpAffine(image,M,(cols,rows))
  return translatee

def getvalues(image): ##recupere les donnees contenues dans la colonne et la ligne du milieu de l image
  rows,cols= image.shape
  T=np.zeros((1,rows+cols))
  for i in range(rows):
    T[0,i]=image[i,cols//2]
  for j in range(cols):
    T[0,rows+j]=image[rows//2,j]
  return T

def add_noise2(vect1D,mag,size):
  vect1D+=np.random.uniform(-1*mag, 1*mag,size)
  return(vect1D)

def create_bdd(path_img,path_bdd,txmin,txmax,Ntx,tymin,tymax,Nty,thetamin,thetamax,Ntheta):

  ## definition des pas de translations et rotation
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
  
  ##chargement de l image nii depuis un fichier pickled
  
  img_=joblib.load(path_img)
  img=img_.T[120]

  ## creation de set ( la matrice qui contiendra les 2 lignes de pixels ) et labels ( matrice qui contient les translations et rotations associes
  set_size=(Ntx+1)*(Nty+1)*(Ntheta+1)
  print('setsize='+str(set_size))
  rows,cols=img.shape
  set=np.zeros((set_size,rows+cols))
  labels=np.zeros((set_size,3))

  compteur=0

  ##On va maintenant creer une combinaison de translations et rotation pour creer la BDD
  for X in range(0,Ntx+1):
    for Y in range(0,Nty+1):
      for O in range(0,Ntheta+1):

        copy=img #on cree une copie pour ne pas modifier img de base

        ##on applique la transformation et on enregistre uniquement la colonne et ligne du milieu de l img transformee

      
        temporaire=translation(rotate(copy,thetamin+O*pas_theta),txmin+X*pas_tx,tymin+Y*pas_ty) #contient l image transformee
        
        set[compteur,:]=getvalues(temporaire) #on nourrit la matrice d apprentissage ..
        
        #.. ainsi que la matrice de label
        labels[compteur,0]=txmin+X*pas_tx
        labels[compteur,1]=tymin+Y*pas_ty
        labels[compteur,2]=thetamin+O*pas_theta
        
        compteur+=1

  #on enregistre les deux matrices sous format pickle
  joblib.dump(set,path_bdd+'/bdd606060CRTL.saved')
  joblib.dump(labels,path_bdd+'/labels606060CRTL.saved')
  
#exemple pour appeler la fonction :
#create_bdd('C:/Projet_2A/Projet_2A_random_forest/bdd/CTRL_alc_M1_t2_1_onto_allen_sans_NaN.saved','C:/Projet_2A/Projet_2A_random_forest/bdd/',-30,30,60,0,0,1,0,0,1)
#create_bdd('/home/johnathan/Desktop/2020/Python ML/bdd/CTRL_alc_M1_t2_1_sans_NaN.saved','/home/johnathan/Desktop/2020/Python ML/Projet_2A_random_forest/bdd',-30,30,60,-30,30,60,-30,30,60)
  

def get_fit(path_bdd,path_labels): #construit le modele a partir des data et labels


  #on importe les donnees precedemment creees
  
  bdd=joblib.load(path_bdd)
  labels=joblib.load(path_labels)

  #on construit le modele a partir des data

  ### ### ### PARTIE A CHANGER EN CAS DE CHANGEMENT DE METHODE DE REGRESSION, ICI REGRESSION RANDOM FOREST. POUR D'AUTRES METHODES ALLER SUR SK-LEARN
  t1=time.time()
  regr = RandomForestRegressor(n_estimators=20, random_state=0)
  regr.fit(bdd,labels)
  t2=time.time()
  
  print('R^2=')
  print(regr.score(bdd,labels))

  ### ### ###

  #on enregistre le modele avec pickle
  #joblib.dump(regr,'C:/Projet_2A/Projet_2A_random_forest/bdd/fit_12.saved')
  joblib.dump(regr, '/home/johnathan/Desktop/2020/Python ML' + '/Projet_2A_random_forest/bdd/fit_CRTL.saved')
  print(t2-t1)


#exemple pour appeler la fonction :
#get_fit('C:/Projet_2A/Projet_2A_random_forest/bdd/bdd.saved','C:/Projet_2A/Projet_2A_random_forest/bdd/labels.saved')
#get_fit('/home/johnathan/Desktop/2020/Python ML/Projet_2A_random_forest/bdd/bdd3030120.saved','/home/johnathan/Desktop/2020/Python ML/Projet_2A_random_forest/bdd/labels3030120.saved')



def predict(path_regr,path_img_test,N_tests): #realise des predictions a partir du modele fourni

  x,y,theta=0,0,0
  V=0

  temps_moy=0
  erreur_moy_x=0
  erreur_moy_y=0
  erreur_moy_theta=0


  

  
  #on importe une image qui sera transformee aleatoirement pour realiser des tests
  img_=joblib.load(path_img_test)
  img=img_.T[120]

  #on importe le modele
  regr=joblib.load(path_regr)
  
  res=np.zeros((N_tests,3)) #contiendra le resultat de la prediction
  

  for i in range(N_tests):

    copy=img

    
    
    x=random.uniform(-30,30) #on choisit un coefficient de translation aleatoire
    y=random.uniform(-30,30)
    theta=random.uniform(-30,30)
    
    
    
    
    #on realise une prediction et on enregistre le resultat dans res
    t1=time.time()
    res[i,:]=regr.predict(getvalues(translation(rotate(copy,theta),x,y)))
    t2=time.time()
    
    
    deltax=abs(x-res[i,0])
    deltay=abs(y-res[i,1])
    deltatheta=abs(theta-res[i,2])

   

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
  

  

#predict('C:/Projet_2A/Projet_2A_random_forest/bdd/fit.saved','C:/Projet_2A/Projet_2A_random_forest/bdd/CTRL_alc_M1_t2_1_onto_allen_sans_NaN.saved',1)
#predict('/home/johnathan/Desktop/2020/Python ML/Projet_2A_random_forest/bdd/fit_12.saved','/home/johnathan/Desktop/2020/Python ML/Projet_2A_random_forest/bdd/CTRL_alc_M1_t2_1.nii_sans_NaN.saved',1)    


def noisy_predict(path_regr,path_img_test,N_tests,mag): #realise des predictions a partir du modele fourni en ajoutant du bruit uniforme d'amplitude mag

  x,y,theta=0,0,0
  V=0
  
  temps_moy=0
  erreur_moy_x=0
  erreur_moy_y=0
  erreur_moy_theta=0


  

  
  #on importe une image qui sera transformee aleatoirement pour realiser des tests
  img_=joblib.load(path_img_test)
  img=img_.T[120]
  rows,cols=img.shape
  
  #on importe le modele
  regr=joblib.load(path_regr)
  
  res=np.zeros((N_tests,3)) #contiendra le resultat de la prediction
  

  for i in range(N_tests):

    copy=img

    
    
    x=random.uniform(-30,30) #on choisit un coefficient de translation aleatoire
    y=random.uniform(-30,30)
    theta=random.uniform(-30,30)
    
    
    
    
    #on realise une prediction et on enregistre le resultat dans res
    t1=time.time()
    res[i,:]=regr.predict(add_noise2(getvalues(translation(rotate(copy,theta),x,y)),mag,rows+cols))
    t2=time.time()
    
    
    deltax=abs(x-res[i,0])
    deltay=abs(y-res[i,1])
    deltatheta=abs(theta-res[i,2])

   

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
  

def print_courbes(): #print des courbes qui affiche erreur et veracite en fonction de l'amplitude du bruit ajoute
  X=zeros((1,21))
  Y=zeros((1,21))
  T=zeros((1,21))
  V=zeros((1,21))
  A=zeros((1,21))
  for k in range(0,1050,50):
    #with suppress_stdout():
      #res=noisy_predict('C:/Projet_2A/Projet_2A_random_forest/bdd/fit_50.saved','C:/Projet_2A/Projet_2A_random_forest/bdd/CTRL_alc_M1_t2_1_onto_allen_sans_NaN.saved',1000,k)
      res=noisy_predict('/home/johnathan/Desktop/2020/Python ML' + '/Projet_2A_random_forest/bdd/fit_CRTL.saved', os.getcwd() + '/bdd/CTRL_alc_M1_t2_1_sans_NaN.saved',1000,k)
      V[0,k//50]=res[0]
      X[0,k//50]=res[1]
      Y[0,k//50]=res[2]
      T[0,k//50]=res[3]
      A[0,k//50]=k

  plt.figure(1)
  plt.plot(A[0,:],V[0,:], 'g')
  plt.xlabel('Amplitude du bruit')
  plt.ylabel('Taux veracite (pour 1000 tests)')
  plt.title('Taux veracite en fonction du bruit')
  plt.show()

  plt.figure(2)
  plt.plot(A[0,:],X[0,:], 'g')
  plt.plot(A[0,:],Y[0,:], 'r')
  plt.plot(A[0,:],T[0,:], 'b')
  plt.xlabel('Amplitude du bruit')
  plt.ylabel('Erreur en x(vert), y(rouge) et theta(bleu) (pour 1000 tests)')
  plt.title('Erreurs en fonction du bruit')
  plt.show()
    




                
        





