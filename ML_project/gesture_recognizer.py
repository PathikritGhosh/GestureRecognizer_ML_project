from __future__ import division
import json
import time
import pickle
import gzip
import os
import glob
import random
import numpy as np
import skimage
import pandas as pd
import sklearn as sk
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xgboost as xgb
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from skimage import io,color
from sklearn.svm import SVC
from functools import partial
from itertools import repeat
from PIL import Image, ImageDraw

class GestureRecognizer(object):

	"""class to perform gesture recognition"""

	def __init__(self,data_directory):
		#self.multi_clf_rf=RandomForestClassifier(n_estimators=300,n_jobs=-1,max_features='sqrt',max_depth=20,min_samples_leaf=6)
		#self.multi_clf_xgb=xgb.XGBClassifier(nthread=mp.cpu_count(),max_depth=7,subsample=0.7,min_child_weight=1.8,objective = 'multi:softprob')
		self.multi_clf_svm=SVC(kernel='linear',probability=True,C=1.5)
		self.bin_clf=RandomForestClassifier(n_estimators=500,n_jobs=-1)	# gesture recognizer
		self.user_list=['user_3','user_4','user_5','user_6','user_7','user_9','user_10','user_11','user_12','user_13','user_14','user_15','user_16','user_17','user_18','user_19']
		self.path=data_directory
		self.resolution=130
		self.pyr_list=[100,110,120,130,140]
		self.imagex=320
		self.imagey=240
		self.trainX={}
		self.trainY={}
		self.multi_trainX={}
		self.multi_trainY={}

	def hogify(self,t_data):
		t_data=hog(t_data,feature_vector=True,pixels_per_cell=(8,8),cells_per_block=(2,2),visualise=False,transform_sqrt = True)
		return t_data

	def intersection(self,box1,box2):
		if(box1[0]>box2[2]):
			return 0
		elif(box2[0]>box1[2]):
			return 0
		elif(box1[1]>box2[3]):
			return 0
		elif(box2[1]>box2[3]):
			return 0
		t_xa=max(box1[0],box2[0])
		t_xb=min(box1[2],box2[2])
		t_ya=max(box1[1],box2[1])
		t_yb=min(box1[3],box2[3])
		t_area=(t_xb-t_xa+1)*(t_yb-t_ya+1)
		return t_area

	def union(self,box1,box2):
		t_area_a=(box1[3]-box1[1]+1)*(box1[2]-box1[0]+1)
		t_area_b=(box2[3]-box2[1]+1)*(box2[2]-box2[0]+1)
		t_area=t_area_a+t_area_b-self.intersection(box1,box2)
		return t_area

	def iou(self,box1,box2):
		t_num=self.intersection(box1,box2)
		t_den=self.union(box1,box2)
		return t_num/t_den


	def bin_user_image_load(self,user,step):
		if(user not in self.user_list):
			return
		print (user+':image loading started')
		t_df=pd.read_csv(self.path+user+'/'+user+"_loc.csv",index_col="image")
		t_trainX=[]
		t_trainY=[]
		for filename in glob.glob(self.path+user+'/*.jpg'):
			t_image=color.rgb2gray(io.imread(filename))
			tt_image=t_image
			t_index=filename.split('/')
			t_index=t_df.loc[t_index[-2]+'/'+t_index[-1]]
			t_image=t_image[t_index[1]:t_index[3],t_index[0]:t_index[2]]
			t_image=skimage.transform.resize(t_image,(self.resolution,self.resolution),mode='edge',preserve_range=True)
			t_trainX.append(self.hogify(t_image))
			t_trainY.append([1])

			t_data=[]
			for i in range(0,self.imagex-self.resolution,step):
				for j in range(0,self.imagey-self.resolution,step):
					t_iou=self.iou([t_index[0],t_index[1],t_index[2],t_index[3]],[i,j,i+self.resolution,j+self.resolution])
					if(t_iou < 0.7):
						t_data.append(tt_image[j:j+self.resolution,i:i+self.resolution])

			count=3
			for i in range(0,count):
				t_index=random.randint(0,len(t_data)-1)
				t_trainX.append(self.hogify(t_data[t_index]))
				t_trainY.append([0])
				del t_data[t_index]
		return [user,t_trainX,t_trainY]

	def bin_image_load(self,p_user_list,step):
		pool=mp.Pool(processes=mp.cpu_count())
		t_list=pool.starmap(self.bin_user_image_load,zip(p_user_list,repeat(step)))
		pool.close()
		pool.join()
		for x in t_list:
			self.trainX[x[0]]=x[1]
			self.trainY[x[0]]=x[2]
		print ("image loading done")

	def comb(self,t_list):
		t_trainX=[]
		t_trainY=[]
		for i in t_list:
			t_trainX=t_trainX+self.trainX[i]
			t_trainY=t_trainY+self.trainY[i]
		return [np.array(t_trainX),np.array(t_trainY)]

	def user_hard_mining(self,user,t_clf_obj,step=20):
		if(user not in self.user_list):
			return
		print ('mining:'+user)
		t_clf=pickle.loads(t_clf_obj)
		t_df=pd.read_csv(self.path+user+'/'+user+"_loc.csv",index_col="image")
		t_data=[]
		for filename in glob.glob(self.path+user+'/*.jpg'):
			t_image=color.rgb2gray(io.imread(filename))
			t_index=filename.split('/')
			t_index=t_df.loc[t_index[-2]+'/'+t_index[-1]]
			for i in range(0,self.imagex-self.resolution,step):
				for j in range(0,self.imagey-self.resolution,step):
					t_iou=self.iou([t_index[0],t_index[1],t_index[2],t_index[3]],[i,j,i+self.resolution,j+self.resolution])
					if(t_iou < 0.6):
						t_data.append(self.hogify(t_image[j:j+self.resolution,i:i+self.resolution]))
		print ('data_loaded'+user)
		np.random.shuffle(t_data)
		counter=1
		max_counter=200
		t_val=t_clf.predict_proba(t_data)
		index=[]
		for i in range(0,t_val.shape[0]):
			if(t_val[i,1]>0.6):
				index.append(i)
				counter=counter+1
				if(counter > max_counter):
					break
		print (user+"done")
		t_data=[t_data[i] for i in index]
		t_datay=[[0] for x in t_data]
		return [user,t_data,t_datay]


	def neg_hard_mining(self,p_user_list,iterations=2):
		while(iterations):
			t_clf=RandomForestClassifier(n_estimators=100,n_jobs=-1)
			t_train_val=self.comb(p_user_list)
			t_clf.fit(t_train_val[0],t_train_val[1].reshape(t_train_val[1].shape[0],))
			print (len(t_train_val[0]))
			print (len(t_train_val[1]))
			t_clf_obj=pickle.dumps(t_clf)
			pool=mp.Pool(processes=mp.cpu_count())
			t_list=pool.starmap(self.user_hard_mining,zip(p_user_list,repeat(t_clf_obj)))
			pool.close()
			pool.join()
			for x in t_list:
				self.trainX[x[0]]=self.trainX[x[0]]+x[1]
				self.trainY[x[0]]=self.trainY[x[0]]+x[2]
			iterations=iterations-1

	def bin_train(self,p_user_list,step=30):
		self.bin_image_load(p_user_list,step)
		self.neg_hard_mining(p_user_list)
		t_train_val=self.comb(p_user_list)
		trainX=t_train_val[0]
		trainY=t_train_val[1].reshape(t_train_val[1].shape[0],)
		self.bin_clf.fit(trainX,trainY)

	def train(self, train_list):
		self.bin_train(train_list)
		self.multi_train(train_list)

	def save_model(self, **params):
		"""
			save your GestureRecognizer to disk.
		"""
		self.version = params['version']
		self.author = params['author']
		file_name = params['name']
		pickle.dump(self,gzip.open(file_name,'wb'))


	def alt_image_pyramid_level(self,t_size,t_image,step=30):
		#returns highest probability bounding box in form of [[x1,y1,x2,y2]],output_probability
		t_bin_clf=self.bin_clf
		t_box=[]
		tt_box=[]
		t_arr=[]
		for i in range(0,self.imagex-t_size,step):
			for j in range(0,self.imagey-t_size,step):
				t_arr.append(t_image[j:j+t_size,i:i+t_size])
				t_box.append([i,j])
		for i in range(0,len(t_arr)):
			t_arr[i]=self.hogify(skimage.transform.resize(t_arr[i],(self.resolution,self.resolution),mode='edge',preserve_range=True))
		t_outp=t_bin_clf.predict_proba(np.array(t_arr))
		t_index=np.argsort(t_outp[:,1])
		t_index=t_index[::-1][0]
		tt_box.append([t_box[t_index][0],t_box[t_index][1],t_box[t_index][0]+t_size,t_box[t_index][1]+t_size])
		return (tt_box,t_outp[t_index][1])

	def nms(self,boxes,t_proba,overlapThresh=0.3):
		#return [[x1,y1,x2,y2],proba]
		if (len(boxes) == 0):
			return []
		if (boxes.dtype.kind == 'i'):
			boxes=boxes.astype('float')
		pick=[]
		x1=boxes[:,0]
		y1=boxes[:,1]
		x2=boxes[:,2]
		y2=boxes[:,3]
		area=(x2-x1+1)*(y2-y1+1)
		idxs=np.argsort(t_proba)
		while (len(idxs)>0):
			last=len(idxs)-1
			i=idxs[last]
			pick.append(i)
			xx1=np.maximum(x1[i],x1[idxs[:last]])
			yy1=np.maximum(y1[i],y1[idxs[:last]])
			xx2=np.minimum(x2[i],x2[idxs[:last]])
			yy2=np.minimum(y2[i],y2[idxs[:last]])
			width=np.maximum(0,xx2-xx1+1)
			height=np.maximum(0,yy2-yy1+1)
			overlap=(width*height)/area[idxs[:last]]
			idxs=np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
		new_box=boxes[pick].astype('int')
		t_arr=[]
		for x in new_box[0]:
			t_arr.append(x)
		return [t_arr,t_proba[-1]]

	def nms_user(self,boxes,overlapThresh=0.5):
		#return [[x1,y1,x2,y2],...]
		if (len(boxes) == 0):
			return []
		if (boxes.dtype.kind == 'i'):
			boxes=boxes.astype('float')
		pick=[]
		x1=boxes[:,0]
		y1=boxes[:,1]
		x2=boxes[:,2]
		y2=boxes[:,3]
		area=(x2-x1+1)*(y2-y1+1)
		idxs=np.argsort(y2)
		while (len(idxs)>0):
			last=len(idxs)-1
			i=idxs[last]
			pick.append(i)
			xx1=np.maximum(x1[i],x1[idxs[:last]])
			yy1=np.maximum(y1[i],y1[idxs[:last]])
			xx2=np.minimum(x2[i],x2[idxs[:last]])
			yy2=np.minimum(y2[i],y2[idxs[:last]])
			width=np.maximum(0,xx2-xx1+1)
			height=np.maximum(0,yy2-yy1+1)
			overlap=(width*height)/area[idxs[:last]]
			idxs=np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
		t_box=boxes[pick].astype('int')
		tt_box=[]
		for x in t_box:
			tt_box.append(x)
		return tt_box

	def image_pyramid_level(self,t_size,t_image,step=20):
		#return [[x1,y1,x2,y2],proba]
		t_bin_clf=self.bin_clf
		t_box=[]
		tt_box=[]
		t_arr=[]
		for i in range(0,self.imagex-t_size,step):
			for j in range(0,self.imagey-t_size,step):
				t_arr.append(t_image[j:j+t_size,i:i+t_size])
				t_box.append([i,j])
		for i in range(0,len(t_arr)):
			t_arr[i]=self.hogify(skimage.transform.resize(t_arr[i],(self.resolution,self.resolution),mode='edge',preserve_range=True))
		t_outp=t_bin_clf.predict_proba(np.array(t_arr))
		t_arg=np.argsort(t_outp[:,1])
		t_arg=t_arg[::-1]
		tt_arg=[]
		for i in range(0,len(t_arg)):
			tt_arg.append([t_outp[t_arg[i],1],t_arg[i]])
		counter=1
		max_counter=4
		for x in tt_arg:
			if(x[0] >= 0.5):
				tt_box.append([t_box[x[1]][0],t_box[x[1]][1],t_box[x[1]][0]+t_size,t_box[x[1]][1]+t_size])
				counter=counter+1
			else:
				break
			if(counter > max_counter):
				break
		tt_box=np.array(tt_box)
		if(len(tt_box)==0):
			return []
		return self.nms_user(tt_box)

	def probify(self,t_box,t_image):
		t_prob=[]
		for x in t_box:
			#print (t_box)
			#print (x)
			tt_image=self.boxify(t_image,x).reshape(1,-1)
			t_prob.append(self.bin_clf.predict_proba(tt_image)[0][1])
		return t_prob


	def image_pyramid(self,t_image):
		t_image=color.rgb2gray(t_image)
		t_fin_boxes=[]
		pool=mp.Pool(processes=mp.cpu_count())
		t_list=pool.starmap(self.image_pyramid_level,zip(self.pyr_list,repeat(t_image)))
		pool.close()
		pool.join()

		t_proba_list=[]
		for t_val in t_list:
			if(len(t_val)==0):
				continue
			t_fin_boxes=t_fin_boxes+t_val
			t_proba_list=t_proba_list+self.probify(t_val,t_image)
		if(t_fin_boxes):	
			t_fin_boxes=[self.nms(np.array(t_fin_boxes),np.array(t_proba_list))[0]]
		if(len(t_fin_boxes)==0):
			pool=mp.Pool(processes=1)
			t_list=pool.starmap(self.alt_image_pyramid_level,zip(self.pyr_list,repeat(t_image)))
			pool.close()
			pool.join()
			t_ref=0
			for (t_val,t_proba) in t_list:
				if(t_proba > t_ref):
					t_ref=t_proba
					t_fin_boxes=t_val
		return np.array(t_fin_boxes[0]).astype('int')

	def get_image(self,t_dir):
		"""t_dir of the form user_17/A0.jpg"""
		t_name=self.path+t_dir
		t_image=io.imread(t_name)
		return t_image

	def visualize(self,t_dir):
		self=pickle.load(gzip.open('gr.pkl.gz','rb'))
		t_image=self.get_image(t_dir)
		t_box=[self.image_pyramid(t_image)]
		print (t_box)
		io.imsave('/home/ghosh/Desktop/'+'img.jpg',t_image)
		temp=Image.open('/home/ghosh/Desktop/'+'img.jpg')
		draw = ImageDraw.Draw(temp)
		for x in t_box:
			y=[(x[0],x[1]),(x[2],x[3])]
			draw.rectangle(y,outline='red')
		print ("doneee!")
		del draw
		temp.save('/home/ghosh/Desktop/imgi.jpg')

	def multi_comb(self,t_list):
		t_trainX=[]
		t_trainY=[]
		for i in t_list:
			t_trainX=t_trainX+self.multi_trainX[i]
			t_trainY=t_trainY+self.multi_trainY[i]
		return [np.array(t_trainX),np.array(t_trainY)]

	def multi_user_image_load(self,user):
		if(user not in self.user_list):
			return
		print (user+'->multi:image loading started')
		t_df=pd.read_csv(self.path+user+'/'+user+"_loc.csv",index_col="image")
		t_trainX=[]
		t_trainY=[]
		for filename in glob.glob(self.path+user+'/*.jpg'):
			t_image=color.rgb2gray(io.imread(filename))
			tt_image=t_image
			t_index=filename.split('/')
			t_label=t_index[-1].split('.')[0][0]
			t_index=t_df.loc[t_index[-2]+'/'+t_index[-1]]
			t_image=t_image[t_index[1]:t_index[3],t_index[0]:t_index[2]]
			t_image=skimage.transform.resize(t_image,(self.resolution,self.resolution),mode='edge',preserve_range=True)
			t_trainX.append(self.hogify(t_image))
			t_trainY.append([t_label])
		return [user,t_trainX,t_trainY]

	def multi_image_load(self,p_user_list):
		pool=mp.Pool(processes=mp.cpu_count())
		t_list=pool.map(self.multi_user_image_load,p_user_list)
		pool.close()
		pool.join()
		for x in t_list:
			self.multi_trainX[x[0]]=x[1]
			self.multi_trainY[x[0]]=x[2]
		print ("image loading done")

	def multi_train(self,p_user_list):
		self.multi_image_load(p_user_list)
		t_train_val=self.multi_comb(p_user_list)
		trainX=t_train_val[0]
		trainY=t_train_val[1].reshape(t_train_val[1].shape[0],)
		print ('training started_SVM')
		self.multi_clf_svm.fit(trainX,trainY)
		print ('training done')

	def boxify(self,t_full_image,t_box):
		tt_image=color.rgb2gray(t_full_image)
		tt_image=tt_image[t_box[1]:t_box[3],t_box[0]:t_box[2]]
		tt_image=skimage.transform.resize(tt_image,(self.resolution,self.resolution),mode='edge',preserve_range=True)
		tt_image=self.hogify(np.array(tt_image))
		return tt_image

	def multi_predict(self,t_full_image,t_box):
		t_image=self.boxify(t_full_image,t_box).reshape(1,-1)
		t_b=self.multi_clf_svm.predict_proba(t_image)
		l_b=self.multi_clf_svm.classes_[np.argsort(t_b)[:,-5:]][0][::-1]
		return l_b

	def recognize_gesture(self, image):
		"""
			image : a 320x240 pixel RGB image in the form of a numpy array
			This function should locate the hand and classify the gesture.

			returns : (position, labels)
			position : a tuple of (x1,y1,x2,y2) coordinates of bounding box
					   x1,y1 is top left corner, x2,y2 is bottom right
			labels : a list of top 5 character predictions
						eg - ['A', 'B', 'C', 'D', 'E']
		"""
		#localization
		t_pos=self.image_pyramid(image)
		position=tuple(t_pos)

		time.sleep(0.1) # simulate computation delay

		#classification
		labels=self.multi_predict(image,t_pos)
		return position, labels



	@staticmethod
	def load_model(**params):
		"""
			Returns a saved instance of GestureRecognizer.
			load your trained GestureRecognizer from disk with provided params
		"""
		file_name = params['name']
		return pickle.load(gzip.open(file_name,'rb'))

	def bin_test(self):
		train_list=['user_3', 'user_4', 'user_5','user_6','user_7','user_9','user_10','user_11','user_12','user_13','user_14','user_15']
		test_list=['user_17','user_18','user_16','user_19']
		self.bin_image_load(self.user_list,30)
		train=self.comb(train_list)
		trainX=train[0]
		trainY=train[1].reshape(train[1].shape[0],)
		test=self.comb(test_list)
		testX=test[0]
		testY=test[1].reshape(test[1].shape[0],)
		list1=[150,200,250,300,350,400,450,500]
		list2=[0]
		list3=[0]
		t_clf=RandomForestClassifier(n_jobs=-1,n_estimators=100,random_state=1)
		t_clf.fit(trainX,trainY)
		print (t_clf.score(testX,testY))
		for x in list1:
			for y in list2:
				for z in list3:
					t_clf=RandomForestClassifier(n_jobs=-1,n_estimators=x,random_state=1)
					t_clf.fit(trainX,trainY)
					print (x)
					print (t_clf.score(testX,testY))





if __name__ == "__main__":
	gr = GestureRecognizer('/home/ghosh/Desktop/ML/dataset/')
	gr.train(['user_3', 'user_4', 'user_5','user_6','user_7','user_9','user_10','user_11','user_12','user_13','user_14','user_15','user_16','user_17','user_18','user_19'])

	# # # Now the GestureRecognizer is trained. We can save it to disk

	gr.save_model(	name = "gr.pkl.gz",
					version = "0.0.1",
					author = "pathikrit"
				 )
	# # # Now the GestureRecognizer is saved to disk
	# # # It will be dumped as a compressed .gz file

	new_gr = GestureRecognizer.load_model(name = "gr.pkl.gz") # automatic dict unpacking
	print (new_gr.version)
	#gr.visualize('user_17/A1.jpg')
