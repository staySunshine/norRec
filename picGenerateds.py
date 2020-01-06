# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:03:21 2019

@author: Administrator
"""
import numpy as np
from PIL import Image
import os
N = 28
threshold = 34
ROOT_PATH="./testPic"
OUT_PATH="./cross_data_jpg/cross_train_jpg_60000"

def gray_pic(array,picName):
    image = Image.fromarray(array) 
    image.save(os.path.join(OUT_PATH, picName))

def picRotate(picName,picName2,degree):
    img = Image.open(os.path.join(ROOT_PATH, picName))
    img = img.rotate(degree) 
    img = img.resize((N,N), Image.ANTIALIAS)
    img = img.convert('L')
    im_arr = np.array(img)    	
    for i in range(N):
	for j in range(N):
	    if (im_arr[i][j] < threshold):
		im_arr[i][j] = 255
	    else:
		im_arr[i][j] = 0
    image = Image.fromarray(im_arr) 
    image.save(os.path.join(OUT_PATH, picName2))

def array_pic_dark_bg(picName,picName2,degree):
	img = Image.open(os.path.join(ROOT_PATH, picName))
	img = img.rotate(degree) 
	reIm = img.resize((N,N), Image.ANTIALIAS)
	im_arr = np.array(reIm.convert('L'))
       	
	for i in range(N):
		for j in range(N):
 			if (im_arr[i][j] < threshold):
 				im_arr[i][j] = 255
 			else:
 				im_arr[i][j] = 0
	gray_pic(im_arr,picName2)
    
def application():
    j = 0
    for i in range(60000):
	target = i % 38
        picName = str(target) + '.png'
	picName2 = str(i) + '.png'
	
	j %= 4
	if j==0 :
	    array_pic_dark_bg(picName,picName2,0)
        elif j==2:
	    array_pic_dark_bg(picName,picName2,90)
        elif j==3:
	    array_pic_dark_bg(picName,picName2,180)
        else:
	    array_pic_dark_bg(picName,picName2,270)
 	j += 1 

	if i % 50 == 0 :
	    print (picName2 + ' is done...')


        
def main():
	application()

if __name__ == '__main__':
	main()	
