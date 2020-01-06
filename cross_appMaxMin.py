#coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import cross_backward
import cross_forward
import os

ROOT_PATH="./picByNoiseReduce34"
STLVALUE = 5000
THRESHOLD = 50
N = 28

def restore_model(testPicArr):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, cross_forward.INPUT_NODE])
		y = cross_forward.forward(x, None)
		#preValue = tf.argmax(y, 1)
		preValue = y

		variable_averages = tf.train.ExponentialMovingAverage(cross_backward.MOVING_AVERAGE_DECAY)
 		variables_to_restore = variable_averages.variables_to_restore()
 		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(cross_backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
		
				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue
			else:
				print("No checkpoint file found")
				return -1

def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((N,N), Image.ANTIALIAS)
	im_arr = np.array(reIm.convert('L'))

	for i in range(N):
		for j in range(N):
 			if (im_arr[i][j] < THRESHOLD):
 				im_arr[i][j] = 0
 			else:
 				im_arr[i][j] = 255
	for i in range(N):
		for j in range(N):
			im_arr[i][j] = 255 - im_arr[i][j]

	nm_arr = im_arr.reshape([1, 784])
	nm_arr = nm_arr.astype(np.float32)
	img = np.multiply(nm_arr, 1.0/255.0)

	return nm_arr  #img

def application():
	maxValue = -10000
	minValue = 10000 
	count = 0
	for i in range(39):
		picName = str(i) + '.png'
		testPic = os.path.join(ROOT_PATH,picName)
		testPicArr = pre_pic(testPic)
		preValue = restore_model(testPicArr)

		result = 0
		pos = np.argmax(preValue, 1)
		index = pos[0]
		'''
		if 0 == index :
		    prediction = 1.0 - abs(STLVALUE - preValue[0][index]) / STLVALUE
		    print "The prediction is:", prediction
		    if prediction >= 0.9:
		        result = 1

		print "The result is:", result
		'''
		funValue = preValue[0][index]
		count += funValue
		maxValue = max(maxValue , funValue)
		minValue = min(minValue , funValue)
		print ' index ' + str(i) + " The result is:", funValue
	print 'max:' + str(maxValue) + '  min:' + str(minValue)#max:2224.4456  min:319.64392 aver:1084.0067436581567
	print 'aver:' + str(count/42)

def main():
	application()

if __name__ == '__main__':
	main()		
