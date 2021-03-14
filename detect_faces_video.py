# USAGE
# python detect_faces_video.py

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import base64
import urllib.parse
import requests
import json
import timeit
import sys
import threading
from multiprocessing import Process, Queue

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
url = 'http://service.mmlab.uit.edu.vn/checkinService_demo/user_login/post/'
# url = 'http://192.168.28.73:81/user_login/post/'
# ------------------------------------
data = {'user_name': 'tester1', 'password': 'tester1'}
headers = {'Content-type': 'application/json'}
data_json = json.dumps(data)
response = requests.post(url, data=data_json, headers=headers)
# print(response)
response = response.json()
# print(response['token'])
token = response['token']

url = 'http://service.mmlab.uit.edu.vn/checkinService_demo/search_face/post/'
####################################
# q = Queue()


def get_info(image_read,q):
	url = 'http://service.mmlab.uit.edu.vn/checkinService_demo/search_face/post/'
	_, a_numpy = cv2.imencode('.jpg', image_read)
	a = a_numpy.tobytes()
	encoded = base64.encodebytes(a)
	image_encoded = encoded.decode('utf-8')

	# ###################################

	data = {'token': token, 'data': {'image_encoded': image_encoded,
	    'class_id': '0', 'model': '0', 'classifier': '0'}}
	headers = {'Content-type': 'application/json'}
	data_json = json.dumps(data)
	response = requests.post(url, data=data_json, headers=headers)
	# print(response)
	response = response.json()
	# print(response)
	if len(response) > 2:
		get_name = list(response['data'].values())[6:8]
		name = str(get_name[-1])
		# print(name)
		q.put(name)
	else:
		q.put("Unknow")
	# return (response)

t = threading.Thread()

# x = 0
q = Queue()
q.put("Unknow")
response = "Unknow"
def api_face(frame, x, y, x1, y1):
	global url
	global token
	global t
	global q
	global response

	image_read = frame[startY:endY, startX:endX]
	# threading
	
	# q = Queue()
	# q.put("Unknow")
	
	if not t.isAlive():	
		t = threading.Thread(target=get_info, args=(image_read,q,))
		response = q.get()
		print(response)
		t.start()
		# print("New: ",q.get())
		# t.join()
		
	# Multiprocessing
	# x = Queue()
	# p = Process(target=test_info, args=(x,))
	# p.start()
	# print("Info: ",x.get())    # prints "[42, None, 'hello']"
	# p.join()

	# response = get_info(image_read)
	# if len(response)>2:
	if response != "": 
		# data = list(response['data'].values())[6:8]
		# name = str(data[-1])
		# info = data            
		# font = cv2.FONT_HERSHEY_DUPLEX
		# # font = cv2.FONT_HERSHEY_SIMPLEX
		# print(name)
		cv2.rectangle(frame, (x, y1), (x1,y1+15), (0, 0, 255), cv2.FILLED)
		cv2.putText(frame, str(response), (x + 3, y1 + 12), font, 0.5, (255, 255, 255), 2)
	return frame

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
video_capture = cv2.VideoCapture(0)
# time.sleep(2.0)
prev_frame_time = 0
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	# frame = vs.read()
	# frame = imutils.resize(frame, width=1000)

	ret, frame = video_capture.read()

	new_frame_time = time.time()
	fps = 1/(new_frame_time-prev_frame_time) 
	prev_frame_time = new_frame_time 
	fps = str(int(fps))
	font = cv2.FONT_HERSHEY_SIMPLEX 
	cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	print(detections.shape)
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < 0.5:
			continue
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10

		# frame = api_face(frame,startX,startY,endX,endY)
		
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		# cv2.putText(frame, text, (startX, y),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
# vs.stop()
