# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream 
import numpy as np
import argparse
import imutils
import time
import cv2

import numpy as np
import argparse
import imutils
import time
import cv2
import os
import enum 
########################################################
#
# COCO Names: 
#
#########################################################
class COCOClsID(enum.Enum): 
    car = 2
    bicycle = 3
    motorbike = 4
    bus = 6
    truck = 8

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
#DIU-aerial.names
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
#labelsPath = os.path.sep.join([args["yolo"], "DIU-aerial.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
#weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
weightsPath = os.path.sep.join([args["yolo"], "yolov3-tiny-aerial_20000.weights"])


#configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
configPath = os.path.sep.join([args["yolo"], "yolov3-tiny-aerial.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO

###########################################################################
#  Darknet to OpenCV (Pre-trined Weights) 
#
#
###########################################################################
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])

writer = None
objects = None
currFrame = 0

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	currFrame += 1
	print("Curr Frame:", currFrame)

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	#frame = imutils.resize(frame, height=416, width=416)

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	#blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
	#	(104.0, 177.0, 123.0))

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	#layerOutputs = net.forward()
	layerOutputs = net.forward(ln)
	end = time.time()

	#detections = net.forward()
	#rects = []

	# initialize our lists of detected bounding boxes, confidences,label
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	rects = []

	###################################################################################
	#
	#  Detection Code
	#
	####################################################################################

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			#scores = detection[0:10]
			classID = np.argmax(scores)
			#print("Label:", classID)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			#if (confidence > args["confidence"] and (classID == COCOClsID.car or  classID == COCOClsID.bicycle 
			#				or classID == COCOClsID.bus or classID != COCOClsID.motorbike or classID != COCOClsID.truck) ):			
			if ( confidence > args["confidence"] and classID == 2 ): 
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	#Check for any boxes/detections of our class
	if (len(boxes) == 0):
		continue

	# check if the video writer is None
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	#######################################################################################
	#Tracking Code
	#
	# ensure at least one detection exists
	#######################################################################################
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		# initialize an array of input centroids for the current frame
		inputrects = np.zeros((len(boxes), 4), dtype="int")
		inputrects_min = np.zeros((0, 4), dtype="int")

		#Valid objects
		nValid = -1
		for (i, (startX, startY, width, height)) in enumerate(boxes):
			# use the bounding box coordinates to derive the centroid
			#if (i in idxs):
			#if (i in idxs+1):
				nValid += 1
				inputrects[nValid] = [startX, startY, startX+width, startY+height]

		zerorows = np.all((inputrects == 0), axis=1)
		for i in range(len(zerorows)):
			if (zerorows[i]):
				inputrects_min = np.delete(inputrects, i, axis=0)

		if (len(inputrects_min)):
			objects = ct.update(inputrects_min)
		else:
			objects = ct.update(inputrects)

		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and YOLO-label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			#cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			#text = "{}: {:.4f}".format(LABELS[classIDs[i]],
			#	confidences[i])
			color = (0,0,255)#RED
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format("Truck",
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			# Draw IOU Tracking information
			for (objectID, centroid) in objects.items():
				# draw both the ID of the object and the centroid of the
				# object on the output frame
				text = "ID {}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
 			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# Display to screen
	cv2.imshow('AFRL Wright Pat: SFFP-2020', frame)      
	if cv2.waitKey(1) & 0xFF == ord('q'):
            break
	# write the output frame to disk
	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
#writer.release()
#vs.release()
# do a bit of cleanup
#cv2.destroyAllWindows()
#vs.stop()