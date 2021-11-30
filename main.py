#In[0] import, setting
from numpy.core.numeric import True_
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from playsound import playsound
import threading

daDeo = False


def deoKhauTrang(daDeo):
    if(daDeo==True):
        playsound("sound/thucHien.mp3")
    else:
        playsound("sound/khongThucHien.mp3")

def detect_and_predict_mask(frame, faceNet, maskNet):
    # Lấy kích thước của khung và tạo mảng các điểm để nhận diện
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
 
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	#Khởi tạo danh sách các khuôn mặt, vị trí tương ứng của các điểm trên khuôn mặt, và danh sách các dự đoán
	faces = []
	locs = []
	preds = []
	#frame: Khung để nhận dạng khuôn mặt
	#faceNet: Model để phát hiện vị trí của khuôn mặt trong ảnh
	#maskNet: Model để phân loại mặt có đeo khẩu trang
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# trích xuất độ tin cậy(xác suất) được liên kết với phát hiện
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		# lọc ra các phát hiện yếu để đảm bảo độ tin cậy lớn hơn độ tin cậy tối thiểu
		if confidence > 0.5:
			# tính toạn độ(x, y) của box để giới hạn cho mặt của đối tượng
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# đảm bảo các hộp giới hạn nằm trong kích thước của khung
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# trích xuất vùng mà chúng ta quan tâm(ROI) của khuôn mặt
			# Chuyển đối từ BGR sang RGB, odering, chuẩn hóa kích thước và tiến hành tiền xử lý
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# thêm các hộp khuôn mặt và viền tương ứng của chúng trong danh sách
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Chỉ đưa ra dự đoán nếu phát hiện ra ít nhất một khuôn mặt
	if len(faces) > 0:
		# để tăng tốc độ nhận diện thì chúng ta sẽ đưa ra dự đoán hàng loạt trên tất cả các khuôn mặt cùng một lúc 
  		# thay vì nhận diện từng khuôn mặt
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# trả về bộ vị trí của các khuôn mặt
	return (locs, preds)
# In[1]: load model
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load facemask model
maskNet = load_model("mask_detector.model")

# In[2] chuẩn bị stream video
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# In[3]: xử lý video stream
while True:
	# lấy từng khung hình (frame) từ video 
 	# và thay đổi kích thước của chúng để có chiều rộng tối đa là 400pixel
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# phát hện các khuôn mặt có trong khung hình và xác định liệu họ có đeo mặt nạ hay không 
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# sử dụng vòng lặp để phát hiện khuôn mặt và vị trí của chúng trên khung hình
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# the bounding box and text
		# xác định các class labe và màu mà chúng ta sẽ dùng để box giới hạn và văn bản hiển thị
		label = "Have a nice day" if mask > withoutMask else "Please wear a mask"
		color = (0, 255, 0) if label == "Have a nice day" else (0, 0, 255)
		daDeo = label
		# hiển thị xác suất lên khung hình
		label = "{}".format(label)


		# hiển thị xác suất nhận diện và bõ giới hạn
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	if (daDeo == True):
		alarm = threading.Thread(target=deoKhauTrang, args=(daDeo,))
        #alarm.start()
	# show the output frame
	# xuất khung hình
	cv2.imshow("Mask Detection", frame)
	key = cv2.waitKey(1) & 0xFF
	# nhấn q để thoát chương trình
	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()
