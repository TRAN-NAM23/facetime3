import cv2 #Nhập thư viện OpenCV, được sử dụng để xử lý hình ảnh và video.
from keras.models import model_from_json #Nhập hàm model_from_json từ thư viện Keras, dùng để tải mô hình từ file JSON.
import numpy as np #Nhập thư viện NumPy, dùng để xử lý mảng và toán học.
# from keras_preprocessing.image import load_img dùng để tải ảnh từ file 
json_file = open("emotiondetector.json", "r")#Mở file "emotiondetector.json" ở chế độ đọc ("r"). File này chứa cấu trúc của mô hình đã được huấn luyện.
model_json = json_file.read()#Đọc nội dung của file JSON vào biến model_json.
json_file.close()#Đóng file JSON sau khi đọc xong.
model = model_from_json(model_json)#Tạo mô hình Keras từ cấu trúc đã đọc từ file JSON.

model.load_weights("emotiondetector.keras")#Tải trọng lượng đã huấn luyện cho mô hình từ file "emotiondetector.keras".
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'#Định nghĩa đường dẫn đến file XML chứa bộ phân loại khuôn mặt Haar.
face_cascade=cv2.CascadeClassifier(haar_file)#Tạo đối tượng phân loại khuôn mặt Haar từ file XML đã định nghĩa.

def extract_features(image):#Hàm này trích xuất đặc trưng từ ảnh đầu vào.
    feature = np.array(image)#Chuyển ảnh đầu vào thành mảng NumPy.
    feature = feature.reshape(1,48,48,1)#Reshape mảng thành kích thước (1, 48, 48, 1) phù hợp với đầu vào của mô hình.
    return feature/255.0#Trả về mảng đặc trưng đã được chuẩn hóa (chia cho 255) để đưa vào mô hình.

webcam=cv2.VideoCapture(0)# Khởi tạo đối tượng webcam để đọc video từ camera mặc định.
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
#Tạo từ điển ánh xạ từ nhãn số (0-6) sang các loại cảm xúc tương ứng.
while True:#Vòng lặp vô hạn để xử lý video từ webcam.
    i,im=webcam.read()#Đọc một khung hình từ webcam
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)#Chuyển ảnh từ màu RGB sang xám.
    faces=face_cascade.detectMultiScale(im,1.3,5)#Sử dụng bộ phân loại Haar để tìm kiếm khuôn mặt trong ảnh. Tham số 1.3 là tỉ lệ phóng to, 5 là số điểm tối thiểu cho một vùng là khuôn mặt.
    try:            #phần này sử lí lỗi 
        for (p,q,r,s) in faces:#Lặp qua các khuôn mặt đã được phát hiện.
            image = gray[q:q+s,p:p+r]#Cắt khuôn mặt từ ảnh xám.
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)#Vẽ khung hình chữ nhật màu đỏ quanh khuôn mặt trên ảnh gốc.
            image = cv2.resize(image,(48,48))#Resize khuôn mặt về kích thước (48, 48) phù hợp với đầu vào của mô hình.
            img = extract_features(image)#Trích xuất đặc trưng từ khuôn mặt đã resize.
            pred = model.predict(img)#dự đoán cảm súc từ mô hình 
            prediction_label = labels[pred.argmax()]#Lấy nhãn cảm xúc tương ứng với kết quả dự đoán.
            # print("Predicted Output:", prediction_label)
            # cv2.putText(im,prediction_label)
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))#hiển thị nhãn dán dự đoán lên ảnh gốc 
        cv2.imshow("Output",im)#cập nhật hình ảnh 
        cv2.waitKey(27)#chờ 27mili giây cập nhật khung hình mới 
    except cv2.error:
        pass