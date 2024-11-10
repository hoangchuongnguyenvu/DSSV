# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import os

# Khởi tạo Haar Cascade Classifier
def init_haar_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar Cascade file not found: {cascade_path}")
    return cv2.CascadeClassifier(cascade_path)

# Khởi tạo YuNet và SFace
def init_yunet_sface():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yunet_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(current_dir, "face_recognition_sface_2021dec.onnx")

    if not os.path.exists(yunet_path) or not os.path.exists(sface_path):
        raise FileNotFoundError("YuNet or SFace model file not found")

    face_detector = cv2.FaceDetectorYN.create(yunet_path, "", (0, 0), 0.6, 0.3, 1)
    face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    
    return face_detector, face_recognizer

# Phát hiện khuôn mặt bằng Haar Cascade
def detect_face_haar(image, cascade):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), faces

# Phát hiện và nhận dạng khuôn mặt bằng YuNet và SFace
def detect_recognize_face_yunet(image, face_detector, face_recognizer):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(img)
    
    if faces is not None and len(faces) > 0:
        face = faces[0]
        aligned_face = face_recognizer.alignCrop(img_rgb, face)
        feature = face_recognizer.feature(aligned_face)
        return img_rgb, faces[0], feature
    return img_rgb, None, None

# So sánh khuôn mặt
def compare_faces(feature1, feature2, face_recognizer):
    cosine_score = face_recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
    return cosine_score

# Vẽ hình chữ nhật xung quanh khuôn mặt
def draw_faces(img, faces, is_haar=True):
    if is_haar:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        if faces is not None:
            bbox = faces[:4].astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
    return img

# Streamlit UI
st.title("Ứng dụng So sánh Ảnh Chân dung và Thẻ Sinh viên")

haar_cascade = init_haar_cascade()
yunet_detector, sface_recognizer = init_yunet_sface()

col1, col2 = st.columns(2)

with col1:
    st.header("Ảnh Chân dung")
    portrait_image = st.file_uploader("Tải lên ảnh chân dung", type=['jpg', 'jpeg', 'png'])

with col2:
    st.header("Ảnh Thẻ Sinh viên")
    id_image = st.file_uploader("Tải lên ảnh thẻ sinh viên", type=['jpg', 'jpeg', 'png'])

# Thêm nút kiểm tra
check_button = st.button("Kiểm tra")

if portrait_image and id_image and check_button:
    # Xử lý ảnh chân dung với Haar Cascade
    portrait_img, portrait_faces = detect_face_haar(portrait_image, haar_cascade)
    
    # Xử lý ảnh thẻ sinh viên với YuNet và SFace
    id_img, id_face, id_feature = detect_recognize_face_yunet(id_image, yunet_detector, sface_recognizer)

    if len(portrait_faces) > 0 and id_face is not None:
        # Lấy khuôn mặt lớn nhất từ ảnh chân dung
        largest_face = max(portrait_faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Trích xuất đặc trưng từ khuôn mặt chân dung
        portrait_face_img = portrait_img[y:y+h, x:x+w]
        portrait_face_feature = sface_recognizer.feature(cv2.resize(portrait_face_img, (112, 112)))
        
        # So sánh khuôn mặt
        similarity_score = compare_faces(portrait_face_feature, id_feature, sface_recognizer)

        st.header("Kết quả So sánh")
        st.write(f"Độ tương đồng: {similarity_score:.4f}")

        if similarity_score > 0.3:  # Ngưỡng này có thể điều chỉnh
            st.success("Ảnh chân dung và ảnh thẻ sinh viên KHỚP!")
            color = (0, 255, 0)  # Xanh lá
        else:
            st.error("Ảnh chân dung và ảnh thẻ sinh viên KHÔNG KHỚP!")
            color = (0, 0, 255)  # Đỏ

        # Vẽ hình chữ nhật
        portrait_img_with_rect = draw_faces(portrait_img.copy(), [largest_face])
        id_img_with_rect = draw_faces(id_img.copy(), id_face, is_haar=False)

        col1, col2 = st.columns(2)
        with col1:
            st.image(portrait_img_with_rect, caption="Ảnh Chân dung", use_column_width=True)
        with col2:
            st.image(id_img_with_rect, caption="Ảnh Thẻ Sinh viên", use_column_width=True)
    else:
        st.error("Không thể phát hiện khuôn mặt trong một hoặc cả hai ảnh. Vui lòng thử lại với ảnh khác.")
elif check_button:
    st.warning("Vui lòng tải lên cả ảnh chân dung và ảnh thẻ sinh viên trước khi kiểm tra.")