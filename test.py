# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace

# Khởi tạo Haar Cascade Classifier
def init_haar_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar Cascade file not found: {cascade_path}")
    return cv2.CascadeClassifier(cascade_path)

# Phát hiện khuôn mặt bằng Haar Cascade
def detect_face_haar(image, cascade):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), faces

# Phát hiện và nhận dạng khuôn mặt bằng DeepFace
def detect_recognize_faces_deepface(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = DeepFace.extract_faces(img_path=img_rgb, enforce_detection=False, detector_backend="retinaface")
    
    faces = []
    embeddings = []
    for result in results:
        face = result['facial_area']
        faces.append((face['x'], face['y'], face['w'], face['h']))
        
        embedding = DeepFace.represent(img_path=img_rgb[face['y']:face['y']+face['h'], face['x']:face['x']+face['w']], 
                                       model_name="Facenet", 
                                       enforce_detection=False)
        embeddings.append(embedding)
    
    return img_rgb, faces, embeddings

# So sánh khuôn mặt
def compare_faces_deepface(embedding1, embedding2):
    return DeepFace.verify(embedding1, embedding2, model_name="Facenet", distance_metric="cosine")['distance']

# Vẽ hình chữ nhật xung quanh khuôn mặt
def draw_faces(img, faces, is_haar=False, match_index=None):
    for i, face in enumerate(faces):
        if is_haar:
            (x, y, w, h) = face
        else:
            x, y, w, h = face
        color = (0, 255, 0) if i == match_index else (255, 0, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    return img

# Streamlit UI
st.title("Kiểm tra sự hiện diện của sinh viên trong lớp học")

similarity_threshold = st.slider(
    "Ngưỡng độ tương đồng",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.01,
    help="Điều chỉnh ngưỡng này để thay đổi mức độ khớp của khuôn mặt. Giá trị thấp hơn sẽ nghiêm ngặt hơn."
)

# Khởi tạo Haar Cascade
haar_cascade = init_haar_cascade()

col1, col2 = st.columns(2)

with col1:
    st.header("Ảnh sinh viên")
    student_image = st.file_uploader("Tải lên ảnh sinh viên", type=['jpg', 'jpeg', 'png'])

with col2:
    st.header("Ảnh lớp học")
    class_image = st.file_uploader("Tải lên ảnh lớp học", type=['jpg', 'jpeg', 'png'])

# Thêm nút kiểm tra
check_button = st.button("Kiểm tra")

if student_image and class_image and check_button:
    # Xử lý ảnh sinh viên với Haar Cascade
    student_img, student_faces = detect_face_haar(student_image, haar_cascade)
    
    # Xử lý ảnh lớp học với DeepFace
    class_img, class_faces, class_embeddings = detect_recognize_faces_deepface(class_image)

    if len(student_faces) > 0 and len(class_embeddings) > 0:
        # Lấy khuôn mặt lớn nhất từ ảnh sinh viên
        largest_face = max(student_faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Trích xuất đặc trưng từ khuôn mặt sinh viên sử dụng DeepFace
        student_face_img = student_img[y:y+h, x:x+w]
        student_embedding = DeepFace.represent(img_path=student_face_img, 
                                               model_name="Facenet", 
                                               enforce_detection=False)
        
        min_distance = float('inf')
        match_index = -1
        
        for i, class_embedding in enumerate(class_embeddings):
            distance = compare_faces_deepface(student_embedding, class_embedding)
            if distance < min_distance:
                min_distance = distance
                match_index = i

        st.header("Kết quả Kiểm tra")
        st.write(f"Khoảng cách nhỏ nhất: {min_distance:.4f}")

        if min_distance < similarity_threshold:
            st.success("Sinh viên có mặt trong lớp học!")
        else:
            st.error("Không tìm thấy sinh viên trong lớp học.")

        # Vẽ hình chữ nhật
        student_img_with_rect = draw_faces(student_img.copy(), [largest_face], is_haar=True)
        class_img_with_rect = draw_faces(class_img.copy(), class_faces, match_index=match_index)

        col1, col2 = st.columns(2)
        with col1:
            st.image(student_img_with_rect, caption="Ảnh Sinh viên", use_column_width=True)
        with col2:
            st.image(class_img_with_rect, caption="Ảnh Lớp học", use_column_width=True)
    else:
        st.error("Không thể phát hiện khuôn mặt trong một hoặc cả hai ảnh. Vui lòng thử lại với ảnh khác.")
elif check_button:
    st.warning("Vui lòng tải lên cả ảnh sinh viên và ảnh lớp học trước khi kiểm tra.")