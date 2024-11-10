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

# Khởi tạo SFace
def init_sface():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sface_path = os.path.join(current_dir, "face_recognition_sface_2021dec.onnx")

    if not os.path.exists(sface_path):
        raise FileNotFoundError("SFace model file not found")

    face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    return face_recognizer

def process_student_image(image, cascade, face_recognizer):
    """
    Xử lý ảnh sinh viên:
    1. Dùng Haar Cascade để phát hiện khuôn mặt
    2. Dùng SFace để trích xuất đặc trưng
    """
    # Đọc và chuyển đổi ảnh
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt bằng Haar Cascade
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Lấy khuôn mặt lớn nhất (giả sử là khuôn mặt chính)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_img = img_rgb[y:y+h, x:x+w]
        
        # Tiền xử lý khuôn mặt
        face_img = cv2.resize(face_img, (112, 112))
        
        # Trích xuất đặc trưng bằng SFace
        feature = face_recognizer.feature(face_img)
        return img_rgb, (x, y, w, h), feature
    return img_rgb, None, None

def process_class_image(image, cascade, face_recognizer):
    """
    Xử lý ảnh lớp học:
    1. Dùng Haar Cascade để phát hiện tất cả khuôn mặt
    2. Dùng SFace để trích xuất đặc trưng từng khuôn mặt
    """
    # Đọc và chuyển đổi ảnh
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt bằng Haar Cascade
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    face_features = []
    if len(faces) > 0:
        for face in faces:
            x, y, w, h = face
            face_img = img_rgb[y:y+h, x:x+w]
            
            # Tiền xử lý khuôn mặt
            face_img = cv2.resize(face_img, (112, 112))
            
            # Trích xuất đặc trưng bằng SFace
            feature = face_recognizer.feature(face_img)
            face_features.append((face, feature))
    
    return img_rgb, faces, face_features

def compare_faces(feature1, feature2, face_recognizer):
    """So sánh hai đặc trưng khuôn mặt"""
    cosine_score = face_recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
    return cosine_score

def draw_results(img, faces, matches, scores):
    """Vẽ kết quả nhận dạng lên ảnh"""
    img_copy = img.copy()
    for face, is_match, score in zip(faces, matches, scores):
        x, y, w, h = face
        color = (0, 255, 0) if is_match else (0, 0, 255)
        text = f"Match ({score:.2f})" if is_match else f"No Match ({score:.2f})"
        
        # Vẽ hình chữ nhật và text
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_copy, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_copy

# Thiết lập giao diện Streamlit
st.set_page_config(page_title="Nhận diện Khuôn mặt", layout="wide")
st.title("Tìm kiếm Sinh viên trong Ảnh Lớp học")

# Khởi tạo các mô hình
haar_cascade = init_haar_cascade()
sface_recognizer = init_sface()

# Tạo layout hai cột
col1, col2 = st.columns(2)

with col1:
    st.header("Ảnh Sinh viên")
    student_image = st.file_uploader("Tải lên ảnh sinh viên", type=['jpg', 'jpeg', 'png'])

with col2:
    st.header("Ảnh Lớp học")
    class_image = st.file_uploader("Tải lên ảnh lớp học", type=['jpg', 'jpeg', 'png'])

# Thêm thanh trượt để điều chỉnh ngưỡng
threshold = st.slider("Ngưỡng nhận dạng (0-1)", 0.0, 1.0, 0.3, 0.001)

# Thêm nút tìm kiếm
search_button = st.button("Tìm kiếm")

if student_image and class_image and search_button:
    try:
        # Xử lý ảnh sinh viên
        student_img, student_face, student_feature = process_student_image(
            student_image, haar_cascade, sface_recognizer)
        
        if student_face is not None:
            # Xử lý ảnh lớp học
            class_img, class_faces, class_features = process_class_image(
                class_image, haar_cascade, sface_recognizer)
            
            if len(class_faces) > 0:
                # So sánh với từng khuôn mặt trong lớp học
                matches = []
                scores = []
                for _, class_feature in class_features:
                    score = compare_faces(student_feature, class_feature, sface_recognizer)
                    matches.append(score > threshold)
                    scores.append(score)
                
                # Vẽ kết quả
                result_img = draw_results(class_img, class_faces, matches, scores)
                
                # Hiển thị kết quả
                st.header("Kết quả Tìm kiếm")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(student_img, caption="Ảnh Sinh viên", use_column_width=True)
                with col2:
                    st.image(result_img, caption="Kết quả tìm kiếm trong lớp học", use_column_width=True)
                
                # Hiển thị thống kê
                matches_count = sum(matches)
                st.write(f"Tìm thấy {matches_count} khuôn mặt khớp trong ảnh lớp học")
                
                # Hiển thị chi tiết điểm số
                st.write("Chi tiết điểm số:")
                for i, (score, is_match) in enumerate(zip(scores, matches), 1):
                    status = "Khớp" if is_match else "Không khớp"
                    st.write(f"Khuôn mặt {i}: {status} (Điểm số: {score:.4f})")
            else:
                st.error("Không thể phát hiện khuôn mặt trong ảnh lớp học")
        else:
            st.error("Không thể phát hiện khuôn mặt trong ảnh sinh viên")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {str(e)}")
elif search_button:
    st.warning("Vui lòng tải lên cả ảnh sinh viên và ảnh lớp học trước khi tìm kiếm")