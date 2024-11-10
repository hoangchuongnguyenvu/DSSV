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
def init_yunet_sface(score_threshold):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yunet_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(current_dir, "face_recognition_sface_2021dec.onnx")

    if not os.path.exists(yunet_path) or not os.path.exists(sface_path):
        raise FileNotFoundError("YuNet or SFace model file not found")

    face_detector = cv2.FaceDetectorYN.create(
        yunet_path,
        "",
        (0, 0),
        score_threshold,
        0.3,
        5000
    )
    face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    
    return face_detector, face_recognizer

# Phát hiện khuôn mặt bằng Haar Cascade
def detect_face_haar(image, cascade):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), faces

# Phát hiện và nhận dạng khuôn mặt bằng YuNet và SFace
def detect_recognize_faces(image, face_detector, face_recognizer):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(img)
    
    features = []
    if faces is not None:
        for face in faces:
            aligned_face = face_recognizer.alignCrop(img_rgb, face)
            feature = face_recognizer.feature(aligned_face)
            features.append(feature)
    
    return img_rgb, faces, features

# So sánh khuôn mặt
def compare_faces(feature1, feature2, face_recognizer):
    cosine_score = face_recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
    return cosine_score

# Vẽ hình chữ nhật xung quanh khuôn mặt
def draw_faces(img, faces, is_haar=False, match_index=None):
    if is_haar:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        for i, face in enumerate(faces):
            bbox = face[:4].astype(int)
            color = (0, 255, 0) if i == match_index else (255, 0, 0)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)
    return img

# Streamlit UI
st.title("Kiểm tra sự hiện diện của sinh viên trong lớp học")

# Thêm thanh trượt cho ngưỡng điểm số
score_threshold = st.slider(
    "Ngưỡng điểm số phát hiện khuôn mặt",
    min_value=0.1,
    max_value=0.9,
    value=0.6,
    step=0.1,
    help="Điều chỉnh ngưỡng này để thay đổi độ nhạy của việc phát hiện khuôn mặt. Giá trị cao hơn sẽ nghiêm ngặt hơn."
)

similarity_threshold = st.slider(
    "Ngưỡng độ tương đồng",
    min_value=0.0,
    max_value=1.0,
    value=0.363,
    step=0.001,
    help="Điều chỉnh ngưỡng này để thay đổi mức độ khớp của khuôn mặt. Giá trị cao hơn sẽ nghiêm ngặt hơn."
)

# Khởi tạo các mô hình
haar_cascade = init_haar_cascade()
yunet_detector, sface_recognizer = init_yunet_sface(score_threshold)

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
    
    # Xử lý ảnh lớp học với YuNet và SFace
    class_img, class_faces, class_features = detect_recognize_faces(class_image, yunet_detector, sface_recognizer)

    if len(student_faces) > 0 and len(class_features) > 0:
        # Lấy khuôn mặt lớn nhất từ ảnh sinh viên
        largest_face = max(student_faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Trích xuất đặc trưng từ khuôn mặt sinh viên
        student_face_img = student_img[y:y+h, x:x+w]
        student_feature = sface_recognizer.feature(cv2.resize(student_face_img, (112, 112)))
        
        max_similarity = 0
        match_index = -1
        
        for i, class_feature in enumerate(class_features):
            similarity_score = compare_faces(student_feature, class_feature, sface_recognizer)
            if similarity_score > max_similarity:
                max_similarity = similarity_score
                match_index = i

        st.header("Kết quả Kiểm tra")
        st.write(f"Độ tương đồng cao nhất: {max_similarity:.4f}")

        if max_similarity > similarity_threshold:
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
        st.error("Không thể phát hiện khuôn mặt trong một hoặc cả hai ảnh. Vui lòng thử lại với ảnh khác hoặc điều chỉnh ngưỡng điểm số.")
elif check_button:
    st.warning("Vui lòng tải lên cả ảnh sinh viên và ảnh lớp học trước khi kiểm tra.")