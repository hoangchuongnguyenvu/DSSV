# -*- coding: utf-8 -*-

import streamlit as st
import cv2
import numpy as np
import os

# Khởi tạo SFace và YuNet
def init_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yunet_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(current_dir, "face_recognition_sface_2021dec.onnx")

    if not os.path.exists(yunet_path):
        raise FileNotFoundError(f"YuNet model file not found: {yunet_path}")
    if not os.path.exists(sface_path):
        raise FileNotFoundError(f"SFace model file not found: {sface_path}")

    try:
        face_detector = cv2.FaceDetectorYN.create(
            yunet_path,
            "",
            (0, 0),
            0.2,
            0.3,
            1
        )
    except cv2.error as e:
        raise Exception(f"Error loading YuNet model: {str(e)}")

    try:
        face_recognizer = cv2.FaceRecognizerSF.create(
            sface_path,
            ""
        )
    except cv2.error as e:
        raise Exception(f"Error loading SFace model: {str(e)}")

    return face_detector, face_recognizer

# Xử lý ảnh
def process_image(image, face_detector):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(img)
    return img, img_rgb, faces

# So sánh khuôn mặt
def compare_faces(img1, face1, img2, face2, face_recognizer):
    aligned_face1 = face_recognizer.alignCrop(img1, face1)
    aligned_face2 = face_recognizer.alignCrop(img2, face2)
    feature1 = face_recognizer.feature(aligned_face1)
    feature2 = face_recognizer.feature(aligned_face2)
    cosine_score = face_recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
    l2_score = face_recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_NORM_L2)
    return cosine_score, l2_score

# Hàm vẽ hình chữ nhật
def draw_faces(img, faces, color):
    for face in faces:
        bbox = face[:4].astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)
    return img

# Streamlit UI
st.title("Ứng dụng So sánh Ảnh Chân dung và Thẻ Sinh viên")

face_detector, face_recognizer = init_models()

col1, col2 = st.columns(2)

with col1:
    st.header("Ảnh Chân dung")
    portrait_image = st.file_uploader("Tải lên ảnh chân dung", type=['jpg', 'jpeg', 'png'])

with col2:
    st.header("Ảnh Thẻ Sinh viên")
    id_image = st.file_uploader("Tải lên ảnh thẻ sinh viên", type=['jpg', 'jpeg', 'png'])

if portrait_image and id_image:
    portrait_img, portrait_img_rgb, portrait_faces = process_image(portrait_image, face_detector)
    id_img, id_img_rgb, id_faces = process_image(id_image, face_detector)

    if portrait_faces is not None and id_faces is not None and len(portrait_faces) > 0 and len(id_faces) > 0:
        cosine_score, l2_score = compare_faces(portrait_img_rgb, portrait_faces[0], id_img_rgb, id_faces[0], face_recognizer)

        st.header("Kết quả So sánh")
        st.write(f"Cosine Similarity Score: {cosine_score:.4f}")
        st.write(f"L2 Distance: {l2_score:.4f}")

        if cosine_score > 0.363:  # Ngưỡng này có thể điều chỉnh
            st.success("Ảnh chân dung và ảnh thẻ sinh viên KHỚP!")
            color_portrait = color_id = (0, 255, 0)  # Xanh lá
        else:
            st.error("Ảnh chân dung và ảnh thẻ sinh viên KHÔNG KHỚP!")
            color_portrait = (0, 255, 0)  # Xanh lá cho ảnh chân dung
            color_id = (0, 0, 255)  # Đỏ cho ảnh thẻ sinh viên

        # Vẽ hình chữ nhật
        portrait_img_with_rect = draw_faces(portrait_img, portrait_faces, color_portrait)
        id_img_with_rect = draw_faces(id_img, id_faces, color_id)

        # Chuyển đổi ảnh về RGB để hiển thị
        portrait_img_with_rect_rgb = cv2.cvtColor(portrait_img_with_rect, cv2.COLOR_BGR2RGB)
        id_img_with_rect_rgb = cv2.cvtColor(id_img_with_rect, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(portrait_img_with_rect_rgb, caption="Ảnh Chân dung", use_column_width=True)
        with col2:
            st.image(id_img_with_rect_rgb, caption="Ảnh Thẻ Sinh viên", use_column_width=True)
    else:
        st.error("Không thể phát hiện khuôn mặt trong một hoặc cả hai ảnh. Vui lòng thử lại với ảnh khác.")