import streamlit as st
import cv2
import numpy as np
import os
from tqdm import tqdm
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image
import requests
from io import BytesIO

# Khởi tạo Firebase (chỉ thực hiện một lần)
if not firebase_admin._apps:
    cred = credentials.Certificate("hchuong-firebase-adminsdk-1m82k-829fb1690b.json")
    firebase_admin.initialize_app(cred, {'storageBucket': 'hchuong.appspot.com'})

# Kết nối đến Firestore và Storage
db = firestore.client()
bucket = storage.bucket()

# Tải YuNet và SFace
@st.cache_resource
def load_models():
    yunet = cv2.FaceDetectorYN.create(
        "face_detection_yunet_2023mar.onnx",
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )
    sface = cv2.FaceRecognizerSF.create(
        "face_recognition_sface_2021dec.onnx",
        ""
    )
    return yunet, sface

yunet, sface = load_models()

def preprocess_face(face_img):
    return cv2.resize(face_img, (112, 112))

def load_known_faces_from_firebase():
    known_faces = []
    known_names = []
    students_ref = db.collection("Students")
    students = students_ref.get()
    for student in tqdm(students, desc="Xử lý ảnh chân dung"):
        student_data = student.to_dict()
        chandung_url = student_data.get("ChanDung")
        if chandung_url:
            try:
                response = requests.get(chandung_url)
                img = Image.open(BytesIO(response.content))
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                face = preprocess_face(img)
                face_feature = sface.feature(face)
                known_faces.append(face_feature)
                known_names.append(student_data.get("Name", "Unknown"))
            except Exception as e:
                st.error(f"Lỗi khi xử lý ảnh của {student_data.get('Name', 'Unknown')}: {e}")
    return known_faces, known_names

def recognize_faces(class_image, known_faces, known_names):
    height, width, _ = class_image.shape
    yunet.setInputSize((width, height))
    _, faces = yunet.detect(class_image)

    recognized_names = set()
    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            face_img = class_image[y:y+h, x:x+w]
            face_img = preprocess_face(face_img)
            face_feature = sface.feature(face_img)

            distances = [sface.match(face_feature, known_face, cv2.FaceRecognizerSF_FR_COSINE) for known_face in known_faces]
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]

            if min_distance < 0.3 and known_names[min_distance_index] not in recognized_names:
                name = known_names[min_distance_index]
                recognized_names.add(name)
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(class_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(class_image, f"{name} ({min_distance:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return class_image, recognized_names

st.title("Nhận diện khuôn mặt trong ảnh lớp học")

uploaded_file = st.file_uploader("Chọn ảnh lớp học", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh lớp học đã tải lên", use_column_width=True)

    if st.button("Nhận diện khuôn mặt"):
        with st.spinner("Đang xử lý..."):
            known_faces, known_names = load_known_faces_from_firebase()
            class_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            result_image, recognized_names = recognize_faces(class_image, known_faces, known_names)

        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption="Kết quả nhận diện", use_column_width=True)
        st.success(f"Số lượng khuôn mặt được nhận diện: {len(recognized_names)}")
        st.write("Danh sách sinh viên được nhận diện:")
        for name in recognized_names:
            st.write(f"- {name}")