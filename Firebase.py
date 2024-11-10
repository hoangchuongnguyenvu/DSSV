# -*- coding: utf-8 -*-

import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import uuid
import cv2
import numpy as np
import os

st.set_page_config(layout="wide")

# Khởi tạo Firebase (chỉ thực hiện một lần)
if not firebase_admin._apps:
    key_dict = st.secrets["firebase"]
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred, {'storageBucket': 'hchuong.appspot.com'})

# Kết nối đến Firestore và Storage
db = firestore.client()
bucket = storage.bucket()

# Khởi tạo session state
if 'current_action' not in st.session_state:
    st.session_state.current_action = None
if 'selected_student' not in st.session_state:
    st.session_state.selected_student = None
if 'action' not in st.session_state:
    st.session_state.action = None

# CSS cho ứng dụng
st.markdown("""
<style>
    .table-container {
        display: flex;
        justify-content: center;
        width: 100%;
        overflow-x: auto;
    }
    .dataframe {
        font-size: 14px;
        width: 100%;
        border-collapse: collapse;
    }
    .dataframe th, .dataframe td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    .dataframe td:nth-child(3), .dataframe td:nth-child(4) {
        text-align: center;
    }
    .dataframe img {
        max-width: 80px;
        max-height: 80px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .stApp {
        max-width: 100%;
        margin: 0 auto;
    }
    .search-result {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin: 20px 0;
        background-color: #f8f9fa;
    }
    .search-result-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    .search-result-label {
        font-weight: bold;
        margin-bottom: 10px;
        color: #2c3e50;
    }
    .search-result img {
        max-width: 200px;
        max-height: 200px;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        background-color: white;
    }
    .stDataFrame {
        width: 100%;
        max-width: none !important;
    }
    .add-form {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .search-form {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .search-result-divider {
        width: 100%;
        height: 2px;
        background-color: #e9ecef;
        margin: 20px 0;
    }
    .face-comparison {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .comparison-result {
        margin-top: 20px;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
    }
    .comparison-score {
        font-size: 1.2em;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Hàm để tải lên hình ảnh vào Firebase Storage
def upload_image(file):
    if file is not None:
        file_name = str(uuid.uuid4()) + "." + file.name.split(".")[-1]
        blob = bucket.blob(file_name)
        blob.upload_from_file(file)
        blob.make_public()
        return blob.public_url
    return None

# Hàm để lấy dữ liệu sinh viên từ Firestore
def get_student_data():
    students_ref = db.collection("Students")
    students = students_ref.get()
    table_data = []
    for student in students:
        student_data = student.to_dict()
        table_data.append({
            "ID": student.id,
            "Name": student_data.get("Name", ""),
            "TheSV": student_data.get("TheSV", ""),
            "ChanDung": student_data.get("ChanDung", "")
        })
    return table_data

# Hàm khởi tạo Haar Cascade
def init_haar_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar Cascade file not found: {cascade_path}")
    return cv2.CascadeClassifier(cascade_path)

# Hàm khởi tạo YuNet và SFace
def init_yunet_sface():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yunet_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(current_dir, "face_recognition_sface_2021dec.onnx")

    if not os.path.exists(yunet_path) or not os.path.exists(sface_path):
        raise FileNotFoundError("YuNet or SFace model file not found")

    face_detector = cv2.FaceDetectorYN.create(yunet_path, "", (0, 0), 0.6, 0.3, 1)
    face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    
    return face_detector, face_recognizer

# Hàm phát hiện khuôn mặt bằng Haar Cascade
def detect_face_haar(image, cascade):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), faces

# Hàm phát hiện và nhận dạng khuôn mặt bằng YuNet và SFace
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

# Hàm so sánh khuôn mặt
def compare_faces(feature1, feature2, face_recognizer):
    cosine_score = face_recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
    return cosine_score

# Hàm vẽ hình chữ nhật xung quanh khuôn mặt
def draw_faces(img, faces, is_haar=True):
    if is_haar:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        if faces is not None:
            bbox = faces[:4].astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
    return img

# Add these functions before the tab3 section
def process_class_image(image, cascade, recognizer):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    features = []
    for face in faces:
        x, y, w, h = face
        face_img = img[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        resized_face = cv2.resize(face_rgb, (112, 112))
        feature = recognizer.feature(resized_face)
        features.append((face, feature))
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), faces, features

def process_student_image(image, cascade, recognizer):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Use the first detected face
        face_img = img[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        resized_face = cv2.resize(face_rgb, (112, 112))
        feature = recognizer.feature(resized_face)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), faces[0], feature
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None, None

# Phần UI chính
st.title("Hệ thống Quản lý Sinh viên")

# Tạo 3 tabs thay vì 2 tabs
tab1, tab2, tab3 = st.tabs(["1. Danh sách Sinh viên", 
                           "2. Kiểm tra Ảnh", 
                           "3. Tìm kiếm trong Lớp học"])

# Trong tab1
with tab1:
    st.header("1. Danh sách Sinh viên")
    # Tạo các nút cho các chức năng
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Thêm Sinh viên"):
            st.session_state.current_action = 'add'
    with col2:
        if st.button("Tìm kiếm Sinh viên"):
            st.session_state.current_action = 'search'
    with col3:
        if st.button("Danh sách Sinh viên"):
            st.session_state.current_action = 'list'

    # Hiển thị danh sách sinh viên
    if st.session_state.current_action == 'list' or st.session_state.current_action is None:
        st.subheader("Danh sách Sinh viên")
        table_data = get_student_data()
        df = pd.DataFrame(table_data)

        # Thêm cột cho việc chọn sinh viên
        df['Edit'] = False
        df['Delete'] = False

        edited_df = st.data_editor(
            df,
            hide_index=True,
            column_config={
                "TheSV": st.column_config.ImageColumn("Thẻ SV", help="Thẻ sinh viên", width="medium"),
                "ChanDung": st.column_config.ImageColumn("Chân dung", help="Ảnh chân dung", width="medium"),
                "ID": st.column_config.TextColumn("ID", help="ID sinh viên", width="medium"),
                "Name": st.column_config.TextColumn("Tên", help="Tên sinh viên", width="large"),
                "Edit": st.column_config.CheckboxColumn("Chỉnh sửa", default=False),
                "Delete": st.column_config.CheckboxColumn("Xóa", default=False)
            },
            disabled=["ID", "Name", "TheSV", "ChanDung"],
            use_container_width=True,
        )

        # Xử lý chỉnh sửa
        students_to_edit = edited_df[edited_df['Edit']].iloc[:1]
        if not students_to_edit.empty:
            student = students_to_edit.iloc[0]
            st.subheader(f"Chỉnh sửa thông tin cho sinh viên: {student['Name']}")
            edit_id = st.text_input("ID mới", value=student['ID'])
            edit_name = st.text_input("Tên mới", value=student['Name'])
            edit_thesv = st.file_uploader("Thẻ Sinh viên mới", type=["jpg", "png", "jpeg"], key="edit_thesv")
            edit_chandung = st.file_uploader("Ảnh Chân dung mới", type=["jpg", "png", "jpeg"], key="edit_chandung")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Cập nhật"):
                    update_data = {"Name": edit_name}
                    if edit_thesv:
                        thesv_url = upload_image(edit_thesv)
                        update_data["TheSV"] = thesv_url
                    if edit_chandung:
                        chandung_url = upload_image(edit_chandung)
                        update_data["ChanDung"] = chandung_url
                    
                    if edit_id != student['ID']:
                        current_data = db.collection("Students").document(student['ID']).get().to_dict()
                        current_data.update(update_data)
                        db.collection("Students").document(edit_id).set(current_data)
                        db.collection("Students").document(student['ID']).delete()
                        st.success(f"Đã cập nhật thông tin và ID sinh viên từ {student['ID']} thành {edit_id}!")
                    else:
                        db.collection("Students").document(student['ID']).update(update_data)
                        st.success(f"Đã cập nhật thông tin sinh viên {student['ID']}!")
                    st.rerun()
            with col2:
                if st.button("Hủy"):
                    st.rerun()

        # Xử lý xóa
        students_to_delete = edited_df[edited_df['Delete']].iloc[:1]
        if not students_to_delete.empty:
            student = students_to_delete.iloc[0]
            st.subheader(f"Xác nhận xóa sinh viên: {student['Name']}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Xác nhận xóa"):
                    db.collection("Students").document(student['ID']).delete()
                    st.success(f"Đã xóa sinh viên {student['ID']}!")
                    st.rerun()
            with col2:
                if st.button("Hủy xóa"):
                    st.rerun()

    # Chức năng thêm sinh viên mới
    elif st.session_state.current_action == 'add':
        st.markdown('<div class="add-form">', unsafe_allow_html=True)
        st.subheader("Thêm Sinh viên mới")
        col1, col2 = st.columns(2)
        with col1:
            new_id = st.text_input("ID")
        with col2:
            new_name = st.text_input("Tên")
        
        col3, col4 = st.columns(2)
        with col3:
            new_thesv = st.file_uploader("Thẻ Sinh viên", type=["jpg", "png", "jpeg"], key="new_thesv")
        with col4:
            new_chandung = st.file_uploader("Ảnh Chân dung", type=["jpg", "png", "jpeg"], key="new_chandung")

        col5, col6 = st.columns(2)
        with col5:
            if st.button("Xác nhận thêm"):
                if new_id and new_name and new_thesv and new_chandung:
                    thesv_url = upload_image(new_thesv)
                    chandung_url = upload_image(new_chandung)
                    db.collection("Students").document(new_id).set({
                        "Name": new_name,
                        "TheSV": thesv_url,
                        "ChanDung": chandung_url
                    })
                    st.success("Đã thêm sinh viên mới!")
                    st.session_state.current_action = None
                    st.rerun()
                else:
                    st.warning("Vui lòng điền đầy đủ thông tin!")
        with col6:
            if st.button("Hủy"):
                st.session_state.current_action = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Chức năng tìm kiếm
    elif st.session_state.current_action == 'search':
        st.markdown('<div class="search-form">', unsafe_allow_html=True)
        st.subheader("Tìm kiếm Sinh viên")
        search_option = st.radio("Chọn cách tìm kiếm:", ["Tìm theo ID", "Tìm theo Tên"])
        
        if search_option == "Tìm theo ID":
            col1, col2 = st.columns([3,1])
            with col1:
                search_id = st.text_input("Nhập ID sinh viên cần tìm")
            with col2:
                search_button = st.button("Tìm kiếm")
                if search_button:
                    student = db.collection("Students").document(search_id).get()
                    if student.exists:
                        student_data = student.to_dict()
                        st.markdown(f"""
                        <div class="search-result">
                            <div class="search-result-item">
                                <div class="search-result-label">Mã số sinh viên:</div>
                                <div>{student.id}</div>
                            </div>
                            <div class="search-result-item">
                                <div class="search-result-label">Họ và tên:</div>
                                <div>{student_data.get('Name', '')}</div>
                            </div>
                            <div class="search-result-item">
                                <div class="search-result-label">Thẻ sinh viên:</div>
                                <img src="{student_data.get('TheSV', '')}" alt="Thẻ Sinh viên">
                            </div>
                            <div class="search-result-item">
                                <div class="search-result-label">Ảnh chân dung:</div>
                                <img src="{student_data.get('ChanDung', '')}" alt="Ảnh Chân dung">
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Không tìm thấy sinh viên với ID này!")
        
        else:  # Tìm theo Tên
            col1, col2 = st.columns([3,1])
            with col1:
                search_name = st.text_input("Nhập tên sinh viên cần tìm")
            with col2:
                search_button = st.button("Tìm kiếm")
                if search_button:
                    students = db.collection("Students").where(filter=("Name", ">=", search_name))\
                                                    .where(filter=("Name", "<=", search_name + '\uf8ff'))\
                                                    .get()
                    
                    if not students:
                        st.warning("Không tìm thấy sinh viên với tên này!")
                    else:
                        for student in students:
                            student_data = student.to_dict()
                            st.markdown(f"""
                            <div class="search-result">
                                <div class="search-result-item">
                                    <div class="search-result-label">Mã số sinh viên:</div>
                                    <div>{student.id}</div>
                                </div>
                                <div class="search-result-item">
                                    <div class="search-result-label">Họ và tên:</div>
                                    <div>{student_data.get('Name', '')}</div>
                                </div>
                                <div class="search-result-item">
                                    <div class="search-result-label">Thẻ sinh viên:</div>
                                    <img src="{student_data.get('TheSV', '')}" alt="Thẻ Sinh viên">
                                </div>
                                <div class="search-result-item">
                                    <div class="search-result-label">Ảnh chân dung:</div>
                                    <img src="{student_data.get('ChanDung', '')}" alt="Ảnh Chân dung">
                                </div>
                            </div>
                            <div class="search-result-divider"></div>
                            """, unsafe_allow_html=True)
        
        col7, col8 = st.columns(2)
        with col8:
            if st.button("Quay lại"):
                st.session_state.current_action = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Trong tab2
with tab2:
    st.header("2. Kiểm tra ảnh chân dung và ảnh thẻ sinh viên")
    try:
        haar_cascade = init_haar_cascade()
        yunet_detector, sface_recognizer = init_yunet_sface()

        st.markdown('<div class="face-comparison">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ảnh Chân dung")
            portrait_image = st.file_uploader("Tải lên ảnh chân dung", type=['jpg', 'jpeg', 'png'], key="portrait")

        with col2:
            st.subheader("Ảnh Thẻ Sinh viên")
            id_image = st.file_uploader("Tải lên ảnh thẻ sinh viên", type=['jpg', 'jpeg', 'png'], key="id")

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

                st.markdown('<div class="comparison-result">', unsafe_allow_html=True)
                st.subheader("Kết quả So sánh")
                st.markdown(f'<div class="comparison-score">Độ tương đồng: {similarity_score:.4f}</div>', unsafe_allow_html=True)

                if similarity_score > 0.3:
                    st.success("Ảnh chân dung và ảnh thẻ sinh viên KHỚP!")
                    color = (0, 255, 0)
                else:
                    st.error("Ảnh chân dung và ảnh thẻ sinh viên KHÔNG KHỚP!")
                    color = (0, 0, 255)

                # Vẽ hình chữ nhật
                portrait_img_with_rect = draw_faces(portrait_img.copy(), [largest_face])
                id_img_with_rect = draw_faces(id_img.copy(), id_face, is_haar=False)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(portrait_img_with_rect, caption="Ảnh Chân dung", use_column_width=True)
                with col2:
                    st.image(id_img_with_rect, caption="Ảnh Thẻ Sinh viên", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Không thể phát hiện khuôn mặt trong một hoặc cả hai ảnh. Vui lòng thử lại với ảnh khác.")
        elif check_button:
            st.warning("Vui lòng tải lên cả ảnh chân dung và ảnh thẻ sinh viên trước khi kiểm tra.")
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Lỗi khi khởi tạo các model nhận diện khuôn mặt: {str(e)}")
# Trong tab3
with tab3:
    st.header("3. Tìm kiếm Sinh viên trong Ảnh Lớp học")
    try:
        # Khởi tạo các model
        haar_cascade = init_haar_cascade()
        yunet_detector, sface_recognizer = init_yunet_sface()

        # Lấy danh sách sinh viên từ database
        students_data = get_student_data()
        
        # Upload ảnh lớp học
        st.header("Ảnh Lớp học")
        class_image = st.file_uploader("Tải lên ảnh lớp học", type=['jpg', 'jpeg', 'png'], key="class_photo")

        # Thêm thanh trượt để điều chỉnh ngưỡng
        threshold = st.slider("Ngưỡng nhận dạng (0-1)", 0.0, 1.0, 0.3, 0.001)

        # Thêm nút tìm kiếm
        search_button = st.button("Tìm kiếm tất cả sinh viên")

        if class_image and search_button:
            try:
                # Xử lý ảnh lớp học trước
                class_img, class_faces, class_features = process_class_image(
                    class_image, haar_cascade, sface_recognizer)
                
                if len(class_faces) > 0:
                    st.write(f"Đã phát hiện {len(class_faces)} khuôn mặt trong ảnh lớp học")
                    
                    # Tạo bản sao của ảnh lớp học để vẽ kết quả
                    result_img = class_img.copy()
                    
                    # Duyệt qua từng sinh viên trong database
                    found_students = []
                    for student in students_data:
                        try:
                            # Lấy và xử lý ảnh chân dung từ URL
                            response = requests.get(student['ChanDung'])
                            student_image = BytesIO(response.content)
                            
                            # Xử lý ảnh sinh viên
                            student_img, student_face, student_feature = process_student_image(
                                student_image, haar_cascade, sface_recognizer)
                            
                            if student_feature is not None:
                                # So sánh với từng khuôn mặt trong lớp học
                                for i, (face, class_feature) in enumerate(class_features):
                                    score = compare_faces(student_feature, class_feature, sface_recognizer)
                                    if score > threshold:
                                        found_students.append({
                                            'student': student,
                                            'face': face,
                                            'score': score,
                                            'face_index': i
                                        })
                        except Exception as e:
                            st.warning(f"Không thể xử lý ảnh của sinh viên {student['Name']}: {str(e)}")
                    
                    # Vẽ kết quả
                    for match in found_students:
                        x, y, w, h = match['face']
                        student = match['student']
                        score = match['score']
                        
                        # Vẽ hình chữ nhật xanh và tên sinh viên
                        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        text = f"{student['Name']} ({score:.2f})"
                        cv2.putText(result_img, text, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Hiển thị kết quả
                    st.header("Kết quả Tìm kiếm")
                    st.image(result_img, caption="Kết quả nhận diện", use_column_width=True)
                    
                    # Hiển thị danh sách sinh viên được tìm thấy
                    if found_students:
                        st.success(f"Tìm thấy {len(found_students)} sinh viên trong ảnh lớp học")
                        st.write("Danh sách sinh viên được tìm thấy:")
                        for match in found_students:
                            student = match['student']
                            score = match['score']
                            st.write(f"- {student['Name']} (MSSV: {student['ID']}, Độ tương đồng: {score:.4f})")
                    else:
                        st.warning("Không tìm thấy sinh viên nào trong ảnh lớp học")
                else:
                    st.error("Không thể phát hiện khuôn mặt trong ảnh lớp học")
            except Exception as e:
                st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
        elif search_button:
            st.warning("Vui lòng tải lên ảnh lớp học trước khi tìm kiếm")

    except Exception as e:
        st.error(f"Lỗi khi khởi tạo các model nhận diện khuôn mặt: {str(e)}")