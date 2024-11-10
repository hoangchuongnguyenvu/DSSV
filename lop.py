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

st.set_page_config(layout="wide")

# Kh�i t�o Firebase (ch� th�c hi�n m�t l�n)
if not firebase_admin._apps:
    cred = credentials.Certificate("hchuong-firebase-adminsdk-1m82k-829fb1690b.json")
    firebase_admin.initialize_app(cred, {'storageBucket': 'hchuong.appspot.com'})

# K�t n�i �n Firestore v� Storage
db = firestore.client()
bucket = storage.bucket()

# Kh�i t�o YuNet v� SFace
yunet_path = "face_detection_yunet_2023mar.onnx"
sface_path = "face_recognition_sface_2021dec.onnx"

face_detector = cv2.FaceDetectorYN.create(yunet_path, "", (0, 0))
face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

# Kh�i t�o session state
if 'current_action' not in st.session_state:
    st.session_state.current_action = None

# CSS � t�o ki�u cho �ng d�ng
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
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-top: 10px;
    }
    .search-result img {
        max-width: 100px;
        max-height: 100px;
    }
    .stDataFrame {
        width: 100%;
        max-width: none !important;
    }
</style>
""", unsafe_allow_html=True)

# H�m � t�i l�n h�nh �nh v�o Firebase Storage
def upload_image(file):
    if file is not None:
        file_name = str(uuid.uuid4()) + "." + file.name.split(".")[-1]
        blob = bucket.blob(file_name)
        blob.upload_from_file(file)
        blob.make_public()
        return blob.public_url
    return None

# H�m � t�i �nh t� URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# H�m � ph�t hi�n khu�n m�t b�ng YuNet
def detect_face_yunet(image):
    height, width, _ = image.shape
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(image)
    if faces is not None and len(faces) > 0:
        face = faces[0]
        x, y, w, h = face[:4].astype(int)
        return image[y:y+h, x:x+w], face
    return None, None

# H�m � tr�ch xu�t �c tr�ng khu�n m�t b�ng SFace
def extract_face_feature(face_image, face_box):
    if face_image is None or face_box is None:
        return None
    face_align = face_recognizer.alignCrop(face_image, face_box)
    face_feature = face_recognizer.feature(face_align)
    return face_feature

# H�m � ph�t hi�n v� nh�n d�ng khu�n m�t trong �nh l�p h�c
def detect_students_in_class(class_image, student_features):
    height, width, _ = class_image.shape
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(class_image)
    
    recognized_students = []
    
    if faces is not None:
        for face in faces:
            face_align = face_recognizer.alignCrop(class_image, face)
            face_feature = face_recognizer.feature(face_align)
            
            for student_id, student_feature in student_features.items():
                if student_feature is not None:
                    score = face_recognizer.match(face_feature, student_feature, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > 0.363:  # Ng��ng n�y c� th� i�u ch�nh
                        student = db.collection("Students").document(student_id).get().to_dict()
                        recognized_students.append(student['Name'])
                        break
    
    return recognized_students

# H�m � l�y v� x� l� d� li�u sinh vi�n
def get_student_data_and_features():
    students_ref = db.collection("Students")
    students = students_ref.get()
    table_data = []
    student_features = {}
    for student in students:
        student_data = student.to_dict()
        table_data.append({
            "ID": student.id,
            "Name": student_data.get("Name", ""),
            "TheSV": student_data.get("TheSV", ""),
            "ChanDung": student_data.get("ChanDung", "")
        })
        
        # X� l� �nh ch�n dung v� tr�ch xu�t �c tr�ng
        chandung_image = load_image_from_url(student_data.get("ChanDung", ""))
        face, face_box = detect_face_yunet(chandung_image)
        if face is not None:
            feature = extract_face_feature(chandung_image, face_box)
            student_features[student.id] = feature
        else:
            student_features[student.id] = None
    
    return table_data, student_features

# Thanh b�n cho c�c ch�c nng
st.sidebar.header("Ch�c nng")
action = st.sidebar.radio("Ch�n ch�c nng:", ("Qu�n l� Sinh vi�n", "Ph�t hi�n Sinh vi�n trong �nh L�p h�c"))

if action == "Qu�n l� Sinh vi�n":
    st.header("1. Qu�n l� Sinh vi�n")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Th�m Sinh vi�n m�i"):
            st.session_state.current_action = 'add'
    with col2:
        if st.button("T�m ki�m Sinh vi�n"):
            st.session_state.current_action = 'search'

    # Ch�c nng th�m sinh vi�n m�i
    if st.session_state.current_action == 'add':
        st.subheader("Th�m Sinh vi�n m�i")
        new_id = st.text_input("ID")
        new_name = st.text_input("T�n")
        new_thesv = st.file_uploader("Th� Sinh vi�n", type=["jpg", "png", "jpeg"])
        new_chandung = st.file_uploader("�nh Ch�n dung", type=["jpg", "png", "jpeg"])

        if st.button("X�c nh�n th�m"):
            if new_id and new_name and new_thesv and new_chandung:
                thesv_url = upload_image(new_thesv)
                chandung_url = upload_image(new_chandung)
                db.collection("Students").document(new_id).set({
                    "Name": new_name,
                    "TheSV": thesv_url,
                    "ChanDung": chandung_url
                })
                st.success("� th�m sinh vi�n m�i!")
                st.session_state.current_action = None
                st.rerun()
            else:
                st.warning("Vui l�ng i�n �y � th�ng tin!")

    # Ch�c nng t�m ki�m
    elif st.session_state.current_action == 'search':
        st.subheader("T�m ki�m Sinh vi�n")
        search_id = st.text_input("Nh�p ID sinh vi�n c�n t�m")
        if st.button("X�c nh�n t�m ki�m"):
            student = db.collection("Students").document(search_id).get()
            if student.exists:
                student_data = student.to_dict()
                st.markdown(f"""
                <div class="search-result">
                    <div>ID: {student.id}</div>
                    <div>T�n: {student_data.get('Name', '')}</div>
                    <img src="{student_data.get('TheSV', '')}" alt="Th� Sinh vi�n">
                    <img src="{student_data.get('ChanDung', '')}" alt="�nh Ch�n dung">
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Kh�ng t�m th�y sinh vi�n v�i ID n�y!")

    # Hi�n th� b�ng d� li�u v�i ch�c nng ch�nh s�a v� x�a
    st.subheader("Danh s�ch Sinh vi�n")
    table_data, _ = get_student_data_and_features()
    df = pd.DataFrame(table_data)

    df['Edit'] = False
    df['Delete'] = False

    edited_df = st.data_editor(
        df,
        hide_index=True,
        column_config={
            "Edit": st.column_config.CheckboxColumn("Ch�nh s�a", default=False, width="small"),
            "Delete": st.column_config.CheckboxColumn("X�a", default=False, width="small"),
            "TheSV": st.column_config.ImageColumn("Th� SV", help="Th� sinh vi�n", width="medium"),
            "ChanDung": st.column_config.ImageColumn("Ch�n dung", help="�nh ch�n dung", width="medium"),
            "ID": st.column_config.TextColumn("ID", help="ID sinh vi�n", width="medium"),
            "Name": st.column_config.TextColumn("T�n", help="T�n sinh vi�n", width="large"),
        },
        disabled=["ID", "Name", "TheSV", "ChanDung"],
        use_container_width=True,
        num_rows="dynamic"
    )

    # X� l� ch�nh s�a
    students_to_edit = edited_df[edited_df['Edit']]
    if not students_to_edit.empty:
        for _, student in students_to_edit.iterrows():
            st.subheader(f"Ch�nh s�a th�ng tin cho sinh vi�n: {student['Name']}")
            edit_id = st.text_input(f"ID m�i cho {student['ID']}", value=student['ID'])
            edit_name = st.text_input(f"T�n m�i cho {student['ID']}", value=student['Name'])
            edit_thesv = st.file_uploader(f"Th� Sinh vi�n m�i cho {student['ID']}", type=["jpg", "png", "jpeg"])
            edit_chandung = st.file_uploader(f"�nh Ch�n dung m�i cho {student['ID']}", type=["jpg", "png", "jpeg"])

            if st.button(f"C�p nh�t cho {student['ID']}"):
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
                    st.success(f"� c�p nh�t th�ng tin v� ID sinh vi�n t� {student['ID']} th�nh {edit_id}!")
                else:
                    db.collection("Students").document(student['ID']).update(update_data)
                    st.success(f"� c�p nh�t th�ng tin sinh vi�n {student['ID']}!")
                
                st.rerun()

    # X� l� x�a
    students_to_delete = edited_df[edited_df['Delete']]
    if not students_to_delete.empty:
        for _, student in students_to_delete.iterrows():
            st.subheader(f"X�c nh�n x�a sinh vi�n: {student['Name']}")
            if st.button(f"X�c nh�n x�a {student['ID']}"):
                db.collection("Students").document(student['ID']).delete()
                st.success(f"� x�a sinh vi�n {student['ID']}!")
                st.rerun()

elif action == "Ph�t hi�n Sinh vi�n trong �nh L�p h�c":
    st.header("2. Ph�t hi�n Sinh vi�n trong �nh L�p h�c")

    uploaded_class_image = st.file_uploader("T�i l�n �nh l�p h�c", type=["jpg", "png", "jpeg"])

    if uploaded_class_image is not None:
        class_image = cv2.imdecode(np.frombuffer(uploaded_class_image.read(), np.uint8), 1)
        st.image(class_image, caption="�nh L�p h�c", use_column_width=True)
        
        if st.button("Ph�t hi�n Sinh vi�n"):
            with st.spinner('ang x� l�...'):
                _, student_features = get_student_data_and_features()
                recognized_students = detect_students_in_class(class_image, student_features)
            
            st.subheader("Sinh vi�n ��c nh�n d�ng:")
            if recognized_students:
                for student in recognized_students:
                    st.write(f"- {student}")
            else :
                st.write("Kh�ng nh�n d�ng ��c sinh vi�n n�o trong �nh.")

            # V� khung nh�n d�ng l�n �nh
            height, width, _ = class_image.shape
            face_detector.setInputSize((width, height))
            _, faces = face_detector.detect(class_image)
            
            if faces is not None:
                for face in faces:
                    x, y, w, h = face[:4].astype(int)
                    cv2.rectangle(class_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                st.image(cv2.cvtColor(class_image, cv2.COLOR_BGR2RGB), caption="�nh L�p h�c v�i khung nh�n d�ng", use_column_width=True)