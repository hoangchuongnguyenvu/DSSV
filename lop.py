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

# Khßi t¡o Firebase (chÉ thñc hiÇn mÙt l§n)
if not firebase_admin._apps:
    cred = credentials.Certificate("hchuong-firebase-adminsdk-1m82k-829fb1690b.json")
    firebase_admin.initialize_app(cred, {'storageBucket': 'hchuong.appspot.com'})

# K¿t nÑi ¿n Firestore và Storage
db = firestore.client()
bucket = storage.bucket()

# Khßi t¡o YuNet và SFace
yunet_path = "face_detection_yunet_2023mar.onnx"
sface_path = "face_recognition_sface_2021dec.onnx"

face_detector = cv2.FaceDetectorYN.create(yunet_path, "", (0, 0))
face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

# Khßi t¡o session state
if 'current_action' not in st.session_state:
    st.session_state.current_action = None

# CSS Ã t¡o kiÃu cho éng dång
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

# Hàm Ã t£i lên hình £nh vào Firebase Storage
def upload_image(file):
    if file is not None:
        file_name = str(uuid.uuid4()) + "." + file.name.split(".")[-1]
        blob = bucket.blob(file_name)
        blob.upload_from_file(file)
        blob.make_public()
        return blob.public_url
    return None

# Hàm Ã t£i £nh të URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Hàm Ã phát hiÇn khuôn m·t b±ng YuNet
def detect_face_yunet(image):
    height, width, _ = image.shape
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(image)
    if faces is not None and len(faces) > 0:
        face = faces[0]
        x, y, w, h = face[:4].astype(int)
        return image[y:y+h, x:x+w], face
    return None, None

# Hàm Ã trích xu¥t ·c tr°ng khuôn m·t b±ng SFace
def extract_face_feature(face_image, face_box):
    if face_image is None or face_box is None:
        return None
    face_align = face_recognizer.alignCrop(face_image, face_box)
    face_feature = face_recognizer.feature(face_align)
    return face_feature

# Hàm Ã phát hiÇn và nh­n d¡ng khuôn m·t trong £nh lÛp hÍc
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
                    if score > 0.363:  # Ng°áng này có thÃ iÁu chÉnh
                        student = db.collection("Students").document(student_id).get().to_dict()
                        recognized_students.append(student['Name'])
                        break
    
    return recognized_students

# Hàm Ã l¥y và xí lý dï liÇu sinh viên
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
        
        # Xí lý £nh chân dung và trích xu¥t ·c tr°ng
        chandung_image = load_image_from_url(student_data.get("ChanDung", ""))
        face, face_box = detect_face_yunet(chandung_image)
        if face is not None:
            feature = extract_face_feature(chandung_image, face_box)
            student_features[student.id] = feature
        else:
            student_features[student.id] = None
    
    return table_data, student_features

# Thanh bên cho các chéc nng
st.sidebar.header("Chéc nng")
action = st.sidebar.radio("ChÍn chéc nng:", ("Qu£n lý Sinh viên", "Phát hiÇn Sinh viên trong ¢nh LÛp hÍc"))

if action == "Qu£n lý Sinh viên":
    st.header("1. Qu£n lý Sinh viên")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Thêm Sinh viên mÛi"):
            st.session_state.current_action = 'add'
    with col2:
        if st.button("Tìm ki¿m Sinh viên"):
            st.session_state.current_action = 'search'

    # Chéc nng thêm sinh viên mÛi
    if st.session_state.current_action == 'add':
        st.subheader("Thêm Sinh viên mÛi")
        new_id = st.text_input("ID")
        new_name = st.text_input("Tên")
        new_thesv = st.file_uploader("Th» Sinh viên", type=["jpg", "png", "jpeg"])
        new_chandung = st.file_uploader("¢nh Chân dung", type=["jpg", "png", "jpeg"])

        if st.button("Xác nh­n thêm"):
            if new_id and new_name and new_thesv and new_chandung:
                thesv_url = upload_image(new_thesv)
                chandung_url = upload_image(new_chandung)
                db.collection("Students").document(new_id).set({
                    "Name": new_name,
                    "TheSV": thesv_url,
                    "ChanDung": chandung_url
                })
                st.success("ã thêm sinh viên mÛi!")
                st.session_state.current_action = None
                st.rerun()
            else:
                st.warning("Vui lòng iÁn §y ç thông tin!")

    # Chéc nng tìm ki¿m
    elif st.session_state.current_action == 'search':
        st.subheader("Tìm ki¿m Sinh viên")
        search_id = st.text_input("Nh­p ID sinh viên c§n tìm")
        if st.button("Xác nh­n tìm ki¿m"):
            student = db.collection("Students").document(search_id).get()
            if student.exists:
                student_data = student.to_dict()
                st.markdown(f"""
                <div class="search-result">
                    <div>ID: {student.id}</div>
                    <div>Tên: {student_data.get('Name', '')}</div>
                    <img src="{student_data.get('TheSV', '')}" alt="Th» Sinh viên">
                    <img src="{student_data.get('ChanDung', '')}" alt="¢nh Chân dung">
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Không tìm th¥y sinh viên vÛi ID này!")

    # HiÃn thË b£ng dï liÇu vÛi chéc nng chÉnh sía và xóa
    st.subheader("Danh sách Sinh viên")
    table_data, _ = get_student_data_and_features()
    df = pd.DataFrame(table_data)

    df['Edit'] = False
    df['Delete'] = False

    edited_df = st.data_editor(
        df,
        hide_index=True,
        column_config={
            "Edit": st.column_config.CheckboxColumn("ChÉnh sía", default=False, width="small"),
            "Delete": st.column_config.CheckboxColumn("Xóa", default=False, width="small"),
            "TheSV": st.column_config.ImageColumn("Th» SV", help="Th» sinh viên", width="medium"),
            "ChanDung": st.column_config.ImageColumn("Chân dung", help="¢nh chân dung", width="medium"),
            "ID": st.column_config.TextColumn("ID", help="ID sinh viên", width="medium"),
            "Name": st.column_config.TextColumn("Tên", help="Tên sinh viên", width="large"),
        },
        disabled=["ID", "Name", "TheSV", "ChanDung"],
        use_container_width=True,
        num_rows="dynamic"
    )

    # Xí lý chÉnh sía
    students_to_edit = edited_df[edited_df['Edit']]
    if not students_to_edit.empty:
        for _, student in students_to_edit.iterrows():
            st.subheader(f"ChÉnh sía thông tin cho sinh viên: {student['Name']}")
            edit_id = st.text_input(f"ID mÛi cho {student['ID']}", value=student['ID'])
            edit_name = st.text_input(f"Tên mÛi cho {student['ID']}", value=student['Name'])
            edit_thesv = st.file_uploader(f"Th» Sinh viên mÛi cho {student['ID']}", type=["jpg", "png", "jpeg"])
            edit_chandung = st.file_uploader(f"¢nh Chân dung mÛi cho {student['ID']}", type=["jpg", "png", "jpeg"])

            if st.button(f"C­p nh­t cho {student['ID']}"):
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
                    st.success(f"ã c­p nh­t thông tin và ID sinh viên të {student['ID']} thành {edit_id}!")
                else:
                    db.collection("Students").document(student['ID']).update(update_data)
                    st.success(f"ã c­p nh­t thông tin sinh viên {student['ID']}!")
                
                st.rerun()

    # Xí lý xóa
    students_to_delete = edited_df[edited_df['Delete']]
    if not students_to_delete.empty:
        for _, student in students_to_delete.iterrows():
            st.subheader(f"Xác nh­n xóa sinh viên: {student['Name']}")
            if st.button(f"Xác nh­n xóa {student['ID']}"):
                db.collection("Students").document(student['ID']).delete()
                st.success(f"ã xóa sinh viên {student['ID']}!")
                st.rerun()

elif action == "Phát hiÇn Sinh viên trong ¢nh LÛp hÍc":
    st.header("2. Phát hiÇn Sinh viên trong ¢nh LÛp hÍc")

    uploaded_class_image = st.file_uploader("T£i lên £nh lÛp hÍc", type=["jpg", "png", "jpeg"])

    if uploaded_class_image is not None:
        class_image = cv2.imdecode(np.frombuffer(uploaded_class_image.read(), np.uint8), 1)
        st.image(class_image, caption="¢nh LÛp hÍc", use_column_width=True)
        
        if st.button("Phát hiÇn Sinh viên"):
            with st.spinner('ang xí lý...'):
                _, student_features = get_student_data_and_features()
                recognized_students = detect_students_in_class(class_image, student_features)
            
            st.subheader("Sinh viên °ãc nh­n d¡ng:")
            if recognized_students:
                for student in recognized_students:
                    st.write(f"- {student}")
            else :
                st.write("Không nh­n d¡ng °ãc sinh viên nào trong £nh.")

            # V½ khung nh­n d¡ng lên £nh
            height, width, _ = class_image.shape
            face_detector.setInputSize((width, height))
            _, faces = face_detector.detect(class_image)
            
            if faces is not None:
                for face in faces:
                    x, y, w, h = face[:4].astype(int)
                    cv2.rectangle(class_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                st.image(cv2.cvtColor(class_image, cv2.COLOR_BGR2RGB), caption="¢nh LÛp hÍc vÛi khung nh­n d¡ng", use_column_width=True)