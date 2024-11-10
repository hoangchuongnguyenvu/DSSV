# -*- coding: utf-8 -*-

import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import uuid

st.set_page_config(layout="wide")

# Khởi tạo Firebase (chỉ thực hiện một lần)
if not firebase_admin._apps:
    cred = credentials.Certificate("hchuong-firebase-adminsdk-1m82k-829fb1690b.json")
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

# CSS để tạo kiểu cho ứng dụng
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

# Tiêu đề ứng dụng
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
        edit_thesv = st.file_uploader("Thẻ Sinh viên mới", type=["jpg", "png", "jpeg"])
        edit_chandung = st.file_uploader("Ảnh Chân dung mới", type=["jpg", "png", "jpeg"])

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
        new_thesv = st.file_uploader("Thẻ Sinh viên", type=["jpg", "png", "jpeg"])
    with col4:
        new_chandung = st.file_uploader("Ảnh Chân dung", type=["jpg", "png", "jpeg"])

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