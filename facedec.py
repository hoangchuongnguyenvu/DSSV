import cv2
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os

# 1. Tạo Haar-like features
def create_haar_features():
    features = []
    for r in range(5):  # Giảm số lượng loại feature để tăng tốc độ
        for x in range(24):
            for y in range(24):
                for w in range(1, 25-x):
                    for h in range(1, 25-y):
                        if r == 0 and 2*w <= 24-x:  # Hai hình chữ nhật ngang
                            features.append(((x,y,w,h), (x+w,y,w,h)))
                        elif r == 1 and 2*h <= 24-y:  # Hai hình chữ nhật dọc
                            features.append(((x,y,w,h), (x,y+h,w,h)))
                        elif r == 2 and 3*w <= 24-x:  # Ba hình chữ nhật ngang
                            features.append(((x,y,w,h), (x+w,y,w,h), (x+2*w,y,w,h)))
                        elif r == 3 and 3*h <= 24-y:  # Ba hình chữ nhật dọc
                            features.append(((x,y,w,h), (x,y+h,w,h), (x,y+2*h,w,h)))
                        elif r == 4 and 2*w <= 24-x and 2*h <= 24-y:  # Bốn hình chữ nhật
                            features.append(((x,y,w,h), (x+w,y,w,h), (x,y+h,w,h), (x+w,y+h,w,h)))
    return features

# 2. Tính giá trị Haar-like feature
def compute_haar_feature(ii, feature):
    value = 0
    for i, rect in enumerate(feature):
        x, y, w, h = rect
        rect_sum = ii[y+h, x+w] + ii[y, x] - ii[y, x+w] - ii[y+h, x]
        value += rect_sum * (1 if i % 2 == 0 else -1)
    return value

# 3. Trích xuất đặc trưng Haar từ ảnh
def extract_haar_features(image, haar_features):
    ii = cv2.integral(image)
    return [compute_haar_feature(ii, feature) for feature in haar_features]

# 4. Đọc và chuẩn bị dữ liệu
def load_images(directory, label, haar_features):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.shape == (24, 24):
                features = extract_haar_features(img, haar_features)
                images.append(features)
                labels.append(label)
    return images, labels

# 5. Train mô hình với AdaBoost và KNN
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
    adaboost.fit(X_train, y_train)
    
    feature_importance = adaboost.feature_importances_
    selected_features = feature_importance > np.mean(feature_importance)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train[:, selected_features], y_train)
    
    accuracy = knn.score(X_test[:, selected_features], y_test)
    print(f"Độ chính xác: {accuracy}")
    
    return adaboost, knn, selected_features

# 6. Phát hiện gương mặt trong ảnh lớn
def detect_faces(image, haar_features, adaboost, knn, selected_features, window_size=24, step=4):
    faces = []
    height, width = image.shape[:2]
    for y in range(0, height - window_size, step):
        for x in range(0, width - window_size, step):
            window = image[y:y+window_size, x:x+window_size]
            features = extract_haar_features(window, haar_features)
            features = np.array(features).reshape(1, -1)
            prediction = knn.predict(features[:, selected_features])
            if prediction[0] == 1:
                faces.append((x, y, window_size, window_size))
    return faces

# 7. Vẽ hình vuông xung quanh gương mặt phát hiện được
def draw_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

# Sử dụng
if __name__ == "__main__":
    face_dir = 'faces_and_non_faces_data/faces_24x24'
    non_face_dir = 'faces_and_non_faces_data/non_faces_24x24'
    
    haar_features = create_haar_features()
    print(f"Số lượng Haar-like features: {len(haar_features)}")
    
    # Đọc dữ liệu
    face_images, face_labels = load_images(face_dir, 1, haar_features)
    non_face_images, non_face_labels = load_images(non_face_dir, 0, haar_features)
    
    X = np.array(face_images + non_face_images)
    y = np.array(face_labels + non_face_labels)
    
    # Train mô hình
    adaboost, knn, selected_features = train_model(X, y)
    
    # Phát hiện gương mặt trong ảnh mới
    test_image = cv2.imread("oar2.jpg", cv2.IMREAD_GRAYSCALE)
    detected_faces = detect_faces(test_image, haar_features, adaboost, knn, selected_features)
    
    # Vẽ hình vuông xung quanh gương mặt
    result_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
    result_image = draw_faces(result_image, detected_faces)
    
    # Hiển thị kết quả
    cv2.imshow("Detected Faces", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()