import cv2
import mediapipe as mp
import os
import json
import tkinter as tk
from datetime import datetime
from threading import Thread

# Mediapipe Face Mesh'i başlat
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Masaüstünde duygular.json dosyasını oluştur
folder_path = os.path.join(os.path.expanduser("~"), "Desktop", "duygular.json")
if not os.path.exists(folder_path):
    with open(folder_path, "w") as f:
        json.dump([], f)

last_detected_emotion = None

def save_emotion(emotion):
    global last_detected_emotion
    if emotion != last_detected_emotion:
        with open(folder_path, "r+") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            
            data.append({"emotion": emotion, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            f.seek(0)
            json.dump(data, f, indent=4)
        last_detected_emotion = emotion
        update_emotion_label(emotion)

def analyze_emotions(frame):
    try:
        frame = cv2.resize(frame, (1280, 960))  # Piksel sayısını artır
        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Gürültüyü azalt
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)  # Daha fazla nokta çiz
                
                # Belirli landmark noktalarını al
                landmarks = face_landmarks.landmark
                
                left_eyebrow = landmarks[70].y
                right_eyebrow = landmarks[300].y
                left_eye = landmarks[159].y
                right_eye = landmarks[386].y
                mouth_top = landmarks[13].y
                mouth_bottom = landmarks[14].y
                mouth_left = landmarks[61].x
                mouth_right = landmarks[291].x
                
                eye_distance = abs(left_eye - right_eye)
                mouth_height = abs(mouth_bottom - mouth_top)
                mouth_width = abs(mouth_right - mouth_left)
                eyebrow_distance = abs(left_eyebrow - right_eyebrow)
                eyebrow_position = (left_eyebrow + right_eyebrow) / 2
                
                # Duygu analizi
                if mouth_height > 0.05 and mouth_width > 0.05:
                    detected_emotion = "Mutlu 😀"
                elif left_eyebrow < left_eye and right_eyebrow < right_eye and eye_distance < 0.02:
                    detected_emotion = "Kızgın 😠"
                elif mouth_height < 0.02 and eyebrow_distance > 0.02 and eyebrow_position < left_eye:
                    detected_emotion = "Üzgün 😢"
                elif mouth_height < 0.02 and eyebrow_distance < 0.02 and eyebrow_position > left_eye:
                    detected_emotion = "Mutsuz 😞"
                elif eye_distance > 0.02 and mouth_height < 0.02 and eyebrow_position == left_eye:
                    detected_emotion = "Nötr 😐"
                elif eye_distance > 0.03 and mouth_height > 0.03:
                    detected_emotion = "Şaşkın 😲"
                else:
                    detected_emotion = "Nötr 😐"
                
                save_emotion(detected_emotion)
        
    except Exception as e:
        print(f"Hata oluştu: {e}")

def start_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        analyze_emotions(frame)
        cv2.imshow("Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def update_emotion_label(emotion):
    emotion_label.config(text=f"Duygu: {emotion}")

def run_camera_thread():
    camera_thread = Thread(target=start_camera)
    camera_thread.daemon = True
    camera_thread.start()

# Tkinter arayüzü
root = tk.Tk()
root.title("Duygu Analizi")
root.geometry("400x200")

# İkon ekleme (ico dosyasının yolu)
icon_path = os.path.join(os.getcwd(), "btc.ico")  # Kod klasöründeki ico dosyasını al
if os.path.exists(icon_path):
    root.iconbitmap(icon_path)
else:
    print(f"İkon dosyası bulunamadı: {icon_path}")

emotion_label = tk.Label(root, text="Duygu: Algılanıyor...", font=("Arial", 14))
emotion_label.pack(pady=20)

start_button = tk.Button(root, text="Kamerayı Başlat", command=run_camera_thread)
start_button.pack()

root.mainloop()
