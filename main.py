import cv2
import numpy as np
import face_recognition
import os
import time
import requests
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
from database import create_table, get_person

# ------------------- إعداد -------------------
create_table()

model = YOLO("./yolov8n.pt")

TOKEN = "<Your Telegram Token>"
CHAT_ID = "<Your Telegram Chat ID>"

# Tkinter
root = tk.Tk()
root.withdraw()

# ------------------- تحميل الوجوه -------------------
path = 'target'
images = []
names = []

for file in os.listdir(path):
    img = cv2.imread(f"{path}/{file}")
    if img is not None:
        images.append(img)
        names.append(os.path.splitext(file)[0].lower())

def encode_faces(images):
    encodes = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        e = face_recognition.face_encodings(rgb)
        if e:
            encodes.append(e[0])
    return encodes

known_encodes = encode_faces(images)
print("Encoding Complete")

# ------------------- Telegram -------------------
def send_telegram(data, face):
    name, age, status, action = data
    _, img = cv2.imencode('.jpg', face)

    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    caption = f"{name}\nAge: {age}\nStatus: {status}\nAction: {action}"

    requests.post(url,
        files={"photo": img.tobytes()},
        data={"chat_id": CHAT_ID, "caption": caption}
    )

# ------------------- Logging -------------------
def save_log(name, face):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    filename = f"logs/{name}_{int(time.time())}.jpg"
    cv2.imwrite(filename, face)

# ------------------- Popup -------------------
def show_popup(data, face):
    name, age, status, action = data

    win = tk.Toplevel()
    win.title(name)
    win.geometry("300x400")

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(face).resize((250, 250))
    photo = ImageTk.PhotoImage(img)

    lbl = tk.Label(win, image=photo)
    lbl.image = photo
    lbl.pack(pady=10)

    tk.Label(win, text=f"Name: {name}").pack()
    tk.Label(win, text=f"Age: {age}").pack()
    tk.Label(win, text=f"Status: {status}").pack()
    tk.Label(win, text=f"Action: {action}").pack()

    tk.Button(win, text="Close", command=win.destroy).pack(pady=10)

# ------------------- التحكم -------------------
last_seen = {}
DELAY = 30
frame_count = 0

# ------------------- الكاميرا -------------------
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    root.update()
    frame_count += 1

    # تشغيل YOLO كل 3 فريم
    if frame_count % 3 == 0:

        small = cv2.resize(frame, (0,0), None, 0.5, 0.5)
        results = model(small)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # إعادة الحجم
                x1, y1, x2, y2 = x1*2, y1*2, x2*2, y2*2

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                enc = face_recognition.face_encodings(rgb)

                if not enc:
                    continue

                enc = enc[0]
                matches = face_recognition.compare_faces(known_encodes, enc)
                dist = face_recognition.face_distance(known_encodes, enc)

                if len(dist) == 0:
                    continue

                i = np.argmin(dist)

                if not (matches[i] and dist[i] < 0.45):
                    continue

                name = names[i]

                now = time.time()
                if name not in last_seen or now - last_seen[name] > DELAY:
                    data = get_person(name)

                    if data:
                        root.after(0, show_popup, data, face)
                        send_telegram(data, face) 
                        save_log(name, face)
                        last_seen[name] = now

                
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, name,(x1,y2-10),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    
    cv2.imshow("Face Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
