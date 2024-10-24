import cv2
import mediapipe as mp
import numpy as np

# เริ่มการจับภาพจากกล้อง
cap = cv2.VideoCapture(0)

# สร้างวัตถุสำหรับการตรวจจับมือ
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # ดึงตำแหน่งของจุดที่ต้องการ
            thumb = handLms.landmark[4]  # ปลายนิ้วโป้ง
            index = handLms.landmark[8]  # นิ้วชี้
            middle = handLms.landmark[12]  # นิ้วกลาง

            # แปลงตำแหน่งเป็นพิกัดในภาพ
            h, w, c = img.shape
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            index_x, index_y = int(index.x * w), int(index.y * h)
            middle_x, middle_y = int(middle.x * w), int(middle.y * h)

            # คำนวณเวกเตอร์
            vector_thumb_index = np.array([index_x - thumb_x, index_y - thumb_y])
            vector_thumb_middle = np.array([middle_x - thumb_x, middle_y - thumb_y])

            # คำนวณมุมระหว่างเวกเตอร์
            angle = np.arctan2(vector_thumb_index[1], vector_thumb_index[0]) - np.arctan2(vector_thumb_middle[1], vector_thumb_middle[0])
            angle = np.degrees(angle)  # แปลงเป็นองศา

            # แสดงมุม
            angle_text = f"Angle: {int(angle)} degrees"
            cv2.putText(img, angle_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # ตรวจสอบเงื่อนไข
            if angle < 0:
                result_text = "rock"
            elif 0 < angle <= 20:
                result_text = "perper"
            else:  # angle > 20
                result_text = "scissors"

            # แสดงผลลัพธ์
            cv2.putText(img, result_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # วาดจุดและการเชื่อมต่อ
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
