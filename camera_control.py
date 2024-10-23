import cv2
from datetime import datetime
import os

save_path = '/Users/OPA_images'
if not os.path.exists(save_path):
    os.makedirs(save_path)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print('fail to turn on the camera')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print('fail to catch the frame')
        break

    cv2.imshow('camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        filename = f'image_{current_time}.png'
        file_full_path = os.path.join(save_path, filename)
        cv2.imwrite(file_full_path, frame)
        print(f'image has been saved as "{filename}"')
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



#print('hello world!')