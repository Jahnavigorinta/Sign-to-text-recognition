import cv2
import mediapipe as mp
import os

label = input("Enter sign label: ")

# create folders
image_path = "dataset/images/" + label
landmark_file = "dataset/landmarks/" + label + ".txt"

os.makedirs(image_path, exist_ok=True)
os.makedirs("dataset/landmarks", exist_ok=True)

file = open(landmark_file, "a")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            landmarks = []

            for lm in hand.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            key = cv2.waitKey(1)

            if key == ord('s'):

                # save landmarks
                file.write(",".join(map(str, landmarks))+"\n")

                # save image
                img_name = image_path + "/" + str(count) + ".jpg"
                cv2.imwrite(img_name, frame)

                count += 1

                print("Saved sample:", count)

    cv2.imshow("Collecting Dataset", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
file.close()