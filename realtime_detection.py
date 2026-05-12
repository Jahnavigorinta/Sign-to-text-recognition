import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load model
model = pickle.load(open("best_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            landmarks = []

            for lm in hand.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            if len(landmarks) == 63:
                prediction = model.predict([landmarks])
                label = encoder.inverse_transform(prediction)[0]

                cv2.putText(frame, label, (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2)

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Sign Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()