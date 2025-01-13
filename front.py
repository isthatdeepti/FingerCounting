# This is used to find wheather th hand is facing front or back 
# after tht its logic can be applied iin finnger Counting
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def get_palm_orientation(hand_landmarks, hand_handedness):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # Calculate vectors
    wrist_to_index = np.array([index_finger_tip.x - wrist.x, index_finger_tip.y - wrist.y])
    wrist_to_middle = np.array([middle_finger_tip.x - wrist.x, middle_finger_tip.y - wrist.y])

    # Calculate cross product
    cross_product = np.cross(wrist_to_index, wrist_to_middle)

    # Adjust for left hand
    if hand_handedness.classification[0].label == "Left":
        cross_product = -cross_product

    # Determine palm orientation
    if cross_product > 0:
        return "front"
    else:
        return "back"

while True:
    success, img = cap.read()
    img=cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            orientation = get_palm_orientation(hand_landmarks, hand_handedness)

            cv2.putText(img, f"Palm Orientation: {orientation}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
