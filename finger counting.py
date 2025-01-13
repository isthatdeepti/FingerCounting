import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=4)
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

def count_fingers(hand_landmarks, palm_orientation,hand_handedness):

    tipIds = [4, 8, 12, 16, 20]  # Index finger tip IDs
    fingers = [False] * 5
    mcp_ids = [2, 5, 9, 13, 17]
    # Thumb
    if palm_orientation == "front":
        if hand_handedness.classification[0].label == "Left":
            fingers[0]=hand_landmarks.landmark[tipIds[0]].x > hand_landmarks.landmark[mcp_ids[1]].x

        else:
            fingers[0]=hand_landmarks.landmark[tipIds[0]].x < hand_landmarks.landmark[mcp_ids[1]].x
    else:  # Palm facing back
        if hand_handedness.classification[0].label == "Left":
            fingers[0]=hand_landmarks.landmark[tipIds[0]].x < hand_landmarks.landmark[mcp_ids[0]].x
        else:
            fingers[0] = hand_landmarks.landmark[tipIds[0]].x > hand_landmarks.landmark[mcp_ids[0]].x

    # Fingers
    for i in range(1, 5):
        if palm_orientation == "front":
            fingers[i]=hand_landmarks.landmark[tipIds[i]].y < hand_landmarks.landmark[tipIds[i] - 2].y

        else:  # Palm facing back
            fingers[i]=hand_landmarks.landmark[tipIds[i]].y < hand_landmarks.landmark[tipIds[i] - 2].y

    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    total_fingers = 0

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            orientation = get_palm_orientation(hand_landmarks, hand_handedness)
            fingers = count_fingers(hand_landmarks, orientation,hand_handedness)

            total_fingers += sum(fingers)

    cv2.putText(img, f"Total Fingers: {total_fingers}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 100, 0), 5)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()