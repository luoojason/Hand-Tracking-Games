import cv2
import mediapipe as mp
import math
import time
import random

# --- Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera!")
    exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# --- Game State ---
STATE_IDLE = "IDLE"
STATE_COUNTDOWN = "COUNTDOWN"
STATE_RESULT = "RESULT"
current_state = STATE_IDLE
start_time = 0
countdown_duration = 3  # seconds
result_display_duration = 3  # seconds
result_start_time = 0
player_move = None
computer_move = None
winner = None


# --- Helper functions ---
def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def is_finger_folded(hand_landmarks, tip_id, pip_id, wrist_id=0):
    wrist = hand_landmarks.landmark[wrist_id]
    tip = hand_landmarks.landmark[tip_id]
    pip = hand_landmarks.landmark[pip_id]
    return distance(tip, wrist) < distance(pip, wrist)


# --- Gesture detection ---
def detect_gesture(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    pip_ids = [2, 6, 10, 14, 18]
    folded = [is_finger_folded(hand_landmarks, tip, pip) for tip, pip in zip(tips_ids, pip_ids)]
    thumb_folded, index_folded, middle_folded, ring_folded, pinky_folded = folded
    # Rock: All fingers folded (except maybe thumb depending on how people make a fist, but typically thumb is tucked or folded)
    # Actually, let's be lenient. Rock is mainly closed fist.
    if all(folded[1:]):  # Index, Middle, Ring, Pinky folded
        return "Rock"

    # Paper: All fingers open
    if not any(folded):
        return "Paper"
    # Scissors: Index and Middle open, others folded
    if not index_folded and not middle_folded and folded[3] and folded[4]:
        return "Scissors"
    return "Unknown"


def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])


def determine_winner(player, computer):
    if player == computer:
        return "It's a Tie!"
    elif (player == "Rock" and computer == "Scissors") or \
            (player == "Paper" and computer == "Rock") or \
            (player == "Scissors" and computer == "Paper"):
        return "You Win!"
    else:
        return "Computer Wins!"


# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    detected_gesture = "Unknown"
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_gesture = detect_gesture(hand_landmarks)

    # --- Game Logic ---
    current_time = time.time()

    if current_state == STATE_IDLE:
        cv2.putText(frame, "Press 's' to Start", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Gesture: {detected_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            current_state = STATE_COUNTDOWN
            start_time = current_time
    elif current_state == STATE_COUNTDOWN:
        elapsed_time = current_time - start_time
        time_left = math.ceil(countdown_duration - elapsed_time)

        if time_left > 0:
            cv2.putText(frame, str(time_left), (300, 240), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)
            cv2.putText(frame, f"Gesture: {detected_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            player_move = detected_gesture
            if player_move == "Unknown":
                player_move = "No Move"  # Or handle as invalid

            computer_move = get_computer_choice()
            winner = determine_winner(player_move, computer_move)

            current_state = STATE_RESULT
            result_start_time = current_time
    elif current_state == STATE_RESULT:
        elapsed_time = current_time - result_start_time

        cv2.putText(frame, f"You: {player_move}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"PC: {computer_move}", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        color = (0, 255, 0)
        if "Computer" in winner:
            color = (0, 0, 255)
        elif "Tie" in winner:
            color = (255, 255, 0)

        cv2.putText(frame, winner, (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

        if elapsed_time > result_display_duration:
            current_state = STATE_IDLE
    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
