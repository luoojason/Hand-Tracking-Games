import cv2
import mediapipe as mp
import math

# --- Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera!")
    exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
# max_num_hands=1 for clarity in single sign reading
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)


# --- Helper functions ---
def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def is_finger_folded(hand_landmarks, tip_id, pip_id, wrist_id=0):
    wrist = hand_landmarks.landmark[wrist_id]
    tip = hand_landmarks.landmark[tip_id]
    pip = hand_landmarks.landmark[pip_id]
    # Simple check: distance from tip to wrist < distance from pip to wrist
    return distance(tip, wrist) < distance(pip, wrist)


def get_folded_status(hand_landmarks):
    # Returns [Thumb, Index, Middle, Ring, Pinky] booleans (True if folded)
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]

    folded = []

    # Thumb: Checking x distance compared to MCP
    # Right hand: Thumb is to the left of Index (smaller x)
    # This logic depends on handedness.
    # Let's use simple distance check first for general robustnes, and refine if needed.
    # "Folded" thumb usually means tip is close to Pinky MCP or Ring MCP?

    wrist = hand_landmarks.landmark[0]

    # Fingers 1-4 (Index to Pinky)
    for i in range(1, 5):
        folded.append(is_finger_folded(hand_landmarks, tips_ids[i], pip_ids[i]))

    # Thumb processing
    # Heuristic: Thumb tip distance to Index MCP vs Thumb IP distance to Index MCP?
    # Or just standard fold check relative to wrist? Standard is okay for "Fist",
    # but for "A" or "S", thumb placement matters.
    # Let's stick to standard fold check for now, + special logic in gesture detector.

    # Check if thumb tip is "inside" the hand (close to palm center / pinky mcp)
    thumb_tip = hand_landmarks.landmark[4]
    pinky_mcp = hand_landmarks.landmark[17]
    index_mcp = hand_landmarks.landmark[5]

    # If thumb tip is closer to Pinky MCP than Index MCP is to Pinky MCP?
    # Or just check if thumb tip x is between index mcp x and pinky mcp x?
    # Simple boolean:
    if distance(thumb_tip, pinky_mcp) < distance(index_mcp, pinky_mcp):
        folded.insert(0, True)  # Thumb folded
    else:
        folded.insert(0, False)  # Thumb open

    return folded


# --- Recognition Logic ---
def detect_sign(hand_landmarks):
    # Landmarks map
    # 0: Wrist
    # 4: Thumb Tip
    # 8: Index Tip
    # 12: Middle Tip
    # 16: Ring Tip
    # 20: Pinky Tip

    folded = get_folded_status(hand_landmarks)
    # folded: [Thumb, Index, Middle, Ring, Pinky]

    thumb_folded, index_folded, middle_folded, ring_folded, pinky_folded = folded

    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]

    # --- ASL Alphabet (Static) ---

    # A: Fist, Thumb vertical against side of hand (not tucked)
    # Distinct: All 4 fingers folded. Thumb is essentially "Not folded" typically?
    # Or aligned with hand.
    # Let's check: 4 fingers folded. Thumb up?
    if index_folded and middle_folded and ring_folded and pinky_folded:
        # Fist-like family: A, E, S, M, N, T
        # A: Thumb straight up/side.
        # E: Thumb curled under tips.
        # S: Thumb across fingers.

        # Check Thumb closeness to index PIP (6)
        index_pip = hand_landmarks.landmark[6]

        # If thumb tip is high (low y) -> A
        # If thumb tip is low (high y) -> E / S

        # A vs S/E
        if thumb_tip.y < index_pip.y:
            return "A"
        else:
            # E vs S? Hard.
            return "E / S / Fist"
    # B: Open Palm, Thumb tucked (or across palm)
    if not index_folded and not middle_folded and not ring_folded and not pinky_folded:
        if thumb_folded:
            return "B"
        else:
            return "Open Palm / 5"
    # C: Curved fingers.
    # Actually C looks like "open" but with curved tips.
    # Distance between Thumb Tip and Index Tip is large but shape is C.
    # Hard to detect vs "Open". Skip for now or use special curve check.
    # D: Index Up, others folded, Thumb touching Middle/Ring?
    if not index_folded and middle_folded and ring_folded and pinky_folded:
        # Check thumb? Usually thumb touches middle tip/pip.
        if distance(thumb_tip, middle_tip) < 0.1:  # Normalized dist
            return "D"
        return "Pointing / 1"
    # F: OK sign. Index + Thumb touch, others open.
    if distance(thumb_tip, index_tip) < 0.05:
        # Check others
        if not middle_folded and not ring_folded and not pinky_folded:
            return "F / OK"
    # I: Pinky up, others folded.
    if index_folded and middle_folded and ring_folded and not pinky_folded:
        # Check thumb. If thumb is across -> I. If thumb out -> Y.
        if thumb_folded:
            return "I"
        else:
            return "Y / Call Me"

    # K: Index + Middle up, Thumb in between? (Hard)
    # L: Thumb + Index Open, others folded.
    if not thumb_folded and not index_folded and middle_folded and ring_folded and pinky_folded:
        return "L"

    # O: All tips touching thumb? Looks like Fist but tips touch.
    # Fingers folded, but TIPS are high?

    # R: Index + Middle Crossed. (Hard to catch cross).
    # U: Index + Middle up together.
    # V: Index + Middle up V-shape (Peace).
    if not index_folded and not middle_folded and ring_folded and pinky_folded:
        # Check distance between tips
        dist_tips = distance(index_tip, middle_tip)
        if dist_tips < 0.04:
            return "U"
        else:
            return "V / Peace"

    # W: Index+Middle+Ring up.
    if not index_folded and not middle_folded and not ring_folded and pinky_folded:
        return "W / 3"

    return ""


# --- Main/Loop ---
while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)  # Mirror
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw Labels
            sign = detect_sign(hand_landmarks)

            bbox_c = hand_landmarks.landmark[0]  # Wrist
            h, w, c = frame.shape
            cx, cy = int(bbox_c.x * w), int(bbox_c.y * h)

            if sign:
                cv2.putText(frame, sign, (cx - 50, cy - 50), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "...", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    cv2.imshow("Sign Language Reader", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
