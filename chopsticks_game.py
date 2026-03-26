import cv2
import mediapipe as mp
import time
import math
import os
import torch
import torch.nn as nn
import numpy as np


# ==================== NEURAL NETWORK ====================
# Architecture matches the trained PPOActorCritic from train_chopsticks.py
class PPOActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.policy = nn.Linear(64, 5)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        z = self.shared(x)
        return self.policy(z), self.value(z).squeeze(-1)


def load_bot_model():
    """Load the trained neural network model from the script's directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, 'chopsticks_true_ppo.pt')
    model = PPOActorCritic()
    try:
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Neural network model loaded from {path}")
        return model
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("  Bot will use fallback heuristics instead.")
        return None


def state_to_tensor(bot_hands, player_hands):
    """Convert game state to the format the trained model expects:
    [bot_L/5, bot_R/5, player_L/5, player_R/5, 1.0 (turn flag)]"""
    return torch.tensor([
        bot_hands[0] / 5.0, bot_hands[1] / 5.0,
        player_hands[0] / 5.0, player_hands[1] / 5.0,
        1.0
    ], dtype=torch.float32)


def masked_softmax(logits, legal_actions):
    """Softmax over only legal actions (mask illegal ones to -inf)."""
    mask = torch.full_like(logits, -1e9)
    for a in legal_actions:
        mask[a] = 0
    return torch.softmax(logits + mask, -1)


def get_legal_actions(src, dst):
    """Get legal action indices. src=attacker hands, dst=defender hands.
    Actions 0-3 are attacks, action 4 is split."""
    acts = []
    if src[0] > 0 and dst[0] > 0: acts.append(0)  # L->L
    if src[0] > 0 and dst[1] > 0: acts.append(1)  # L->R
    if src[1] > 0 and dst[0] > 0: acts.append(2)  # R->L
    if src[1] > 0 and dst[1] > 0: acts.append(3)  # R->R
    if src.count(0) == 1 and sum(src) % 2 == 0:
        acts.append(4)  # split
    return acts


# ==================== GAME LOGIC ====================
MAX_FINGERS = 5  # mod 5: values 0-4, 0 = dead


def apply_attack(target_val, add_val):
    """Chopsticks attack: (target + add) mod 5. 0 = dead hand."""
    return (target_val + add_val) % MAX_FINGERS


def execute_action(action_idx, bot_hands, player_hands):
    """Execute a training-env action. Returns (bot_hands, player_hands, description)."""
    if action_idx < 4:
        src_idx = action_idx // 2
        dst_idx = action_idx % 2
        player_hands[dst_idx] = apply_attack(player_hands[dst_idx], bot_hands[src_idx])
        sides = ["L", "R"]
        desc = f"Bot {sides[src_idx]} -> Player {sides[dst_idx]}"
    else:
        total = sum(bot_hands)
        half = total // 2
        bot_hands[0], bot_hands[1] = half, half
        desc = f"Bot SPLIT: {half} | {half}"
    return bot_hands, player_hands, desc


# ==================== HEURISTIC FALLBACK ====================
def evaluate_state(bot_h, player_h):
    """Score how good the state is for the bot (higher = better)."""
    score = 0
    bot_alive = sum(1 for h in bot_h if h > 0)
    player_alive = sum(1 for h in player_h if h > 0)
    score += bot_alive * 100 - player_alive * 100
    if bot_alive == 2 and bot_h[0] != bot_h[1]:
        score += 50
    for h in bot_h:
        if h == 4: score -= 30
    for h in player_h:
        if h == 4: score += 40
    score -= sum(bot_h) * 5 + sum(player_h) * (-5)
    return score


def bot_play_heuristic(bot_hands, player_hands):
    """Fallback heuristic bot with lookahead. Returns (action_type, details)."""
    best_move = None
    best_score = float('-inf')

    # Evaluate attacks
    for bot_idx in range(2):
        if bot_hands[bot_idx] == 0:
            continue
        for player_idx in range(2):
            if player_hands[player_idx] == 0:
                continue
            sim_p = player_hands.copy()
            sim_p[player_idx] = apply_attack(sim_p[player_idx], bot_hands[bot_idx])
            score = evaluate_state(bot_hands, sim_p)
            if sim_p[player_idx] == 0: score += 150
            if sim_p[0] == 0 and sim_p[1] == 0: score += 1000
            if score > best_score:
                best_score = score
                best_move = ("attack", bot_idx, player_idx)

    # Evaluate splits
    total = sum(bot_hands)
    if total > 0:
        for new_l in range(0, min(5, total + 1)):
            new_r = total - new_l
            if new_r > 4 or (new_l, new_r) == tuple(bot_hands):
                continue
            sim_b = [new_l, new_r]
            score = evaluate_state(sim_b, player_hands)
            if 0 in bot_hands and 0 not in sim_b: score += 100
            if new_l > 0 and new_r > 0 and new_l != new_r: score += 30
            if score > best_score:
                best_score = score
                best_move = ("split", new_l, new_r)

    return best_move


# ==================== HAND COUNTING ====================
def count_fingers(hand_landmarks):
    """Count extended fingers from MediaPipe hand landmarks."""
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]

    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    wrist = hand_landmarks.landmark[0]
    count = 0

    # Fingers 2-5 (index to pinky)
    for i in range(1, 5):
        tip = hand_landmarks.landmark[tips_ids[i]]
        pip = hand_landmarks.landmark[pip_ids[i]]
        if dist(tip, wrist) > dist(pip, wrist):
            count += 1

    # Thumb
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    index_mcp = hand_landmarks.landmark[5]
    if dist(thumb_tip, index_mcp) > dist(thumb_ip, index_mcp):
        count += 1

    return count


# ==================== MAIN ====================
def main():
    # --- Initialize camera (same pattern as cube_hand.py) ---
    print("Initializing camera...")
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Camera 1 not available, trying camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open any camera!")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Test reading a frame
    ret, test_frame = cap.read()
    if not ret:
        print("ERROR: Camera opened but could not read frame.")
        cap.release()
        return
    print(f"Camera ready. Frame size: {test_frame.shape}")

    # --- MediaPipe hands ---
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

    # --- Load neural network ---
    bot_model = load_bot_model()

    # --- Game state ---
    player_hands = [1, 1]
    bot_hands = [1, 1]
    turn = "Player"
    winner = None
    message = "Your Turn! Hover over bot's hand to attack."
    bot_message = ""

    # --- UI constants ---
    BOT_LEFT_BOX = (50, 50, 200, 200)
    BOT_RIGHT_BOX = (440, 50, 590, 200)

    # Selection timer (attack)
    selection_start_time = 0
    selected_target = None
    SELECTION_THRESHOLD = 1.0

    # Swap timer (split)
    swap_start_time = 0
    SWAP_THRESHOLD = 0.5
    pending_swap = None

    # Bot thinking delay
    bot_think_timer = 0
    BOT_THINK_DELAY = 1.5

    def check_winner():
        nonlocal winner
        if player_hands[0] == 0 and player_hands[1] == 0:
            winner = "Bot Wins!"
        elif bot_hands[0] == 0 and bot_hands[1] == 0:
            winner = "You Win!"

    def do_bot_turn():
        nonlocal bot_hands, player_hands, turn, message, bot_message

        # Try neural network
        if bot_model is not None:
            legal = get_legal_actions(bot_hands, player_hands)
            if legal:
                state = state_to_tensor(bot_hands, player_hands)
                with torch.no_grad():
                    logits, _ = bot_model(state)
                    probs = masked_softmax(logits, legal)

                sorted_actions = torch.argsort(probs, descending=True)
                for action_idx in sorted_actions:
                    action_idx = action_idx.item()
                    if action_idx in legal:
                        bot_hands, player_hands, desc = execute_action(
                            action_idx, bot_hands, player_hands
                        )
                        bot_message = f"NN: {desc}"
                        turn = "Player"
                        message = "Your Turn!"
                        return

        # Fallback to heuristic
        move = bot_play_heuristic(bot_hands, player_hands)
        if move:
            if move[0] == "attack":
                _, bot_idx, player_idx = move
                player_hands[player_idx] = apply_attack(player_hands[player_idx], bot_hands[bot_idx])
                sides = ["L", "R"]
                bot_message = f"Heuristic: {sides[bot_idx]} -> {sides[player_idx]}"
            elif move[0] == "split":
                _, new_l, new_r = move
                bot_hands[0], bot_hands[1] = new_l, new_r
                bot_message = f"Heuristic: SPLIT {new_l} | {new_r}"

        turn = "Player"
        message = "Your Turn!"

    def is_point_in_box(point, box):
        x, y = point
        return box[0] < x < box[2] and box[1] < y < box[3]

    # --- Main loop ---
    print("\nStarting Chopsticks Hand Tracking Game...")
    print("Controls:")
    print("  - Hover finger over bot's hand boxes to attack (hold 1 sec)")
    print("  - Show different finger counts to split/swap")
    print("  - Press 'q' to quit, 'r' to restart\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue  # Skip bad frames instead of crashing

            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(frame_rgb)
            check_winner()

            # Draw bot hand boxes
            color_l = (0, 0, 255) if bot_hands[0] > 0 else (100, 100, 100)
            cv2.rectangle(frame, (BOT_LEFT_BOX[0], BOT_LEFT_BOX[1]),
                          (BOT_LEFT_BOX[2], BOT_LEFT_BOX[3]), color_l, 2)
            cv2.putText(frame, f"Bot L: {bot_hands[0]}",
                        (BOT_LEFT_BOX[0] + 10, BOT_LEFT_BOX[1] + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_l, 2)

            color_r = (0, 0, 255) if bot_hands[1] > 0 else (100, 100, 100)
            cv2.rectangle(frame, (BOT_RIGHT_BOX[0], BOT_RIGHT_BOX[1]),
                          (BOT_RIGHT_BOX[2], BOT_RIGHT_BOX[3]), color_r, 2)
            cv2.putText(frame, f"Bot R: {bot_hands[1]}",
                        (BOT_RIGHT_BOX[0] + 10, BOT_RIGHT_BOX[1] + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_r, 2)

            # Player stats
            cv2.putText(frame, f"You L: {player_hands[0]}   You R: {player_hands[1]}",
                        (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- Hand detection & interaction ---
            hovering_target = None
            player_attacking_hand = None
            visible_hands = {0: 0, 1: 0}

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    label = results.multi_handedness[idx].classification[0].label
                    index_tip = hand_landmarks.landmark[8]
                    ix = int(index_tip.x * frame_w)
                    iy = int(index_tip.y * frame_h)

                    current_hand_idx = 0 if label == "Left" else 1
                    f_count = count_fingers(hand_landmarks)
                    visible_hands[current_hand_idx] = f_count

                    if player_hands[current_hand_idx] == 0:
                        continue

                    cv2.circle(frame, (ix, iy), 10, (255, 255, 0), -1)

                    if turn == "Player" and not winner:
                        if is_point_in_box((ix, iy), BOT_LEFT_BOX) and bot_hands[0] > 0:
                            hovering_target = "L"
                            player_attacking_hand = current_hand_idx
                        elif is_point_in_box((ix, iy), BOT_RIGHT_BOX) and bot_hands[1] > 0:
                            hovering_target = "R"
                            player_attacking_hand = current_hand_idx

            # --- Swap / Split logic ---
            should_swap = False
            new_L, new_R = 0, 0

            if turn == "Player" and not winner and hovering_target is None:
                total_game = sum(player_hands)
                visible_total = visible_hands[0] + visible_hands[1]

                if (visible_total == total_game and
                        visible_hands[0] != player_hands[0] and
                        visible_hands[0] <= 5 and visible_hands[1] <= 5):
                    should_swap = True
                    new_L, new_R = visible_hands[0], visible_hands[1]
                    cv2.putText(frame, f"Swap? {new_L} | {new_R}", (250, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if should_swap:
                if pending_swap == (new_L, new_R):
                    elapsed = time.time() - swap_start_time
                    cv2.line(frame, (250, 320),
                             (250 + int(200 * (elapsed / SWAP_THRESHOLD)), 320),
                             (0, 255, 255), 4)
                    if elapsed >= SWAP_THRESHOLD:
                        player_hands = [new_L, new_R]
                        turn = "Bot"
                        message = "Bot's Turn..."
                        bot_message = "Player Swapped!"
                        pending_swap = None
                        bot_think_timer = time.time()
                else:
                    pending_swap = (new_L, new_R)
                    swap_start_time = time.time()
            else:
                pending_swap = None
                swap_start_time = 0

            # --- Attack selection timer ---
            if hovering_target and turn == "Player" and not should_swap:
                if selected_target == hovering_target:
                    elapsed = time.time() - selection_start_time
                    progress = min(elapsed / SELECTION_THRESHOLD, 1.0)
                    box = BOT_LEFT_BOX if hovering_target == "L" else BOT_RIGHT_BOX
                    cx, cy = box[0] + 75, box[1] + 75
                    cv2.ellipse(frame, (cx, cy), (40, 40), 0, 0,
                                360 * progress, (0, 255, 255), 4)

                    if elapsed >= SELECTION_THRESHOLD:
                        target_idx = 0 if hovering_target == "L" else 1
                        bot_hands[target_idx] = apply_attack(
                            bot_hands[target_idx], player_hands[player_attacking_hand]
                        )
                        bot_message = ""
                        message = "Bot's Turn..."
                        turn = "Bot"
                        selected_target = None
                        bot_think_timer = time.time()
                else:
                    selected_target = hovering_target
                    selection_start_time = time.time()
            else:
                selected_target = None
                selection_start_time = 0

            # --- Display messages ---
            if winner:
                cv2.putText(frame, winner, (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                cv2.putText(frame, "Press 'r' to Restart", (180, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(frame, message, (10, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, bot_message, (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow("Chopsticks Hand Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if winner and key == ord('r'):
                player_hands = [1, 1]
                bot_hands = [1, 1]
                turn = "Player"
                winner = None
                message = "Your Turn!"
                bot_message = ""

            # --- Bot's turn (with thinking delay) ---
            if turn == "Bot" and not winner:
                if time.time() - bot_think_timer >= BOT_THINK_DELAY:
                    do_bot_turn()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
