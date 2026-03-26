# Hand Tracking Game Suite

A collection of computer vision games and simulations powered by MediaPipe hand tracking, OpenCV, and PyTorch.

## Games

| File | Description |
|------|-------------|
| `chopsticks_game.py` | Chopsticks game, played against a PPO-trained neural network AI. Attack by hovering your finger over the bot's hands, split by showing different finger counts. |
| `sph_water_sim.py` | Balls inside a Cube. Tilt and twist one hand to rotate the cube, use two hands to resize it. |
| `asl_recognition.py` | American Sign Language letter recognition. Detects ASL signs (A, B, D, E, F, I, L, U, V, W) in real time. |
| `rock_paper_scissors.py` | Play Rock-Paper-Scissors against the computer using hand gestures. Press 's' to start a countdown, then show your move. |

## Setup

### macOS

1. **Install Python 3.9+** (if not already installed):
   ```bash
   brew install python
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Allow camera access:** When you first run a game, macOS will prompt you to allow camera access for Terminal/your IDE. Click "Allow".

4. **Camera index:** The chopsticks game and water sim default to camera index `1` (external webcam). If you only have a built-in webcam, they will automatically fall back to index `0`. No changes needed.

### Windows

1. **Install Python 3.9+** from [python.org](https://www.python.org/downloads/). Make sure to check "Add Python to PATH" during installation.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Camera index:** The games default to camera index `1`, then fall back to `0`. If your webcam isn't detected, open the game file and change `cv2.VideoCapture(1)` to `cv2.VideoCapture(0)` at the top of the main function.

4. **PyTorch on Windows:** If `pip install torch` fails, install the CPU-only version:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

### Linux

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Camera permissions:** Make sure your user has access to `/dev/video0`:
   ```bash
   sudo usermod -aG video $USER
   ```
   Log out and back in for the change to take effect.

## Running

```bash
python chopsticks_game.py       # Chopsticks vs AI
python sph_water_sim.py         # Water simulation
python asl_recognition.py       # ASL recognition
python rock_paper_scissors.py   # Rock-Paper-Scissors
```

### Controls

All games use `q` to quit. 

**Chopsticks:** Hover your index finger over the bot's hand boxes (hold 1 sec to attack). Show different finger counts on each hand to split.

**Water Sim:** One hand twists/tilts the cube. Two hands spread apart or together to resize.

**ASL Recognition:** Hold up an ASL letter sign in front of the camera.

**Rock-Paper-Scissors:** Press `s` to start, show your gesture when the countdown ends.

## Requirements

- Python 3.9+
- Webcam
- opencv-python
- mediapipe
- torch (for Chopsticks AI only)
- numpy


