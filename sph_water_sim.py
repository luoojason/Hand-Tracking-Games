import cv2
import numpy as np
import math
import time
import random
import urllib.request
import os
import sys

# Try to import mediapipe
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not installed. Install with: pip install mediapipe")

# Model download with better error handling
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def download_model():
    """Download the hand landmarker model with error handling"""
    if os.path.exists(MODEL_PATH):
        return True

    print("Hand landmarker model not found. Attempting to download...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    try:
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"\nERROR: Could not download model: {e}")
        print("\nPLEASE MANUALLY DOWNLOAD THE MODEL:")
        print(f"1. Go to: {url}")
        print(f"2. Save the file as: {MODEL_PATH}")
        print("3. Re-run this script\n")
        return False


class WaterParticle:
    """
    Water particle for SPH (Smoothed Particle Hydrodynamics) simulation.
    Represents a small volume of water with fluid properties.
    """

    def __init__(self, position, mass=0.02):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.force = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.mass = mass

        # SPH properties
        self.density = 0.0
        self.pressure = 0.0
        self.neighbors = []  # List of nearby particles


class PhysicsWorld:
    """
    SPH (Smoothed Particle Hydrodynamics) water simulation:
    - Particle-based fluid dynamics
    - Pressure forces for incompressibility
    - Viscosity for realistic flow
    - Surface tension and cohesion
    - Collision with rotating cube walls
    """

    def __init__(self, num_particles=400, particle_radius=0.04, cube_size=2.0):
        self.gravity = np.array([0.0, -6.0, 0.0])  # Reduced gravity for calmer water
        self.cube_size = cube_size
        self.particle_radius = particle_radius
        self.particles = []
        self.max_velocity = 5.0  # Velocity limiter for stability

        # SPH parameters (tuned for stable, realistic water)
        self.smoothing_radius = particle_radius * 3.0  # Kernel support radius
        self.particle_mass = 0.02  # Mass per particle
        self.rest_density = 1000.0  # Water density (kg/m^3)
        self.gas_constant = 500.0  # Reduced for stability (was 2000)
        self.viscosity = 0.08  # Increased viscosity for smoother flow
        self.surface_tension = 0.0728  # Surface tension coefficient
        self.damping = 0.98  # More damping for stability (was 0.999)

        # Wall collision properties
        self.wall_damping = 0.6  # More energy loss for calmer behavior (was 0.5)
        self.boundary_epsilon = 0.001  # Slightly larger to prevent edge cases

        # Current rotation matrix for the cube
        self.rotation_matrix = np.eye(3)
        self.inv_rotation_matrix = np.eye(3)

        # Precompute kernel constants
        h = self.smoothing_radius
        self.poly6_constant = 315.0 / (64.0 * np.pi * h**9)
        self.spiky_constant = -45.0 / (np.pi * h**6)
        self.viscosity_constant = 45.0 / (np.pi * h**6)

        # Create water particles
        self._create_water_particles(num_particles)

    def _create_water_particles(self, num_particles):
        """Create water particles in a compact volume at the top"""
        particles_per_side = int(np.ceil(num_particles ** (1/3)))
        spacing = self.particle_radius * 2.1

        count = 0
        # Create a cube of water particles
        for i in range(particles_per_side):
            for j in range(particles_per_side):
                for k in range(particles_per_side):
                    if count >= num_particles:
                        break

                    x = (i - particles_per_side/2) * spacing
                    y = 0.3 + j * spacing  # Start above center
                    z = (k - particles_per_side/2) * spacing

                    particle = WaterParticle([x, y, z], mass=self.particle_mass)
                    self.particles.append(particle)
                    count += 1

                if count >= num_particles:
                    break
            if count >= num_particles:
                break

    def set_cube_rotation(self, pitch, yaw, roll):
        """Set the cube rotation from Euler angles (degrees)"""
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        roll_rad = np.radians(roll)

        # Create rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])

        Ry = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])

        Rz = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad), np.cos(roll_rad), 0],
            [0, 0, 1]
        ])

        self.rotation_matrix = Rz @ Ry @ Rx
        self.inv_rotation_matrix = self.rotation_matrix.T

    def update_cube_size(self, new_size):
        """Update cube size and constrain particles inside"""
        self.cube_size = new_size
        self._constrain_particles_to_cube()

    def _constrain_particles_to_cube(self):
        """Ensure all particles are inside the cube"""
        half = self.cube_size / 2 - self.particle_radius - self.boundary_epsilon

        for particle in self.particles:
            # Transform to local cube space
            local_pos = self.inv_rotation_matrix @ particle.position
            local_vel = self.inv_rotation_matrix @ particle.velocity

            for i in range(3):
                # Check boundaries
                if local_pos[i] < -half:
                    local_pos[i] = -half
                    local_vel[i] *= -self.wall_damping
                elif local_pos[i] > half:
                    local_pos[i] = half
                    local_vel[i] *= -self.wall_damping

            # Transform back to world space
            particle.position = self.rotation_matrix @ local_pos
            particle.velocity = self.rotation_matrix @ local_vel

    def step(self, dt):
        """
        Step the SPH fluid simulation.
        Uses standard SPH pipeline: compute density, compute forces, integrate, collisions.
        """
        # Clamp dt for stability
        dt = min(dt, 0.033)  # Cap at ~30fps for better performance

        # Single substep for performance (can increase to 2 for more accuracy)
        self._substep(dt)

    def _substep(self, dt):
        """Single SPH simulation substep"""
        # 1. Find neighbors for each particle (spatial hashing would be better for large simulations)
        self._compute_neighbors()

        # 2. Compute density and pressure for each particle
        self._compute_density_pressure()

        # 3. Compute SPH forces (pressure, viscosity, surface tension)
        self._compute_forces()

        # 4. Integrate (apply forces and move particles)
        for particle in self.particles:
            # Apply gravity
            particle.force += self.gravity * particle.mass

            # Integrate velocity: v = v + (F/m) * dt
            acceleration = particle.force / particle.mass
            particle.velocity += acceleration * dt
            particle.velocity *= self.damping

            # Clamp velocity to prevent instability
            speed = np.linalg.norm(particle.velocity)
            if speed > self.max_velocity:
                particle.velocity = (particle.velocity / speed) * self.max_velocity

            # Check for NaN/Inf and reset if found
            if not np.all(np.isfinite(particle.velocity)):
                particle.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)

            # Integrate position: x = x + v * dt
            particle.position += particle.velocity * dt

            # Reset forces for next iteration
            particle.force = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # 5. Handle boundary collisions
        self._constrain_particles_to_cube()

    def _compute_neighbors(self):
        """Find neighboring particles within smoothing radius (optimized brute force)"""
        h_squared = self.smoothing_radius ** 2

        for particle in self.particles:
            particle.neighbors = []

        for i, particle_i in enumerate(self.particles):
            for j in range(i + 1, len(self.particles)):  # Only check each pair once
                particle_j = self.particles[j]

                # Quick rejection using squared distance (avoids sqrt)
                diff = particle_j.position - particle_i.position
                distance_squared = np.dot(diff, diff)

                if distance_squared < h_squared:
                    distance = np.sqrt(distance_squared)
                    # Add to both particles' neighbor lists
                    particle_i.neighbors.append((particle_j, distance))
                    particle_j.neighbors.append((particle_i, distance))

    def _compute_density_pressure(self):
        """Compute density and pressure for each particle using SPH"""
        h = self.smoothing_radius

        for particle in self.particles:
            # Self-contribution
            density = self.particle_mass * self.poly6_constant * (h**2)**3

            # Neighbor contributions
            for neighbor, distance in particle.neighbors:
                if distance < h:
                    density += self.particle_mass * self.poly6_constant * (h**2 - distance**2)**3

            particle.density = max(density, self.rest_density)  # Prevent negative/zero density

            # Compute pressure using ideal gas state equation
            # P = k * (ρ - ρ_0) where k is gas constant
            particle.pressure = self.gas_constant * (particle.density - self.rest_density)

    def _compute_forces(self):
        """Compute SPH forces: pressure, viscosity, and surface tension"""
        h = self.smoothing_radius

        for particle in self.particles:
            pressure_force = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            viscosity_force = np.array([0.0, 0.0, 0.0], dtype=np.float64)

            for neighbor, distance in particle.neighbors:
                if distance < h and distance > 1e-6:
                    # Direction from particle to neighbor
                    direction = (neighbor.position - particle.position) / distance

                    # Pressure force (repulsion based on pressure gradient)
                    # Using Spiky kernel gradient
                    pressure_term = (particle.pressure + neighbor.pressure) / (2.0 * neighbor.density)
                    pressure_kernel = self.spiky_constant * (h - distance)**2
                    pressure_force += -self.particle_mass * pressure_term * pressure_kernel * direction

                    # Viscosity force (smoothing velocity differences)
                    # Using viscosity kernel Laplacian
                    viscosity_kernel = self.viscosity_constant * (h - distance)
                    velocity_diff = neighbor.velocity - particle.velocity
                    viscosity_force += self.viscosity * self.particle_mass * (velocity_diff / neighbor.density) * viscosity_kernel

            particle.force += pressure_force + viscosity_force

    def get_particle_positions(self):
        """Get current positions of all water particles"""
        return [particle.position.copy() for particle in self.particles]

    def cleanup(self):
        """Cleanup (no-op for this implementation)"""
        pass


class HandTracker:
    """Hand tracking using MediaPipe Tasks API"""

    def __init__(self, model_path):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is not installed. Install with: pip install mediapipe")

        # Create hand landmarker with lower thresholds for better stability
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,  # Lower for better detection
            min_hand_presence_confidence=0.3,   # Lower for better stability
            min_tracking_confidence=0.3         # Lower to maintain tracking
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

    def process(self, frame):
        """Process a frame and return hand landmarks"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        self.frame_timestamp_ms += 33  # ~30fps
        result = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)

        return result

    def close(self):
        """Close the detector"""
        self.detector.close()


def draw_hand_landmarks(frame, detection_result):
    """Draw hand landmarks with futuristic muted colors"""
    if not detection_result.hand_landmarks:
        return

    for hand_landmarks in detection_result.hand_landmarks:
        # Draw connections
        landmark_points = []
        h, w = frame.shape[:2]

        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmark_points.append((x, y))
            # Muted cyan dots
            cv2.circle(frame, (x, y), 5, (100, 80, 50), -1)

        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]

        # Thicker lines with muted purple/blue
        for start, end in connections:
            if start < len(landmark_points) and end < len(landmark_points):
                cv2.line(frame, landmark_points[start], landmark_points[end],
                         (110, 70, 90), 3, cv2.LINE_AA)


def get_hand_distance(landmarks1, landmarks2):
    """Calculate distance between two hands"""
    wrist1 = landmarks1[0]
    wrist2 = landmarks2[0]

    distance = math.sqrt(
        (wrist1.x - wrist2.x) ** 2 +
        (wrist1.y - wrist2.y) ** 2 +
        (wrist1.z - wrist2.z) ** 2
    )
    return distance


def get_hand_rotation(landmarks):
    """Calculate 3D rotation based on wrist rotation with extended pointer and thumb"""
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    pointer_tip = landmarks[8]
    pinky_mcp = landmarks[17]

    wrist_pos = np.array([wrist.x, wrist.y, wrist.z])
    thumb_pos = np.array([thumb_tip.x, thumb_tip.y, thumb_tip.z])
    pointer_pos = np.array([pointer_tip.x, pointer_tip.y, pointer_tip.z])
    pinky_pos = np.array([pinky_mcp.x, pinky_mcp.y, pinky_mcp.z])

    midpoint = (thumb_pos + pointer_pos) / 2
    forward = midpoint - wrist_pos
    forward = forward / (np.linalg.norm(forward) + 1e-6)

    side = thumb_pos - pinky_pos
    side = side / (np.linalg.norm(side) + 1e-6)

    up = np.cross(forward, side)
    up = up / (np.linalg.norm(up) + 1e-6)

    side = np.cross(up, forward)

    roll = math.atan2(side[1], side[0])
    pitch = math.asin(np.clip(-forward[1], -1, 1))
    yaw = math.atan2(forward[0], forward[2])

    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)


def get_rotation_matrix(pitch, yaw, roll):
    """Create a 3D rotation matrix from Euler angles"""
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])

    Ry = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])

    Rz = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def project_3d_to_2d(point_3d, width, height):
    """Simple perspective projection"""
    distance = 5.0
    fov = 800

    if point_3d[2] + distance != 0:
        x = int((point_3d[0] * fov) / (point_3d[2] + distance) + width / 2)
        y = int((-point_3d[1] * fov) / (point_3d[2] + distance) + height / 2)
    else:
        x = int(width / 2)
        y = int(height / 2)

    return (x, y)


def draw_futuristic_grid(frame):
    """Draw subtle futuristic grid overlay"""
    height, width = frame.shape[:2]
    grid_color = (30, 30, 35)  # Very dark, barely visible

    # Vertical lines
    for x in range(0, width, 60):
        cv2.line(frame, (x, 0), (x, height), grid_color, 1, cv2.LINE_AA)

    # Horizontal lines
    for y in range(0, height, 60):
        cv2.line(frame, (0, y), (width, y), grid_color, 1, cv2.LINE_AA)


def draw_center_reticle(frame):
    """Draw subtle center targeting reticle"""
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    reticle_color = (70, 60, 45)  # Muted dark cyan

    # Center dot
    cv2.circle(frame, (center_x, center_y), 3, reticle_color, 1, cv2.LINE_AA)

    # Crosshairs
    line_length = 20
    gap = 8
    cv2.line(frame, (center_x - gap - line_length, center_y),
             (center_x - gap, center_y), reticle_color, 2, cv2.LINE_AA)
    cv2.line(frame, (center_x + gap, center_y),
             (center_x + gap + line_length, center_y), reticle_color, 2, cv2.LINE_AA)
    cv2.line(frame, (center_x, center_y - gap - line_length),
             (center_x, center_y - gap), reticle_color, 2, cv2.LINE_AA)
    cv2.line(frame, (center_x, center_y + gap),
             (center_x, center_y + gap + line_length), reticle_color, 2, cv2.LINE_AA)


def add_futuristic_effects(frame):
    """Add scanlines and subtle vignette for cyberpunk aesthetic"""
    height, width = frame.shape[:2]

    # Subtle scanlines (every 4 pixels)
    for y in range(0, height, 4):
        frame[y:y+1, :] = (frame[y:y+1, :] * 0.95).astype(np.uint8)

    # Very subtle vignette (darker edges)
    overlay = frame.copy()
    center_x, center_y = width // 2, height // 2
    cv2.ellipse(overlay, (center_x, center_y), (width, height), 0, 0, 360, (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.95, overlay, 0.05, 0, frame)


def draw_cube_on_frame(frame, cube_center, cube_size, rotation_matrix):
    """Draw futuristic cube with thick muted lines"""
    height, width = frame.shape[:2]
    half = cube_size / 2

    vertices = [
        np.array([-half, -half, -half]),
        np.array([half, -half, -half]),
        np.array([half, half, -half]),
        np.array([-half, half, -half]),
        np.array([-half, -half, half]),
        np.array([half, -half, half]),
        np.array([half, half, half]),
        np.array([-half, half, half]),
    ]

    world_vertices = []
    for vertex in vertices:
        world_pos = cube_center + rotation_matrix @ vertex
        world_vertices.append(world_pos)

    points_2d = [project_3d_to_2d(v, width, height) for v in world_vertices]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    # Futuristic style: thick dark glow with muted cyan core
    # Outer glow - dark blue/gray
    for edge in edges:
        pt1, pt2 = points_2d[edge[0]], points_2d[edge[1]]
        cv2.line(frame, pt1, pt2, (80, 60, 40), 12, cv2.LINE_AA)

    # Core line - muted cyan
    for edge in edges:
        pt1, pt2 = points_2d[edge[0]], points_2d[edge[1]]
        cv2.line(frame, pt1, pt2, (120, 100, 60), 5, cv2.LINE_AA)


def draw_water_particle_on_frame(frame, particle_pos, particle_radius):
    """Draw futuristic water particle with muted blue tones"""
    height, width = frame.shape[:2]

    center_2d = project_3d_to_2d(particle_pos, width, height)

    distance = 5.0
    depth = particle_pos[2] + distance
    if depth > 0:
        apparent_radius = int((particle_radius * 800) / depth)
    else:
        apparent_radius = 5

    apparent_radius = max(apparent_radius, 4)  # Ensure minimum visibility

    # Futuristic muted water particles - dark blue/cyan
    # Outer glow - darker blue
    cv2.circle(frame, center_2d, int(apparent_radius * 1.8), (120, 80, 40), -1, cv2.LINE_AA)
    # Core - muted cyan
    cv2.circle(frame, center_2d, apparent_radius, (140, 100, 50), -1, cv2.LINE_AA)


def main():
    # Check dependencies
    if not MEDIAPIPE_AVAILABLE:
        print("\nERROR: MediaPipe is not installed!")
        print("Install it with: pip install mediapipe")
        return

    # Try to download model
    if not download_model():
        return

    # Initialize webcam
    print("Initializing camera...")
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Camera 1 not available, trying camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("\nERROR: Could not open any camera!")
            print("\nTroubleshooting:")
            print("1. Make sure your webcam is connected")
            print("2. Check if another application is using the camera")
            print("3. Try running with different camera indices (0, 1, 2...)")
            print("4. On Linux, check camera permissions")
            return

    # PERFORMANCE: Lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # Test reading a frame
    ret, test_frame = cap.read()
    if not ret:
        print("ERROR: Camera opened but could not read frame.")
        cap.release()
        return
    print(f"Camera opened successfully. Frame size: {test_frame.shape}")

    # Initialize hand tracking with new Tasks API
    try:
        hand_tracker = HandTracker(MODEL_PATH)
        print("Hand tracker initialized successfully")
    except Exception as e:
        print(f"ERROR: Could not initialize hand tracker: {e}")
        cap.release()
        return

    # Initialize water physics world
    # PERFORMANCE: Reduce num_particles if too slow (try 150, 200, or 300)
    physics = PhysicsWorld(num_particles=200, particle_radius=0.045, cube_size=2.0)

    # Cube parameters
    cube_center = np.array([0.0, 0.0, 0.0])
    cube_size = 2.0
    target_cube_size = 2.0
    cube_rotation = [0.0, 0.0, 0.0]
    target_rotation = [0.0, 0.0, 0.0]

    # Timing
    last_time = time.time()

    # PERFORMANCE: Physics frame skip (1 = no skip, 2 = update every other frame)
    physics_frame_skip = 1  # Increase to 2 if still too slow
    frame_count = 0

    print("\n" + "=" * 60)
    print("SPH WATER SIMULATION - INITIALIZED")
    print("=" * 60)
    print("Realistic water physics using Smoothed Particle Hydrodynamics")
    print(f"Simulating {len(physics.particles)} water particles")
    print("\nPerformance optimizations enabled:")
    print(f"  - Particle count: {len(physics.particles)}")
    print(f"  - Physics updates: Every {physics_frame_skip} frame(s)")
    print("\nControls:")
    print("  - ONE HAND:")
    print("    * Twist hand left/right → Spin cube horizontally")
    print("    * Tilt hand up/down → Rotate cube vertically")
    print("  - TWO HANDS: Spread apart/together → Resize cube")
    print("  - ESC or Q: Exit")
    print("=" * 60 + "\n")

    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            dt = min(dt, 0.05)
            frame_count += 1

            ret, frame = cap.read()
            if not ret:
                print("WARNING: Could not read frame")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            # Add futuristic grid background
            draw_futuristic_grid(frame)

            try:
                results = hand_tracker.process(frame)

                if results.hand_landmarks and len(results.hand_landmarks) > 0:
                    # Two hands detected - control cube size
                    if len(results.hand_landmarks) == 2:
                        distance = get_hand_distance(
                            results.hand_landmarks[0],
                            results.hand_landmarks[1]
                        )
                        target_cube_size = 0.5 + distance * 8
                        target_cube_size = max(0.5, min(target_cube_size, 3.0))
                    # Only one hand - just control rotation, keep current size
                    else:
                        target_cube_size = cube_size  # Keep current size

                    # Any hands - control rotation with first hand
                    # Intuitive mapping: hand twist → horizontal spin, hand tilt → vertical tilt
                    if len(results.hand_landmarks) >= 1:
                        hand_pitch, hand_yaw, hand_roll = get_hand_rotation(results.hand_landmarks[0])

                        # Map hand movements to cube rotation intuitively:
                        # - Hand tilt up/down (pitch) → Cube tilt up/down (pitch)
                        # - Hand twist left/right (roll) → Cube spin horizontally (yaw)
                        # - Keep roll minimal for stability
                        target_rotation = [
                            hand_pitch,      # Cube pitch: hand tilt controls vertical rotation
                            hand_roll * 1.5, # Cube yaw: hand twist controls horizontal spin (amplified)
                            0                # Cube roll: keep flat for simplicity
                        ]

                    draw_hand_landmarks(frame, results)
                else:
                    # No hands detected - maintain current size and rotation
                    target_cube_size = cube_size

            except Exception as e:
                print(f"Hand tracking error: {e}")
                # On error, maintain current values
                target_cube_size = cube_size

            for i in range(3):
                cube_rotation[i] += (target_rotation[i] - cube_rotation[i]) * 0.15

            old_cube_size = cube_size
            cube_size += (target_cube_size - cube_size) * 0.1

            if abs(cube_size - old_cube_size) > 0.01:
                physics.update_cube_size(cube_size)

            # Update physics only every N frames for performance
            if frame_count % physics_frame_skip == 0:
                physics.set_cube_rotation(cube_rotation[0], cube_rotation[1], cube_rotation[2])
                physics.step(dt * physics_frame_skip)  # Compensate for skipped frames

            particle_positions = physics.get_particle_positions()

            rotation_matrix = get_rotation_matrix(
                cube_rotation[0],
                cube_rotation[1],
                cube_rotation[2]
            )

            draw_cube_on_frame(frame, cube_center, cube_size, rotation_matrix)

            for particle_pos in particle_positions:
                draw_water_particle_on_frame(frame, particle_pos, 0.04)

            # Draw center targeting reticle
            draw_center_reticle(frame)

            # Futuristic dark overlay for UI
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (600, 215), (15, 15, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            fps = 1.0 / dt if dt > 0 else 0

            # Determine hand tracking status
            try:
                num_hands = len(results.hand_landmarks) if results and results.hand_landmarks else 0
            except:
                num_hands = 0

            # Muted cyan/gray text colors for futuristic look
            # Title with glow effect
            cv2.putText(frame, "SPH WATER SIMULATION", (15, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (60, 50, 40), 4, cv2.LINE_AA)
            cv2.putText(frame, "SPH WATER SIMULATION", (15, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (100, 90, 70), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f} | PARTICLES: {len(particle_positions)}", (15, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (80, 100, 80), 1, cv2.LINE_AA)

            # Show hand tracking status
            hand_status = f"HANDS: {num_hands}/2" if num_hands > 0 else "HANDS: NONE"
            hand_color = (80, 120, 80) if num_hands > 0 else (80, 80, 80)
            cv2.putText(frame, hand_status, (15, 90),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, hand_color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"CUBE SIZE: {cube_size:.2f}", (15, 115),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (90, 80, 60), 1, cv2.LINE_AA)

            cv2.putText(frame, "TWO HANDS: Change size", (15, 145),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(frame, "ONE HAND: Twist (horizontal) | Tilt (vertical)", (15, 170),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(frame, "ESC: Exit", (15, 193),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (80, 90, 80), 1, cv2.LINE_AA)

            # Thick muted corner brackets for futuristic frame
            corner_color = (80, 70, 50)  # Muted dark cyan
            cv2.line(frame, (0, 0), (70, 0), corner_color, 4)
            cv2.line(frame, (0, 0), (0, 70), corner_color, 4)
            cv2.line(frame, (width - 70, 0), (width, 0), corner_color, 4)
            cv2.line(frame, (width, 0), (width, 70), corner_color, 4)
            cv2.line(frame, (0, height - 70), (0, height), corner_color, 4)
            cv2.line(frame, (0, height), (70, height), corner_color, 4)
            cv2.line(frame, (width - 70, height), (width, height), corner_color, 4)
            cv2.line(frame, (width, height - 70), (width, height), corner_color, 4)

            # Add final futuristic effects (scanlines, vignette)
            add_futuristic_effects(frame)

            cv2.imshow('SPH Water Simulation - Hand Control', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:  # ESC or Q
                print("\nExiting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        physics.cleanup()
        hand_tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()