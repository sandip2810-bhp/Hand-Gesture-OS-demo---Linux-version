import cv2
import mediapipe as mp
import math
import time
import subprocess

# 🔊 NEW: audio imports for clap detection
import sounddevice as sd
import numpy as np

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# For fingers (thumb, index, middle, ring, pinky)
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

GESTURE_HOLD_TIME = 1.5       # seconds to hold gesture before mode/command
REPEAT_INTERVAL = 0.15        # continuous repeat when eyes held closed

# 🔊 ===== CLAP SETTINGS (from clap_alt_tab.py) =====
SAMPLE_RATE = 16000       # mic sample rate
BLOCK_DURATION = 0.1      # seconds per audio block
CLAP_THRESHOLD = 0.35     # loudness threshold (tune this)
CLAP_MIN_INTERVAL = 0.4   # minimum seconds between claps

last_clap_time = 0.0
# ================================================


def distance(lm1, lm2, w, h):
    """Pixel distance between two normalized landmarks."""
    x1, y1 = lm1.x * w, lm1.y * h
    x2, y2 = lm2.x * w, lm2.y * h
    return math.hypot(x2 - x1, y2 - y1)


def eye_state_for_face(face, w, h):
    """
    Return (left_eye_state, right_eye_state, left_EAR, right_EAR)
    states: 'OPEN' or 'CLOSED'
    """
    lm = face.landmark

    # Right eye (our left)
    r_top = lm[159]
    r_bottom = lm[145]
    r_left = lm[33]
    r_right = lm[133]

    # Left eye (our right)
    l_top = lm[386]
    l_bottom = lm[374]
    l_left = lm[263]
    l_right = lm[362]

    r_vert = distance(r_top, r_bottom, w, h)
    r_horiz = distance(r_left, r_right, w, h)
    l_vert = distance(l_top, l_bottom, w, h)
    l_horiz = distance(l_left, l_right, w, h)

    if r_horiz == 0 or l_horiz == 0:
        return "UNKNOWN", "UNKNOWN", 0.0, 0.0

    r_ear = r_vert / r_horiz
    l_ear = l_vert / l_horiz

    thr = 0.22
    r_state = "OPEN" if r_ear > thr else "CLOSED"
    l_state = "OPEN" if l_ear > thr else "CLOSED"

    return l_state, r_state, round(l_ear, 3), round(r_ear, 3)


def count_extended_fingers_no_thumb(hand):
    """
    Count extended fingers ignoring thumb (index, middle, ring, pinky).
    """
    lm = hand.landmark
    count = 0
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):  # index–pinky
        if lm[tip].y < lm[pip].y - 0.02:
            count += 1
    return count


def is_thumb_extended_by_position(hand, handedness_label, w, h):
    """
    Decide if thumb is ON (visible/extended) using:
    - Its lateral placement (outermost on left/right side)
    - Its distance from wrist
    Thumb is NOT counted as a normal finger.
    """
    lm = hand.landmark
    wrist = lm[0]
    thumb_tip = lm[4]

    tips = [lm[idx] for idx in FINGER_TIPS]  # [thumb, idx, mid, ring, pinky]
    xs = [t.x * w for t in tips]
    thumb_x = xs[0]
    others_x = xs[1:]

    if handedness_label == "Right":
        # Right hand (user POV): thumb on our left side (smaller x)
        thumb_outer = thumb_x <= min(others_x) - 5
    else:
        # Left hand: thumb on our right side (larger x)
        thumb_outer = thumb_x >= max(others_x) + 5

    thumb_dist = distance(thumb_tip, wrist, w, h)
    middle_mcp = lm[9]
    palm_size = distance(wrist, middle_mcp, w, h) + 1e-6
    length_ratio = thumb_dist / palm_size

    extended_enough = length_ratio > 0.35

    return thumb_outer and extended_enough


def classify_hand_gesture(hand, handedness_label, w, h):
    """
    Gestures (thumb is special, not counted as finger):

    - OPEN_PALM: all 4 main fingers up (index–pinky)
    - If all 4 closed:
        - thumb ON  → CLOSED_PALM
        - thumb OFF → FIST
    - TWO_FINGERS: exactly 2 of (index–pinky) up → also returns direction.
    - PINCH: thumb & index touching, other fingers mostly down → pointer mode
    """
    lm = hand.landmark

    extended_no_thumb = count_extended_fingers_no_thumb(hand)
    thumb_on = is_thumb_extended_by_position(hand, handedness_label, w, h)
    total_fingers = extended_no_thumb   # thumb is not counted as "finger"
    wrist = lm[0]

    # Check pinch (thumb + index close)
    thumb_tip = lm[4]
    index_tip = lm[8]
    middle_mcp = lm[9]
    palm_size = distance(wrist, middle_mcp, w, h) + 1e-6
    thumb_index_dist = distance(thumb_tip, index_tip, w, h)
    pinch_ratio = thumb_index_dist / palm_size
    pinch = pinch_ratio < 0.3 and extended_no_thumb <= 1

    if pinch:
        # Special gesture for pointer mode
        return "PINCH", total_fingers, thumb_on, ""

    gesture = "UNKNOWN"
    direction = ""

    if extended_no_thumb == 4:
        gesture = "OPEN_PALM"

    elif extended_no_thumb == 0:
        if thumb_on:
            gesture = "CLOSED_PALM"
        else:
            gesture = "FIST"

    elif extended_no_thumb == 2:
        gesture = "TWO_FINGERS"
        dx = index_tip.x - wrist.x
        dy = index_tip.y - wrist.y

        if abs(dx) > abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
    else:
        gesture = "UNKNOWN"

    return gesture, total_fingers, thumb_on, direction


# ===== OS CONTROL LAYER =====

def run_xdotool(args, label):
    try:
        cmd = ["xdotool"] + args
        print(f"[CMD] {' '.join(cmd)} ({label})")
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("❌ xdotool not found. Install with: sudo apt install xdotool")
    except Exception as e:
        print("❌ xdotool error:", e)


def alt_down():
    run_xdotool(["keydown", "Alt"], "ALT down")


def alt_up():
    run_xdotool(["keyup", "Alt"], "ALT up")


def execute_one_shot(cmd_name):
    """Commands that happen once after hold (no blink)."""
    if cmd_name == "SHOW_APPS":
        # Super key to open overview / apps (dock show apps)
        run_xdotool(["key", "Super_L"], "SHOW_APPS (Super)")
    elif cmd_name == "SHOW_DESKTOP":
        # Super+D: show desktop / minimize all
        run_xdotool(["key", "Super+d"], "SHOW_DESKTOP (Super+d)")
    elif cmd_name == "CLOSE_WINDOW":
        run_xdotool(["key", "Alt+F4"], "CLOSE_WINDOW (Alt+F4)")


def blink_action(mode, alt_tab_dir):
    """What happens on each 'step' (blink or held eyes) in current mode."""
    if mode == "ALT_TAB":
        if alt_tab_dir == "RIGHT":
            run_xdotool(["key", "Right"], "ALT_TAB RIGHT arrow")
        elif alt_tab_dir == "LEFT":
            run_xdotool(["key", "Left"], "ALT_TAB LEFT arrow")

    elif mode == "SCROLL_UP":
        run_xdotool(["click", "4"], "SCROLL_UP (wheel up)")

    elif mode == "SCROLL_DOWN":
        run_xdotool(["click", "5"], "SCROLL_DOWN (wheel down)")

    elif mode == "ZOOM_IN":
        run_xdotool(["key", "ctrl+plus"], "ZOOM_IN (Ctrl+plus)")
        run_xdotool(["key", "ctrl+equal"], "ZOOM_IN (Ctrl+equal)")

    elif mode == "ZOOM_OUT":
        run_xdotool(["key", "ctrl+minus"], "ZOOM_OUT (Ctrl+minus)")


def get_screen_geometry():
    """Get screen size for pointer mode."""
    try:
        result = subprocess.run(
            ["xdotool", "getdisplaygeometry"],
            capture_output=True,
            text=True
        )
        parts = result.stdout.strip().split()
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    # Fallback
    return 1920, 1080


def move_mouse(x, y):
    run_xdotool(["mousemove", str(x), str(y)], "MOUSEMOVE")


# 🔊 ===== CLAP HANDLER (using existing run_xdotool) =====

def on_clap():
    """What to do on each clap: Alt+Tab."""
    run_xdotool(["key", "Alt+Tab"], "ALT+TAB (clap)")


def audio_callback(indata, frames, time_info, status):
    global last_clap_time

    # Convert audio chunk to mono float array
    audio = indata[:, 0]

    # Simple RMS (volume)
    rms = np.sqrt(np.mean(audio ** 2))

    now = time.time()

    # Detect clap: volume above threshold + not too soon after previous
    if rms > CLAP_THRESHOLD and (now - last_clap_time) > CLAP_MIN_INTERVAL:
        print(f"👏 Clap detected! RMS={rms:.3f}")
        last_clap_time = now
        on_clap()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    screen_w, screen_h = get_screen_geometry()

    # Mode state for blink-based controls
    mode = "NONE"          # ALT_TAB, SCROLL_UP, SCROLL_DOWN, ZOOM_IN, ZOOM_OUT, NONE
    mode_start_time = None
    mode_active = False
    alt_tab_dir = ""       # LEFT/RIGHT when in ALT_TAB
    last_repeat_time = 0.0

    # Hold-based command (no blink)
    hold_cmd = "NONE"      # SHOW_APPS, SHOW_DESKTOP, CLOSE_WINDOW
    hold_start_time = None
    hold_executed = False

    # Pointer mode (hand-controlled mouse)
    pointer_mode = False

    prev_eyes_closed = False
    prev_left_eye = "OPEN"
    prev_right_eye = "OPEN"

    print("🎧 Clap Alt+Tab active: every clap = Alt+Tab")
    print("🎥 Hand / eye controller running. Press 'q' to quit.\n")

    with mp_hands.Hands(
        max_num_hands=4,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands, mp_face_mesh.FaceMesh(
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
        callback=audio_callback
    ):

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_result = face_mesh.process(rgb)
            hand_result = hands.process(rgb)

            face_infos = []
            hand_infos = []
            current_left_eye = "OPEN"
            current_right_eye = "OPEN"

            # === Faces & eyes (for blink + wink + HUD) ===
            if face_result.multi_face_landmarks:
                for idx, face_landmarks in enumerate(face_result.multi_face_landmarks):
                    mp_draw.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None
                    )
                    l_state, r_state, l_ear, r_ear = eye_state_for_face(
                        face_landmarks, w, h
                    )
                    if idx == 0:
                        current_left_eye = l_state
                        current_right_eye = r_state

                    face_infos.append({
                        "left_eye": l_state,
                        "right_eye": r_state,
                        "left_ear": l_ear,
                        "right_ear": r_ear,
                    })

            # === Hands & gestures ===
            primary_hand_landmarks = None
            if hand_result.multi_hand_landmarks:
                handedness_list = hand_result.multi_handedness
                for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    handed_label = "Unknown"
                    if handedness_list and idx < len(handedness_list):
                        handed_label = handedness_list[idx].classification[0].label

                    gesture, total_fingers, thumb_on, direction = classify_hand_gesture(
                        hand_landmarks, handed_label, w, h
                    )

                    if idx == 0:
                        primary_hand_landmarks = hand_landmarks

                    hand_infos.append({
                        "gesture": gesture,
                        "total_fingers": total_fingers,
                        "thumb_on": thumb_on,
                        "direction": direction,
                        "handed": handed_label
                    })
            else:
                # No hands → stop modes & pointer
                if mode == "ALT_TAB" and mode_active:
                    alt_up()
                mode = "NONE"
                mode_start_time = None
                mode_active = False
                alt_tab_dir = ""
                hold_cmd = "NONE"
                hold_start_time = None
                hold_executed = False
                pointer_mode = False

            now = time.time()

            # === Decide control mode & hold-command from FIRST hand (if any) ===
            desired_mode = "NONE"
            desired_alt_tab_dir = ""
            new_hold_cmd = "NONE"
            pointer_pose = False

            if hand_infos:
                hinfo = hand_infos[0]
                g = hinfo["gesture"]
                fingers = hinfo["total_fingers"]
                thumb_on = hinfo["thumb_on"]
                direction = hinfo["direction"]

                # Pointer mode gesture: PINCH (index touching thumb)
                if g == "PINCH":
                    pointer_pose = True
                else:
                    # ⚠️ ALT_TAB gesture REMOVED here
                    # SCROLL mode: two fingers, thumb OFF, UP/DOWN
                    if g == "TWO_FINGERS" and not thumb_on and direction in ("UP", "DOWN"):
                        desired_mode = "SCROLL_UP" if direction == "UP" else "SCROLL_DOWN"

                    # Zoom in: thumb ON + one finger
                    elif fingers == 1 and thumb_on:
                        desired_mode = "ZOOM_IN"

                    # Zoom out: thumb ON + two fingers
                    elif g == "TWO_FINGERS" and thumb_on:
                        desired_mode = "ZOOM_OUT"

                    # Hold-based commands only if no mode
                    if desired_mode == "NONE":
                        # CLOSED_PALM -> close window
                        if g == "CLOSED_PALM":
                            new_hold_cmd = "CLOSE_WINDOW"
                        # One finger, thumb OFF -> show apps (Super)
                        elif fingers == 1 and not thumb_on:
                            new_hold_cmd = "SHOW_APPS"
                        # OPEN_PALM -> show desktop
                        elif g == "OPEN_PALM":
                            new_hold_cmd = "SHOW_DESKTOP"

            # Pointer mode on/off
            if pointer_pose:
                # Leaving ALT_TAB mode if active
                if mode == "ALT_TAB" and mode_active:
                    alt_up()
                mode = "NONE"
                mode_start_time = None
                mode_active = False
                alt_tab_dir = ""
                hold_cmd = "NONE"
                hold_start_time = None
                hold_executed = False
                pointer_mode = True
            else:
                pointer_mode = False

            # === Handle mode transitions ===
            prev_mode = mode

            # Leaving ALT_TAB → release Alt
            if prev_mode == "ALT_TAB" and desired_mode != "ALT_TAB" and mode_active:
                alt_up()

            if not pointer_mode:  # only change mode when not in pointer
                if desired_mode != prev_mode:
                    mode = desired_mode
                    mode_start_time = now if mode != "NONE" else None
                    mode_active = False
                    alt_tab_dir = desired_alt_tab_dir if mode == "ALT_TAB" else ""

            # Activate mode after hold time
            if mode != "NONE" and not mode_active and mode_start_time is not None:
                if now - mode_start_time >= GESTURE_HOLD_TIME:
                    if mode == "ALT_TAB":
                        alt_down()
                        run_xdotool(["key", "Tab"], "ALT_TAB initial Tab")
                    mode_active = True
                    last_repeat_time = 0.0

            # === Handle hold-based commands (show apps / desktop / close) ===
            if not pointer_mode and mode == "NONE":
                if new_hold_cmd != hold_cmd:
                    hold_cmd = new_hold_cmd
                    hold_start_time = now if hold_cmd != "NONE" else None
                    hold_executed = False
                else:
                    if hold_cmd != "NONE" and not hold_executed:
                        if now - hold_start_time >= GESTURE_HOLD_TIME:
                            print(f"🔥 Hold command: {hold_cmd}")
                            execute_one_shot(hold_cmd)
                            hold_executed = True
                            hold_cmd = "NONE"
                            hold_start_time = None

            # === Eye logic: blink + continuous hold ===
            eyes_closed = (current_left_eye == "CLOSED" and
                           current_right_eye == "CLOSED")
            blink_event = (not prev_eyes_closed) and eyes_closed

            # Winks for pointer clicks
            left_wink_event = (
                prev_left_eye == "OPEN"
                and current_left_eye == "CLOSED"
                and current_right_eye == "OPEN"
            )
            right_wink_event = (
                prev_right_eye == "OPEN"
                and current_right_eye == "CLOSED"
                and current_left_eye == "OPEN"
            )

            prev_eyes_closed = eyes_closed
            prev_left_eye = current_left_eye
            prev_right_eye = current_right_eye

            # Pointer mode mouse movement + clicks
            if pointer_mode and primary_hand_landmarks is not None:
                idx_tip = primary_hand_landmarks.landmark[8]
                mx = int(idx_tip.x * screen_w)
                my = int(idx_tip.y * screen_h)
                move_mouse(mx, my)

                if left_wink_event:
                    run_xdotool(["click", "1"], "POINTER LEFT CLICK")
                if right_wink_event:
                    run_xdotool(["click", "3"], "POINTER RIGHT CLICK")

            # Continuous actions for ALT_TAB/SCROLL with eyes held closed
            if mode_active and not pointer_mode:
                if mode in ("ALT_TAB", "SCROLL_UP", "SCROLL_DOWN"):
                    if eyes_closed:
                        if last_repeat_time == 0.0 or now - last_repeat_time >= REPEAT_INTERVAL:
                            blink_action(mode, alt_tab_dir)
                            last_repeat_time = now
                    else:
                        last_repeat_time = 0.0
                elif mode in ("ZOOM_IN", "ZOOM_OUT"):
                    # Zoom stays blink-based (not continuous hold)
                    if blink_event:
                        blink_action(mode, alt_tab_dir)

            # === HUD top-left ===
            x0, y0 = 10, 25
            line_h = 22

            cv2.putText(
                frame,
                f"Faces: {len(face_infos)}",
                (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )
            y = y0 + line_h

            for i, info in enumerate(face_infos):
                txt = (f"Face {i+1} | L:{info['left_eye']} R:{info['right_eye']} "
                       f"| L_EAR:{info['left_ear']} R_EAR:{info['right_ear']}")
                cv2.putText(
                    frame,
                    txt,
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA
                )
                y += line_h

            cv2.putText(
                frame,
                f"Hands: {len(hand_infos)}",
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            y += line_h

            for i, info in enumerate(hand_infos):
                txt = (f"Hand {i+1} ({info['handed']}) | G:{info['gesture']} "
                       f"| Fingers:{info['total_fingers']} "
                       f"| Thumb:{'ON' if info['thumb_on'] else 'OFF'}")
                cv2.putText(
                    frame,
                    txt,
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 255, 200),
                    1,
                    cv2.LINE_AA
                )
                y += line_h

                if info["gesture"] == "TWO_FINGERS" and info["direction"]:
                    txt2 = f"        Dir: {info['direction']}"
                    cv2.putText(
                        frame,
                        txt2,
                        (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (150, 255, 150),
                        1,
                        cv2.LINE_AA
                    )
                    y += line_h

            # Show mode / hold / pointer info
            status = f"MODE: {mode} | POINTER: {'ON' if pointer_mode else 'OFF'}"
            if mode != "NONE" and mode_start_time is not None and not mode_active:
                remaining = max(0.0, GESTURE_HOLD_TIME - (now - mode_start_time))
                status += f" | Activating in: {remaining:.1f}s"
            if mode == "ALT_TAB" and alt_tab_dir:
                status += f" | Dir: {alt_tab_dir}"

            if hold_cmd != "NONE":
                if hold_start_time is not None:
                    remaining_hold = max(0.0, GESTURE_HOLD_TIME - (now - hold_start_time))
                else:
                    remaining_hold = 0.0
                status += f" | HOLD_CMD: {hold_cmd} ({remaining_hold:.1f}s)"

            cv2.putText(
                frame,
                status,
                (x0, y + line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Gesture OS Controller - Scroll/AltTab+Mouse+Clap", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)

