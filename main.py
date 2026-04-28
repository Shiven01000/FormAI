# FormAI — AI Fitness Coach
# Run with:  python main.py
#
# Controls (video window must be focused):
#   1–6  switch exercise    R  reset count    Q  quit / view summary

import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess
import sounddevice as sd

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from angles       import calculate_angle
from config       import EXERCISES, EXERCISE_KEYS
from rep_counter  import RepCounter
from form_checker import FormChecker


# ── Color palette (BGR order) ─────────────────────────────────────────────────
PANEL_BG      = (18, 18, 18)
ELECTRIC_BLUE = (255, 120, 30)
GREEN_GOOD    = (60, 210, 60)
ORANGE_WARN   = (30, 140, 255)
WHITE         = (255, 255, 255)
GRAY          = (150, 150, 150)
YELLOW_REP    = (40, 220, 255)
CYAN_STAGE    = (220, 200, 0)

# Upper-body + lower-body connections from MediaPipe's 33-point skeleton
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_panel(img, x, y, w, h, alpha=0.72, radius=14, border_color=ELECTRIC_BLUE):
    """
    Render a semi-transparent rounded rectangle (glassmorphism style).

    OpenCV has no native rounded-rect primitive, so we approximate one with
    two crossing rectangles + four corner circles, then blend onto the frame
    with addWeighted to achieve transparency.
    """
    x2, y2  = x + w, y + h
    overlay = img.copy()

    cv2.rectangle(overlay, (x + radius, y),         (x2 - radius, y2),          PANEL_BG, -1)
    cv2.rectangle(overlay, (x,          y + radius), (x2,          y2 - radius), PANEL_BG, -1)
    cv2.circle(overlay, (x  + radius, y  + radius), radius, PANEL_BG, -1)
    cv2.circle(overlay, (x2 - radius, y  + radius), radius, PANEL_BG, -1)
    cv2.circle(overlay, (x  + radius, y2 - radius), radius, PANEL_BG, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, PANEL_BG, -1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    if border_color:
        cv2.line(img, (x + radius, y),  (x2 - radius, y),  border_color, 1)
        cv2.line(img, (x + radius, y2), (x2 - radius, y2), border_color, 1)
        cv2.line(img, (x,  y + radius), (x,  y2 - radius), border_color, 1)
        cv2.line(img, (x2, y + radius), (x2, y2 - radius), border_color, 1)


def draw_skeleton(frame, landmarks, good_form, active_joint_idx):
    """
    Render the pose skeleton with form-based color (green/orange) and a
    double-ring highlight on the joint currently being measured.
    """
    h, w, _ = frame.shape
    color    = GREEN_GOOD if good_form else ORANGE_WARN
    pts      = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for s, e in POSE_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], color, 2)

    for i, pt in enumerate(pts):
        if i == active_joint_idx:
            cv2.circle(frame, pt, 10, WHITE,         -1)
            cv2.circle(frame, pt, 10, ELECTRIC_BLUE,  2)
            cv2.circle(frame, pt, 15, ELECTRIC_BLUE,  1)
        else:
            cv2.circle(frame, pt, 4, WHITE,  -1)
            cv2.circle(frame, pt, 4, color,   1)


def draw_rep_flash(frame, flash_start_time):
    """Animate a "+1" that drifts upward and fades over 0.5 s."""
    if flash_start_time is None:
        return
    elapsed = time.time() - flash_start_time
    if elapsed >= 0.5:
        return

    h, w, _ = frame.shape
    alpha = 1.0 - (elapsed / 0.5)
    scale = 3.5 + elapsed * 1.5
    thick = max(1, int(7 * alpha))
    color = tuple(int(c * alpha) for c in GREEN_GOOD)

    text = "+1"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    cv2.putText(frame, text,
                ((w - tw) // 2, h // 2 - int(120 * elapsed)),
                cv2.FONT_HERSHEY_DUPLEX, scale, color, thick)


def draw_workout_ui(frame, angle, rep_count, stage, exercise_config,
                    exercise_number, feedback_text, good_form, total_reps):
    """
    HUD layout:
      Left panel  — exercise name, rep count, stage, form feedback
      Right panel — session total, current set, progress bar (0–10 reps)
      Bottom bar  — joint angle | keyboard hint
    """
    h, w, _ = frame.shape

    # Left panel
    draw_panel(frame, 12, 12, 245, 215)
    cv2.putText(frame, f"{exercise_number}. {exercise_config['display_name']}",
                (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ELECTRIC_BLUE, 2)
    cv2.putText(frame, "REPS", (24, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
    cv2.putText(frame, str(rep_count), (24, 150),
                cv2.FONT_HERSHEY_DUPLEX, 4.0, YELLOW_REP, 8)
    stage_color = CYAN_STAGE if stage == "up" else WHITE
    cv2.putText(frame, stage.upper() if stage else "READY",
                (24, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.65, stage_color, 2)
    cv2.putText(frame, feedback_text, (24, 208),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                GREEN_GOOD if good_form else ORANGE_WARN, 2)

    # Right panel
    px = w - 195
    draw_panel(frame, px, 12, 182, 135)
    cv2.putText(frame, "SESSION",           (px + 12, 34),  cv2.FONT_HERSHEY_SIMPLEX, 0.48, GRAY,  1)
    cv2.putText(frame, f"Total: {total_reps} reps", (px + 12, 58),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    cv2.putText(frame, f"Set: {rep_count // 10 + 1}", (px + 12, 82),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    reps_in_set = rep_count % 10
    bar_x, bar_y, bar_max_w = px + 12, 96, 158
    bar_fill_w = int((reps_in_set / 10) * bar_max_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 13), (55, 55, 55), -1)
    if bar_fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill_w, bar_y + 13), ELECTRIC_BLUE, -1)
    cv2.putText(frame, f"{reps_in_set}/10 this set",
                (px + 12, bar_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)

    # Bottom bar
    cv2.putText(frame, f"{exercise_config['joint_label']}: {int(angle)}\u00b0",
                (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, YELLOW_REP, 2)
    hint = "1-6: switch   R: reset   Q: quit"
    (hint_w, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, hint, (w - hint_w - 10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)


# ── Exercise selection menu ───────────────────────────────────────────────────

def run_exercise_menu(camera):
    """
    Render a 2×3 card grid over the live camera feed.
    Returns the chosen exercise key, or None if the user presses Q.
    """
    print("Exercise menu — press 1–6 to begin (video window must be focused).")

    while True:
        success, frame = camera.read()
        if not success:
            continue

        h, w, _ = frame.shape
        cv2.addWeighted(frame, 0.35, np.zeros_like(frame), 0.65, 0, frame)

        # Title
        title_panel_w = min(420, w - 40)
        title_x = (w - title_panel_w) // 2
        draw_panel(frame, title_x, 10, title_panel_w, 58, alpha=0.88)
        title = "FormAI — Select Your Exercise"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        cv2.putText(frame, title, ((w - tw) // 2, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, WHITE, 2)
        sub = "Press 1-6 to begin"
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(frame, sub, ((w - sw) // 2, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)

        # Exercise cards
        margin, gap, card_w, card_h, start_y = 14, 10, (w - 2*14 - 10) // 2, 95, 80

        for i, key in enumerate(EXERCISE_KEYS):
            col = i % 2
            row = i // 2
            cx  = margin + col * (card_w + gap)
            cy  = start_y + row * (card_h + 8)

            draw_panel(frame, cx, cy, card_w, card_h, alpha=0.82)
            cv2.circle(frame, (cx + 22, cy + 22), 16, ELECTRIC_BLUE, -1)
            cv2.putText(frame, str(i + 1), (cx + 16, cy + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)
            cv2.putText(frame, EXERCISES[key]["display_name"], (cx + 46, cy + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 2)
            cv2.putText(frame, EXERCISES[key]["description"], (cx + 46, cy + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
            cv2.putText(frame, f"Tracks: {EXERCISES[key]['joint_label']} angle",
                        (cx + 46, cy + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.37, ELECTRIC_BLUE, 1)

        cv2.imshow("FormAI", frame)
        key = cv2.waitKey(1) & 0xFF
        if ord("1") <= key <= ord("6"):
            chosen = EXERCISE_KEYS[key - ord("1")]
            print(f"  Selected: {EXERCISES[chosen]['display_name']}\n")
            return chosen
        if key == ord("q"):
            return None


# ── Audio ─────────────────────────────────────────────────────────────────────

def play_beep(freq=880, duration=0.07, volume=0.35):
    """Non-blocking sine-wave beep via sounddevice."""
    try:
        sr   = 44100
        t    = np.linspace(0, duration, int(sr * duration), False)
        wave = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sd.play(wave, sr)
    except Exception:
        pass


class VoiceAnnouncer:
    """
    Mac TTS via subprocess.Popen(['say', ...]) — non-blocking, runs in background.

    say_feedback() — only when the message changes; won't interrupt ongoing speech.
    announce()     — milestone messages; always speaks, terminates current speech.
    """

    def __init__(self):
        self._process       = None
        self._last_feedback = ""

    def _is_speaking(self):
        return self._process is not None and self._process.poll() is None

    def say_feedback(self, message):
        if message == self._last_feedback:
            return
        self._last_feedback = message
        if not self._is_speaking():
            self._process = subprocess.Popen(
                ["say", "-r", "210", message],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    def announce(self, message):
        if self._is_speaking():
            self._process.terminate()
        self._process = subprocess.Popen(
            ["say", "-r", "210", message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


# ── Session stats ─────────────────────────────────────────────────────────────

class SessionStats:
    """
    Accumulates reps across exercise switches for the end-of-session summary.

    Call bank() before switching exercises or quitting to freeze the live count.
    total_reps() adds banked totals + the current live count for the HUD.
    """

    def __init__(self):
        self.start_time      = time.time()
        self.exercise_totals = {}

    def bank(self, key, count):
        self.exercise_totals[key] = self.exercise_totals.get(key, 0) + count

    def total_reps(self, live_count):
        return sum(self.exercise_totals.values()) + live_count

    def print_summary(self):
        total = sum(self.exercise_totals.values())
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        print("\n" + "━" * 38)
        print("  FORMAI — SESSION COMPLETE")
        print("━" * 38)
        print(f"  Duration  :  {mins}m {secs:02d}s")
        print(f"  Total reps:  {total}")
        print()
        for key, count in self.exercise_totals.items():
            if count > 0:
                print(f"  {EXERCISES[key]['display_name']:<22} {count:>3} reps")
        print("━" * 38 + "\n")


# ── Summary screen ────────────────────────────────────────────────────────────

def show_summary_screen(camera, stats):
    """
    Full-screen glassmorphism summary panel. Loops until the user presses Q.
    Call after stats.bank() has been called for the final exercise.
    """
    total   = sum(stats.exercise_totals.values())
    elapsed = int(time.time() - stats.start_time)
    mins, secs = divmod(elapsed, 60)
    exercises_done = [(k, v) for k, v in stats.exercise_totals.items() if v > 0]

    while True:
        success, frame = camera.read()
        if not success:
            break

        h, w, _ = frame.shape
        cv2.addWeighted(frame, 0.18, np.zeros_like(frame), 0.82, 0, frame)

        panel_h = 160 + max(len(exercises_done), 1) * 34 + 50
        panel_w = min(460, w - 40)
        panel_x = (w - panel_w) // 2
        panel_y = (h - panel_h) // 2

        draw_panel(frame, panel_x, panel_y, panel_w, panel_h, alpha=0.92)

        title = "SESSION COMPLETE"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        cv2.putText(frame, title, ((w - tw) // 2, panel_y + 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, ELECTRIC_BLUE, 2)

        div_y = panel_y + 58
        cv2.line(frame, (panel_x + 20, div_y), (panel_x + panel_w - 20, div_y), ELECTRIC_BLUE, 1)

        y = panel_y + 85
        cv2.putText(frame, f"Duration:    {mins}m {secs:02d}s",
                    (panel_x + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 2)
        y += 32
        cv2.putText(frame, f"Total reps:  {total}",
                    (panel_x + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 2)
        y += 42
        cv2.putText(frame, "By exercise:", (panel_x + 24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

        for key, count in exercises_done:
            y += 34
            cv2.putText(frame, EXERCISES[key]["display_name"], (panel_x + 36, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
            count_text = f"{count} reps"
            (cw, _), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(frame, count_text, (panel_x + panel_w - cw - 24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW_REP, 1)

        footer = "Press Q again to exit"
        (fw, _), _ = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, footer, ((w - fw) // 2, panel_y + panel_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)

        cv2.imshow("FormAI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    active_key = run_exercise_menu(camera)
    if active_key is None:
        camera.release()
        cv2.destroyAllWindows()
        return

    active_config = EXERCISES[active_key]
    active_number = EXERCISE_KEYS.index(active_key) + 1

    counter      = RepCounter()
    form_checker = FormChecker()
    voice        = VoiceAnnouncer()
    stats        = SessionStats()

    feedback_text  = "Step into frame"
    good_form      = True
    current_angle  = 0.0
    active_joint   = active_config["landmarks"][1]
    rep_flash_time = None
    prev_rep_count = 0

    voice.announce(f"Starting {active_config['display_name']}")
    print("Controls: 1–6 switch exercise   R reset   Q quit")
    print("(Keyboard shortcuts work in the VIDEO window)\n")

    base_options = mp_python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()
            if not success:
                break

            # Pose detection
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Joint angle — bilateral exercises track the more-active arm
                idx_a, idx_b, idx_c = active_config["landmarks"]
                angle_left = calculate_angle(
                    [landmarks[idx_a].x, landmarks[idx_a].y],
                    [landmarks[idx_b].x, landmarks[idx_b].y],
                    [landmarks[idx_c].x, landmarks[idx_c].y],
                )

                if active_config.get("bilateral"):
                    r_a, r_b, r_c = active_config["landmarks_right"]
                    angle_right = calculate_angle(
                        [landmarks[r_a].x, landmarks[r_a].y],
                        [landmarks[r_b].x, landmarks[r_b].y],
                        [landmarks[r_c].x, landmarks[r_c].y],
                    )
                    # Whichever arm deviates more from fully straight is active
                    if abs(angle_left - 180) >= abs(angle_right - 180):
                        current_angle, active_joint = angle_left, idx_b
                    else:
                        current_angle, active_joint = angle_right, r_b
                else:
                    current_angle = angle_left
                    active_joint  = idx_b

                # Form check must run before counter.update so good_form is ready
                feedback_text, good_form = form_checker.check(
                    active_key, landmarks, current_angle, counter.stage
                )

                counter.update(
                    current_angle,
                    active_config["down_angle"],
                    active_config["up_angle"],
                    good_form,
                )

                # Rep completion events
                if counter.count > prev_rep_count:
                    rep_flash_time = time.time()
                    play_beep()
                    if counter.count % 5 == 0:
                        voice.announce(f"{counter.count} reps")
                    if counter.count % 10 == 0:
                        voice.announce(f"Set {counter.count // 10} complete")
                    prev_rep_count = counter.count

                voice.say_feedback(feedback_text)
                draw_skeleton(frame, landmarks, good_form, active_joint)

            draw_workout_ui(
                frame, current_angle, counter.count, counter.stage,
                active_config, active_number, feedback_text, good_form,
                stats.total_reps(counter.count),
            )
            draw_rep_flash(frame, rep_flash_time)
            cv2.imshow("FormAI", frame)

            key_press = cv2.waitKey(1) & 0xFF

            if key_press == ord("q"):
                stats.bank(active_key, counter.count)
                show_summary_screen(camera, stats)
                stats.print_summary()
                break

            elif key_press == ord("r"):
                counter.reset()
                prev_rep_count = 0
                print(f"  Reset: {active_config['display_name']}")

            elif ord("1") <= key_press <= ord("6"):
                new_key = EXERCISE_KEYS[key_press - ord("1")]
                if new_key != active_key:
                    stats.bank(active_key, counter.count)
                    active_key     = new_key
                    active_config  = EXERCISES[active_key]
                    active_number  = key_press - ord("1") + 1
                    active_joint   = active_config["landmarks"][1]
                    counter.reset()
                    prev_rep_count = 0
                    feedback_text  = "Step into frame"
                    good_form      = True
                    voice.announce(f"Switching to {active_config['display_name']}")
                    print(f"  Switched to: {active_config['display_name']}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
