# main.py
#
# FormAI — AI Fitness Coach
# Run with:  python main.py
#
# Controls (in the VIDEO window):
#   1–6  → switch exercise    R → reset count    Q → quit

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


# ── Color palette (BGR) ───────────────────────────────────────────────────────
# A note on BGR: OpenCV stores colors as (Blue, Green, Red), not (Red, Green, Blue).
# So RGB(30, 120, 255) becomes BGR(255, 120, 30).

PANEL_BG      = (18, 18, 18)       # near-black panel fill
ELECTRIC_BLUE = (255, 120, 30)     # RGB(30, 120, 255) — accent / border
GREEN_GOOD    = (60, 210, 60)      # good form indicator
ORANGE_WARN   = (30, 140, 255)     # RGB(255, 140, 30) — warning / bad form
WHITE         = (255, 255, 255)
GRAY          = (150, 150, 150)
YELLOW_REP    = (40, 220, 255)     # RGB(255, 220, 40) — rep count highlight
CYAN_STAGE    = (220, 200, 0)      # RGB(0, 200, 220) — stage label

# ── Skeleton connections ──────────────────────────────────────────────────────
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_panel(img, x, y, w, h, alpha=0.72, radius=14,
               border_color=ELECTRIC_BLUE):
    """
    Draw a semi-transparent rounded rectangle (glassmorphism panel).

    OpenCV has no native rounded-rect, so we build one from overlapping shapes:
      - Two filled rectangles forming a cross/plus shape
      - Four filled circles plugging in the corners
    Together these exactly cover a rounded-rectangle area.

    We draw onto a copy of the frame (overlay), then blend it back using
    addWeighted so it appears transparent.
    """
    x2, y2  = x + w, y + h
    overlay = img.copy()

    # The two rectangles + four corner circles fill the rounded-rect shape
    cv2.rectangle(overlay, (x + radius, y),        (x2 - radius, y2), PANEL_BG, -1)
    cv2.rectangle(overlay, (x,          y + radius),(x2,          y2 - radius), PANEL_BG, -1)
    cv2.circle(overlay, (x  + radius, y  + radius), radius, PANEL_BG, -1)
    cv2.circle(overlay, (x2 - radius, y  + radius), radius, PANEL_BG, -1)
    cv2.circle(overlay, (x  + radius, y2 - radius), radius, PANEL_BG, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, PANEL_BG, -1)

    # Blend: final = overlay * alpha + original * (1 - alpha)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Subtle border (straight lines along the flat edges — clean enough)
    if border_color:
        cv2.line(img, (x + radius, y),  (x2 - radius, y),  border_color, 1)
        cv2.line(img, (x + radius, y2), (x2 - radius, y2), border_color, 1)
        cv2.line(img, (x,  y + radius), (x,  y2 - radius), border_color, 1)
        cv2.line(img, (x2, y + radius), (x2, y2 - radius), border_color, 1)


def draw_skeleton(frame, landmarks, good_form, active_joint_idx):
    """
    Draw the skeleton with two new features vs earlier steps:

    1. Form-based color:
       Green skeleton = good form.  Orange skeleton = correction needed.
       This gives instant visual feedback without reading any text.

    2. Active joint highlight:
       The joint being measured (elbow, knee, etc.) gets a larger double-ring
       so it's obvious what the app is watching.
    """
    h, w, _ = frame.shape
    skel_color = GREEN_GOOD if good_form else ORANGE_WARN
    positions  = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for s, e in POSE_CONNECTIONS:
        cv2.line(frame, positions[s], positions[e], skel_color, 2)

    for i, pos in enumerate(positions):
        if i == active_joint_idx:
            # Active joint: solid white fill, two rings in electric blue
            cv2.circle(frame, pos, 10, WHITE,         -1)
            cv2.circle(frame, pos, 10, ELECTRIC_BLUE,  2)
            cv2.circle(frame, pos, 15, ELECTRIC_BLUE,  1)
        else:
            cv2.circle(frame, pos, 4, WHITE,      -1)
            cv2.circle(frame, pos, 4, skel_color,  1)


def draw_rep_flash(frame, flash_start_time):
    """
    Animate a "+1" that drifts upward and fades out over 0.5 seconds.

    time.time() returns the current Unix timestamp as a float.
    Subtracting the start time gives elapsed seconds.

    We compute alpha (opacity) as a value from 1.0 → 0.0:
        alpha = 1.0 - (elapsed / 0.5)

    We can't make OpenCV text truly transparent, so we multiply the color
    channels by alpha — the text approaches black (invisible on dark overlay)
    as it fades.
    """
    if flash_start_time is None:
        return
    elapsed = time.time() - flash_start_time
    if elapsed >= 0.5:
        return

    h, w, _ = frame.shape
    alpha    = 1.0 - (elapsed / 0.5)        # 1.0 at frame 0, 0.0 at 0.5s
    scale    = 3.5 + elapsed * 1.5          # grows slightly as it rises
    thick    = max(1, int(7 * alpha))
    color    = tuple(int(c * alpha) for c in GREEN_GOOD)

    # Center the "+1" horizontally; drift upward as elapsed increases
    text = "+1"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    cx = (w - tw) // 2
    cy = h // 2 - int(120 * elapsed)        # starts mid-frame, floats up

    cv2.putText(frame, text, (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX, scale, color, thick)


def draw_workout_ui(frame, angle, rep_count, stage, exercise_config,
                    exercise_number, feedback_text, good_form, total_reps):
    """
    Full workout HUD with two panels:

    LEFT PANEL  — live workout data
      Exercise name, rep count (large), stage, form feedback

    RIGHT PANEL — session stats
      Total reps across all exercises, current set, progress bar (0–10)

    BOTTOM BAR  — joint angle (left) + keyboard hint (right)
    """
    h, w, _ = frame.shape

    # ── Left panel ─────────────────────────────────────────────────────────
    draw_panel(frame, 12, 12, 245, 215)

    cv2.putText(frame, f"{exercise_number}. {exercise_config['display_name']}",
                (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ELECTRIC_BLUE, 2)

    cv2.putText(frame, "REPS", (24, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

    # Rep count — large font, heavy stroke
    cv2.putText(frame, str(rep_count), (24, 150),
                cv2.FONT_HERSHEY_DUPLEX, 4.0, YELLOW_REP, 8)

    stage_text  = stage.upper() if stage else "READY"
    stage_color = CYAN_STAGE if stage == "up" else WHITE
    cv2.putText(frame, stage_text, (24, 182),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, stage_color, 2)

    fb_color = GREEN_GOOD if good_form else ORANGE_WARN
    cv2.putText(frame, feedback_text, (24, 208),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, fb_color, 2)

    # ── Right panel ─────────────────────────────────────────────────────────
    px = w - 195
    draw_panel(frame, px, 12, 182, 135)

    cv2.putText(frame, "SESSION", (px + 12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, GRAY, 1)

    cv2.putText(frame, f"Total: {total_reps} reps", (px + 12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    current_set = rep_count // 10 + 1
    cv2.putText(frame, f"Set: {current_set}", (px + 12, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    # Progress bar — fills as reps_in_set approaches 10
    reps_in_set = rep_count % 10
    bar_x, bar_y, bar_max_w = px + 12, 96, 158
    bar_fill_w = int((reps_in_set / 10) * bar_max_w)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 13),
                  (55, 55, 55), -1)
    if bar_fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill_w, bar_y + 13),
                      ELECTRIC_BLUE, -1)

    cv2.putText(frame, f"{reps_in_set}/10 this set", (px + 12, bar_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)

    # ── Bottom bar ───────────────────────────────────────────────────────────
    angle_text = f"{exercise_config['joint_label']}: {int(angle)}\u00b0"
    cv2.putText(frame, angle_text, (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, YELLOW_REP, 2)

    hint = "1-6: switch   R: reset   Q: quit"
    (hint_w, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, hint, (w - hint_w - 10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)


# ─────────────────────────────────────────────────────────────────────────────
# ON-SCREEN EXERCISE MENU
# ─────────────────────────────────────────────────────────────────────────────

def run_exercise_menu(camera):
    """
    Show an on-screen exercise selection menu over the live camera feed.
    Returns the chosen exercise key (e.g. "bicep_curl"), or None if Q is pressed.

    Replaces the old terminal input() menu entirely — the video window IS
    the menu. The user presses 1–6 to pick and the workout starts immediately.
    """
    print("Exercise menu open in video window — press 1–6 to begin.")

    while True:
        success, frame = camera.read()
        if not success:
            continue

        h, w, _ = frame.shape

        # Dim the background so menu panels stand out clearly
        cv2.addWeighted(frame, 0.35, np.zeros_like(frame), 0.65, 0, frame)

        # ── Title panel ──────────────────────────────────────────────────────
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

        # ── Exercise cards — 2 columns × 3 rows ──────────────────────────────
        # Card dimensions fill the available width minus margins
        margin   = 14
        gap      = 10
        card_w   = (w - 2 * margin - gap) // 2
        card_h   = 95
        start_y  = 80

        for i, key in enumerate(EXERCISE_KEYS):
            col  = i % 2
            row  = i // 2
            cx   = margin + col * (card_w + gap)
            cy   = start_y + row * (card_h + 8)

            draw_panel(frame, cx, cy, card_w, card_h, alpha=0.82)

            # Number badge (filled circle + number)
            badge_center = (cx + 22, cy + 22)
            cv2.circle(frame, badge_center, 16, ELECTRIC_BLUE, -1)
            cv2.putText(frame, str(i + 1), (cx + 16, cy + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)

            # Exercise name
            cv2.putText(frame, EXERCISES[key]["display_name"],
                        (cx + 46, cy + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 2)

            # Short description
            cv2.putText(frame, EXERCISES[key]["description"],
                        (cx + 46, cy + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)

            # Tracks label (joint)
            joint_hint = f"Tracks: {EXERCISES[key]['joint_label']} angle"
            cv2.putText(frame, joint_hint, (cx + 46, cy + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, ELECTRIC_BLUE, 1)

        cv2.imshow("FormAI", frame)

        key = cv2.waitKey(1) & 0xFF
        if ord("1") <= key <= ord("6"):
            chosen_key = EXERCISE_KEYS[key - ord("1")]
            print(f"  Selected: {EXERCISES[chosen_key]['display_name']}\n")
            return chosen_key
        if key == ord("q"):
            return None


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────────────────────────────────────

def play_beep(freq=880, duration=0.07, volume=0.35):
    """
    Play a short confirmation beep using sounddevice (already installed).

    We generate the sound mathematically:
      - np.linspace gives us evenly-spaced time points from 0 → duration
      - np.sin produces a pure sine wave at the given frequency (Hz)
      - Multiplying by volume scales the amplitude (loudness)

    sd.play() is non-blocking — it starts a background audio thread and
    returns immediately, so the video loop is never paused.
    """
    try:
        sr  = 44100
        t   = np.linspace(0, duration, int(sr * duration), False)
        wave = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sd.play(wave, sr)
    except Exception:
        pass   # silently skip if no audio device is available


# ─────────────────────────────────────────────────────────────────────────────
# VOICE ANNOUNCER
# ─────────────────────────────────────────────────────────────────────────────

class VoiceAnnouncer:
    """
    Speaks text aloud using the Mac's built-in 'say' command.

    subprocess.Popen() launches 'say' as a separate OS process in the
    background. Unlike subprocess.run(), it returns immediately — the video
    loop keeps running at full speed while the Mac speaks.

    _process.poll() returns None while the process is still running,
    or an integer exit code once it has finished. We use this to check
    whether speech is currently happening.

    Two speak modes:
      say_feedback()  — form corrections. Only speaks when the message
                        changes, and waits for current speech to finish first.
      announce()      — milestones (rep counts, set complete, exercise switch).
                        Always speaks; interrupts any current speech.
    """

    def __init__(self):
        self._process       = None
        self._last_feedback = ""

    def _is_speaking(self):
        return self._process is not None and self._process.poll() is None

    def say_feedback(self, message):
        """Speak form feedback only when it changes and not currently speaking."""
        if message == self._last_feedback:
            return
        self._last_feedback = message
        if not self._is_speaking():
            self._process = subprocess.Popen(
                ["say", "-r", "210", message],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def announce(self, message):
        """Speak a milestone announcement, interrupting current speech if needed."""
        if self._is_speaking():
            self._process.terminate()
        self._process = subprocess.Popen(
            ["say", "-r", "210", message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATS
# ─────────────────────────────────────────────────────────────────────────────

class SessionStats:
    """
    Tracks cumulative reps across all exercises for the end-of-session summary.

    When you switch exercises the counter resets to 0, so we "bank" the current
    count into exercise_totals before resetting.

    total_reps() adds the banked totals + the live counter so the stats panel
    always shows the true running total for the whole session.
    """

    def __init__(self):
        self.start_time      = time.time()
        self.exercise_totals = {}    # { exercise_key: cumulative reps }

    def bank(self, key, count):
        """Save reps for a completed exercise stint. Call before switching."""
        self.exercise_totals[key] = self.exercise_totals.get(key, 0) + count

    def total_reps(self, live_count):
        return sum(self.exercise_totals.values()) + live_count

    def print_summary(self, last_key, last_count):
        """Print a formatted summary to the terminal when the user quits."""
        self.bank(last_key, last_count)
        total   = sum(self.exercise_totals.values())
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
                name = EXERCISES[key]["display_name"]
                print(f"  {name:<22} {count:>3} reps")
        print("━" * 38 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Open camera early so the menu can use it ──────────────────────────────
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # ── On-screen exercise menu ───────────────────────────────────────────────
    active_key = run_exercise_menu(camera)
    if active_key is None:    # user pressed Q on the menu screen
        camera.release()
        cv2.destroyAllWindows()
        return

    active_config = EXERCISES[active_key]
    active_number = EXERCISE_KEYS.index(active_key) + 1

    # ── Core objects ──────────────────────────────────────────────────────────
    counter      = RepCounter()
    form_checker = FormChecker()
    voice        = VoiceAnnouncer()
    stats        = SessionStats()

    # ── UI / animation state ──────────────────────────────────────────────────
    feedback_text  = "Step into frame"
    good_form      = True
    current_angle  = 0.0
    active_joint   = active_config["landmarks"][1]   # middle landmark index
    rep_flash_time = None     # timestamp of last completed rep (None = no anim)
    prev_rep_count = 0        # track rep completions frame-to-frame

    voice.announce(f"Starting {active_config['display_name']}")
    print("Controls: 1–6 switch exercise   R reset   Q quit")
    print("(Keyboard shortcuts work in the VIDEO window)\n")

    # ── MediaPipe setup ───────────────────────────────────────────────────────
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

            # ── Pose detection ────────────────────────────────────────────────
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # ── Angle — bilateral or single-side ─────────────────────────
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
                    # Whichever arm is more bent (further from 180°) is active
                    if abs(angle_left - 180) >= abs(angle_right - 180):
                        current_angle = angle_left
                        active_joint  = idx_b
                    else:
                        current_angle = angle_right
                        active_joint  = r_b
                else:
                    current_angle = angle_left
                    active_joint  = idx_b

                # ── Form feedback (must run before counter.update) ────────────
                feedback_text, good_form = form_checker.check(
                    active_key, landmarks, current_angle, counter.stage
                )

                # ── Rep counting ──────────────────────────────────────────────
                counter.update(
                    current_angle,
                    active_config["down_angle"],
                    active_config["up_angle"],
                    good_form,
                )

                # ── Rep completion events ─────────────────────────────────────
                if counter.count > prev_rep_count:
                    rep_flash_time = time.time()   # start +1 animation
                    play_beep()                    # audio confirmation

                    # Speak every 5th rep
                    if counter.count % 5 == 0:
                        voice.announce(f"{counter.count} reps")

                    # Announce set completion (every 10 reps)
                    if counter.count % 10 == 0:
                        set_num = counter.count // 10
                        voice.announce(f"Set {set_num} complete")

                    prev_rep_count = counter.count

                # ── Voice form feedback ───────────────────────────────────────
                # Only speaks when the message changes; doesn't interrupt milestones
                voice.say_feedback(feedback_text)

                # ── Skeleton ──────────────────────────────────────────────────
                draw_skeleton(frame, landmarks, good_form, active_joint)

            # ── HUD overlay ───────────────────────────────────────────────────
            draw_workout_ui(
                frame, current_angle, counter.count, counter.stage,
                active_config, active_number, feedback_text, good_form,
                stats.total_reps(counter.count),
            )

            # ── +1 flash animation ────────────────────────────────────────────
            draw_rep_flash(frame, rep_flash_time)

            cv2.imshow("FormAI", frame)

            # ── Keyboard handling ─────────────────────────────────────────────
            key_press = cv2.waitKey(1) & 0xFF

            if key_press == ord("q"):
                stats.print_summary(active_key, counter.count)
                break

            elif key_press == ord("r"):
                counter.reset()
                prev_rep_count = 0
                print(f"  Reset: {active_config['display_name']}")

            elif ord("1") <= key_press <= ord("6"):
                new_key = EXERCISE_KEYS[key_press - ord("1")]
                if new_key != active_key:
                    stats.bank(active_key, counter.count)
                    active_key    = new_key
                    active_config = EXERCISES[active_key]
                    active_number = key_press - ord("1") + 1
                    active_joint  = active_config["landmarks"][1]
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
