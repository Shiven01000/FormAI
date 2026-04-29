# FormAI — Architecture

Technical reference for the design decisions behind FormAI.

---

## Module Responsibilities

```
main.py          — orchestration, UI, audio, keyboard handling
config.py        — exercise registry (data-driven, zero new code per exercise)
angles.py        — pure math, no framework dependencies
rep_counter.py   — state machine, exercise-agnostic
form_checker.py  — per-exercise rules, dispatched via getattr
```

---

## Angle Calculation (`angles.py`)

Joint angles are computed with `numpy.arctan2` rather than the dot-product / `arccos` approach:

```python
angle = arctan2(c.y - b.y, c.x - b.x) - arctan2(a.y - b.y, a.x - b.x)
```

**Why arctan2:**
- Handles all four quadrants without ambiguity (unlike `arctan`)
- Avoids the numerical instability of `arccos` near 0° and 180°, where floating-point errors in the dot product can produce values slightly outside `[-1, 1]` and raise exceptions
- Input is MediaPipe's normalized coordinates (0.0–1.0), so no pixel conversion is needed at this stage

The raw result is folded into `[0, 180]` via `abs()` + a `360 - angle` correction, making it consistent regardless of which side of the camera the user faces.

---

## Rep Counting State Machine (`rep_counter.py`)

Two exercise types share one class, auto-detected from the threshold relationship:

```
up_angle < down_angle  →  DESCENDING  (e.g. bicep curl: starts extended, rep at curl)
up_angle > down_angle  →  ASCENDING   (e.g. squat: starts low, rep at extension)
```

State transitions:

```
ASCENDING example (squat):

   Standing (170°)                 Squatting (80°)
        │                               │
   angle < down_angle (90°)       angle > up_angle (160°)
        │                               │
   stage = "down"              stage = "up", count += 1
                                     (if good_form)
```

The `good_form` gate is evaluated at the moment of the `down→up` transition, not on every frame. This means bad-form reps advance the stage normally (preventing the counter from getting stuck) but don't increment the count.

---

## Form Checker Dispatch (`form_checker.py`)

Per-exercise rules are registered as methods named `_check_<exercise_key>`:

```python
method = getattr(self, f"_check_{exercise_name}", self._check_default)
return method(landmarks, angle, stage)
```

This means adding a new exercise requires only:
1. A new entry in `EXERCISES` (config.py)
2. A new `_check_<name>` method (form_checker.py)

No changes to `check()`, `main.py`, or `rep_counter.py`.

---

## Bilateral Limb Tracking

For exercises tagged `"bilateral": True`, both arms are measured and the more-active arm drives the rep count:

```python
# Whichever arm deviates more from fully straight (180°) is considered active
if abs(angle_left - 180) >= abs(angle_right - 180):
    current_angle = angle_left
else:
    current_angle = angle_right
```

This works because at rest a limb hangs near 180°; the arm actively curling/pressing will always have the lower angle. No manual arm selection is required.

---

## Coordinate System

MediaPipe returns `NormalizedLandmark` with `x`, `y` in `[0.0, 1.0]`:
- `(0, 0)` = top-left of frame
- `(1, 1)` = bottom-right of frame
- **y increases downward** — so `landmark.y > shoulder.y` means *below* the shoulder on screen

The angle calculator operates directly on normalized coordinates. Pixel conversion (`int(lm.x * width)`) is only needed for rendering (`draw_skeleton`).

---

## Glassmorphism UI (`draw_panel`)

OpenCV has no rounded-rectangle primitive. The panel shape is built from:
- Two filled rectangles (horizontal and vertical strips forming a cross)
- Four filled circles plugging the corners

This is drawn onto an `overlay` copy of the frame, then blended back:

```python
cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
```

`alpha ≈ 0.72–0.92` gives the semi-transparent "glass" look without fully obscuring the skeleton behind the panel.

---

## Non-Blocking Audio

Two mechanisms run concurrently with the video loop without blocking it:

| Mechanism | Implementation | Use case |
|-----------|---------------|----------|
| Beep | `sounddevice.play()` — starts a background audio thread | Rep confirmation click |
| TTS | `subprocess.Popen(['say', ...])` — spawns OS process | Form cues, milestone announcements |

`Popen.poll()` returns `None` while the process is running, enabling `say_feedback()` to check whether speech is already in progress before starting a new utterance.

---

## Session Stats Flow

```
exercise switch or Q pressed
        │
  stats.bank(key, counter.count)   ← freeze live count into exercise_totals
        │
  counter.reset()                  ← live counter back to 0
        │
  stats.total_reps(live_count)     ← sum(banked) + live   (shown in HUD)
```

This keeps the HUD total accurate mid-session without permanently losing the switched-away count.

---

## Bilateral Tracking — Design Decisions

### Why elbow-to-hip distance was rejected

The first approach measured how far each elbow had drifted laterally from its resting position (hip). Debug output showed this metric was tracking arm *position*, not form quality:

```
Good form (arms straight):  drift = 0.15 – 0.21
Actively curling:            drift = 0.04 – 0.12
```

The numbers inverted. A straight resting arm has a large elbow-to-hip separation; a curled arm brings the elbow close to the hip. No threshold could distinguish bad form from a normal rep.

### Final approach — elbow span vs shoulder span

Comparing the horizontal span between both elbows (`landmarks[13].x – landmarks[14].x`) against the span between both shoulders (`landmarks[11].x – landmarks[12].x`) is body-relative: the ratio stays stable throughout the curl and only widens when the elbows genuinely flare outward.

```python
if elbow_width > shoulder_width + 0.05:
    return ("Keep elbows in", False)
```

The 0.05 tolerance absorbs natural asymmetry without letting real flaring through.

### Peak exemption (< 60°)

At the peak of a curl the forearms are nearly vertical. Geometrically, both elbows sit slightly wider than the shoulders at this angle even with perfect form — the check would fire false positives on every rep. The exemption is a documented tradeoff, not a bug:

```python
if angle < 60:
    return ("Good form!", True)
```

A depth camera (z-coordinate per landmark) would allow accurate shoulder-width comparison in this position and remove the need for the exemption.

### Approaches explored and removed

**25 °/frame noise clamp** — `angle = prev if abs(raw - prev) > 25 else raw` — was added to reject single-frame MediaPipe spikes. It caused a hard failure: when the arm moved faster than 25 °/frame the new value was permanently rejected and `prev` never updated, freezing the displayed angle at ~170 °. Stage never reached "down", reps stopped counting entirely. Removed in favour of accepting raw MediaPipe output, which is stable enough in practice.

**Sticky `_rep_bad` flag** — a flag carried bad-form state across the peak so the skeleton stayed orange for the full rep even if the peak check returned `True`. The reset condition (`stage == "up" and angle > 140`) fired on every frame of the descent (stage stays "up" from 150 ° all the way down to 50 °), clearing the flag before the rep completed. The angle-guard fix was added, but the combination with the frozen-angle bug created a circular lockout: flag set, angle stuck, reset never fires, always orange, reps blocked indefinitely. Removed in favour of the current stateless check.
