
class FormChecker:
    """
    Dispatches per-exercise form checks via getattr, returning (feedback, good_form).

    Adding a new exercise only requires a new _check_<exercise_name> method —
    no changes to check() or any caller.

    good_form=True  → display in green, rep allowed to count
    good_form=False → display in orange, rep blocked
    """

    def check(self, exercise_name, landmarks, angle, stage):
        method = getattr(self, f"_check_{exercise_name}", self._check_default)
        return method(landmarks, angle, stage)

    # ── Per-exercise checks ───────────────────────────────────────────────────

    def _check_bicep_curl(self, landmarks, angle, stage):
        """Detect elbow flaring by comparing the horizontal span between both
        elbows (landmarks 13–14) against the span between both shoulders
        (landmarks 11–12). This ratio is body-relative and stays constant
        throughout the curl, so it reliably flags outward drift without being
        fooled by arm position.

        Skipped when angle < 60° (peak of the curl): at that point the forearms
        are nearly vertical and both elbows sit geometrically wider than the
        shoulders even with perfect form, making the span comparison unreliable.
        This is a known geometric limitation; a depth camera would remove it.

        This check is stateless — no per-rep memory is kept. The form indicator
        reflects the current frame only."""
        if angle < 60:
            return ("Good form!", True)
        shoulder_width = abs(landmarks[11].x - landmarks[12].x)
        elbow_width    = abs(landmarks[13].x - landmarks[14].x)
        if elbow_width > shoulder_width + 0.05:
            return ("Keep elbows in", False)
        return ("Good form!", True)

    def _check_squat(self, landmarks, angle, stage):
        """Depth is proven the moment stage reaches 'down' (angle crossed the
        threshold). Return True for the entire return phase so the rep can count."""
        if stage == "down":
            return ("Good depth!", True)
        return ("Lower into squat", True)

    def _check_pushup(self, landmarks, angle, stage):
        """Require elbow angle < 90° at the bottom for a full-depth pushup."""
        if stage == "down":
            if angle > 90:
                return ("Go lower", False)
            return ("Good depth!", True)
        return ("Lower your chest", True)

    def _check_shoulder_press(self, landmarks, angle, stage):
        """Warn only when BOTH elbows are below shoulder height. The non-pressing
        arm may rest lower naturally, so 'and' avoids one-sided false positives.
        MediaPipe y=0 is top-of-frame; larger y = lower on screen."""
        left_too_low  = landmarks[13].y > landmarks[11].y + 0.1
        right_too_low = landmarks[14].y > landmarks[12].y + 0.1
        if stage == "down" and left_too_low and right_too_low:
            return ("Raise elbows to shoulder height", False)
        return ("Good form!", True)

    def _check_lunge(self, landmarks, angle, stage):
        """Same depth-proven logic as squat."""
        if stage == "down":
            return ("Good depth!", True)
        return ("Step forward and lower", True)

    def _check_lateral_raise(self, landmarks, angle, stage):
        """Three-zone check: above 100° strains the shoulder, below 65° is partial."""
        if angle > 100:
            return ("Too high — stop at shoulder level", False)
        if angle > 65:
            return ("Good height!", True)
        if angle > 30:
            return ("Raise higher", False)
        return ("Lift your arm", False)

    def _check_default(self, landmarks, angle, stage):
        return ("Keep going!", True)
