class RepCounter:
    """
    Counts reps for any exercise by tracking a joint angle through two stages.

    Supports two movement types, auto-detected from the threshold values:
      Descending  (up_angle < down_angle) — e.g. bicep curl: starts extended, curls down
      Ascending   (up_angle > down_angle) — e.g. squat: starts low, extends up

    A rep is recorded on the down→up transition only if good_form is True.
    """

    def __init__(self):
        self.count = 0
        self.stage = None    # None | "down" | "up"

    def reset(self):
        self.count = 0
        self.stage = None

    def update(self, angle, down_angle, up_angle, good_form=True):
        """
        Feed the current joint angle; returns the updated rep count.

        good_form=False lets the stage advance normally but skips the count
        increment, silently discarding reps performed with bad form.
        """
        if up_angle < down_angle:
            # Descending: "down" is the low-angle end (e.g. arm curled)
            if angle > down_angle:
                self.stage = "down"
            if angle < up_angle and self.stage == "down":
                self.stage = "up"
                if good_form:
                    self.count += 1
        else:
            # Ascending: "down" is the low-angle end (e.g. squat position)
            if angle < down_angle:
                self.stage = "down"
            if angle > up_angle and self.stage == "down":
                self.stage = "up"
                if good_form:
                    self.count += 1

        return self.count
