"""Scripted-teacher recorder for Model Wars imitation learning.

Drives the car with the RLBot python-example scripted teacher and, on every
processed callback, writes the exact 13-D student observation (built through the
same `live_adapter` path the shipped runtime uses) together with the teacher's
raw 8-channel controls.

This file lives OUTSIDE src/ on purpose. Training-only code must never enter the
validated runtime sandbox (src/ bans socket/subprocess/requests/urllib/eval/exec),
and the recorder never trains and never loads the student checkpoint.

Output: one JSONL file per process under training/datasets/raw/.
One RLBot process == one match, so each file is one chronological match stream.
Feed these files to training/build_dataset.py.

Run it through the RLBot GUI by adding training/bot.toml as a bot (the GUI only
loads configs named bot.toml or script.toml).
"""
from pathlib import Path
import json
import os
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reference" / "python_example" / "src"))

from rlbot import flat  # noqa: E402
from rlbot.managers import Bot  # noqa: E402
from observation_contract import live_adapter  # noqa: E402
from prediction import find_slice_at_time  # noqa: E402
from runtime import CHANNELS  # noqa: E402
from util.drive import steer_toward_target  # noqa: E402
from util.sequence import ControlStep, Sequence  # noqa: E402
from util.vec import Vec3  # noqa: E402

AGENT_ID = "aviator/teacher-recorder"
NEUTRAL = [0.0, 0.0, 0.0, 0.0, 0.0, False, False, False]


class TeacherRecorder(Bot):
    def __init__(self, teacher_mode):
        self.teacher_mode = teacher_mode
        self.previous_elapsed = None
        self.previous_controls = None
        self.active_sequence = None
        self.flip_cooldown = 0.0
        self.callbacks = 0
        self.predictions = 0
        self.raw_dir = ROOT / "training" / "datasets" / "raw"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.match_id = f"{stamp}_{os.getpid()}"
        self.file = None
        self.path = None
        super().__init__(AGENT_ID)

    def initialize(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.raw_dir / f"{self.match_id}.jsonl"
        self.file = self.path.open("w", encoding="utf-8")
        print(f"Recorder ready ({self.teacher_mode} teacher) -> {self.path}", flush=True)

    # --- exact student observation, mirroring Runtime.observe -------------
    def observation(self, packet):
        T = float(packet.match_info.seconds_elapsed)
        prediction = self.ball_prediction
        slices = prediction.slices if prediction is not None else []
        selected = find_slice_at_time(prediction, T + 2.0) if slices else None
        previous = self.previous_controls
        prior_mode = 0 if previous is None else (
            3 if previous[5] and previous[2] == -1 else
            2 if previous[5] else
            1 if previous[0] == 1 else 0)
        ctx = dict(previous_elapsed=self.previous_elapsed,
                   previous_mode=prior_mode,
                   previous_steer=0.0 if previous is None else float(previous[1]),
                   prediction_valid=selected is not None,
                   first_time=(slices[0].game_seconds if slices else None))
        return list(live_adapter(packet, self.index, ctx, selected))[:13], T

    # --- scripted teacher (mirrors reference/python_example/src/bot.py) ----
    @staticmethod
    def channels(controls):
        return [getattr(controls, k) for k in CHANNELS]

    def teacher(self, packet):
        if self.active_sequence is not None and not self.active_sequence.done:
            return self.channels(self.active_sequence.tick(packet))
        if len(packet.balls) == 0:
            return list(NEUTRAL)
        car = packet.players[self.index]
        car_loc = Vec3(car.physics.location)
        car_vel = Vec3(car.physics.velocity)
        ball_loc = Vec3(packet.balls[0].physics.location)
        if self.teacher_mode == "dense":
            dense = self._dense(packet, car_loc, ball_loc, car_vel)
            if dense is not None:
                return self.channels(dense)
        if 750 < car_vel.length() < 800:
            return self.channels(self._front_flip(packet))
        target = ball_loc
        if (car_loc.dist(ball_loc) > 1500 and self.ball_prediction is not None
                and self.ball_prediction.slices):
            future = find_slice_at_time(self.ball_prediction, packet.match_info.seconds_elapsed + 2.0)
            if future is not None:
                target = Vec3(future.physics.location)
        controls = flat.ControllerState()
        controls.steer = steer_toward_target(car, target)
        controls.throttle = 1.0
        return self.channels(controls)

    def _dense(self, packet, car_loc, ball_loc, car_vel):
        """Optional augmented heuristic: extra jump/flip events for label coverage."""
        now = float(packet.match_info.seconds_elapsed)
        if now < self.flip_cooldown:
            return None
        grounded = car_loc.z < 120
        dist = car_loc.dist(ball_loc)
        if grounded and dist < 900 and car_vel.length() > 700:
            self.flip_cooldown = now + 1.5
            return self._front_flip(packet)
        if grounded and dist < 800 and ball_loc.z > car_loc.z + 250:
            self.flip_cooldown = now + 1.0
            return self._jump(packet)
        return None

    def _front_flip(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=flat.ControllerState(jump=True)),
            ControlStep(duration=0.05, controls=flat.ControllerState(jump=False)),
            ControlStep(duration=0.2, controls=flat.ControllerState(jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=flat.ControllerState()),
        ])
        return self.active_sequence.tick(packet)

    def _jump(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=flat.ControllerState(jump=True)),
            ControlStep(duration=0.05, controls=flat.ControllerState(jump=False)),
            ControlStep(duration=0.8, controls=flat.ControllerState()),
        ])
        return self.active_sequence.tick(packet)

    def _handle_ball_prediction(self, prediction):
        self.predictions += 1
        super()._handle_ball_prediction(prediction)

    def get_output(self, packet):
        self.callbacks += 1
        try:
            # Mirror the student: no processed output until a prediction has been
            # delivered at least once (src/bot.py raises in that state).
            if self.predictions == 0:
                return flat.ControllerState(*self.teacher(packet))
            obs, T = self.observation(packet)
            controls = self.teacher(packet)
            row = dict(v=1, match_id=self.match_id, callback=self.callbacks,
                       frame=int(packet.match_info.frame_num), elapsed=T,
                       index=int(self.index), team=int(self.team),
                       obs=obs, controls=controls)
            self.file.write(json.dumps(row, allow_nan=False) + "\n")
            self.previous_elapsed = T
            self.previous_controls = controls
            return flat.ControllerState(*controls)
        except Exception as error:
            print(f"recorder callback error: {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
            raise

    def retire(self):
        if self.file is not None:
            self.file.flush()
            self.file.close()
        print(f"Recorder finished: {self.callbacks} callbacks -> {self.path}", flush=True)
        super().retire()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=("reference", "dense"), default="reference",
                        help="reference = exact python-example teacher; dense = reference plus extra flip/jump events")
    cli = parser.parse_args()
    TeacherRecorder(cli.teacher).run(wants_ball_predictions=True, wants_match_communications=False)
