"""Phase 3 DAgger recorder: the STUDENT drives, the scripted teacher labels.

Unlike record_teacher.py (teacher drives), here the shipped student policy
(`model/V13.pt`) drives the car, so every logged observation is a state the
student actually visits and the history features (previous_mode / previous_steer
/ previous_elapsed) come from the student's *own* previous controls -- exactly the
distribution seen at live inference. The scripted teacher's controls are logged as
the label for that state.

Why this matters: the teacher-forced dataset stores the teacher's previous
controls, but the live runtime feeds the student its own previous controls. That
train/live mismatch on the history columns is one source of the "random jumps"
the sealed model shows in the GUI. DAgger removes it by construction.

Output schema is identical to record_teacher.py, so training/build_dataset.py
consumes it directly. Keep these matches in the TRAIN split only (val/test stay
the fixed teacher-v3 splits, so numbers remain comparable).

Training-only code: lives outside src/ on purpose, never imported by the runtime.
"""
from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "reference" / "python_example" / "src"))

from rlbot import flat  # noqa: E402
from policy import Policy  # noqa: E402
from runtime import decode  # noqa: E402
from record_teacher import TeacherRecorder  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "model" / "V13.pt"


class DaggerRecorder(TeacherRecorder):
    AGENT_ID = "aviator/dagger-recorder"

    def __init__(self, teacher_mode, checkpoint):
        super().__init__(teacher_mode)
        self.checkpoint = Path(checkpoint).resolve()
        # Keep DAgger data separate from the frozen teacher recordings so the
        # teacher-v3 dataset stays byte-identical.
        self.raw_dir = ROOT / "training" / "datasets" / "raw_dagger"
        self.model = None
        self.hidden = None

    def initialize(self):
        super().initialize()
        torch.set_num_threads(1)
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.model = Policy().eval()
        self.model.load_state_dict(saved["state_dict"], strict=True)
        if sum(p.numel() for p in self.model.parameters()) != 26181:
            raise RuntimeError("student checkpoint architecture mismatch")
        self.hidden = None
        print(f"DAgger recorder ready ({self.teacher_mode} teacher, "
              f"student={self.checkpoint}) -> {self.path}", flush=True)

    def student(self, obs):
        """Exactly src/runtime.Runtime.infer, minus the sandbox guards."""
        x = torch.tensor([obs], dtype=torch.float32)
        with torch.inference_mode():
            logits, steering, hidden = self.model(x, self.hidden)
        self.hidden = hidden
        mode = int(logits.argmax(1).item())
        return decode(mode, float(steering.item())), mode

    def get_output(self, packet):
        self.callbacks += 1
        try:
            # Mirror src/bot.py: the live student cannot act until a
            # BallPrediction has been delivered, so hold neutral (no log row).
            if self.predictions == 0:
                return flat.ControllerState()
            # observation() uses self.previous_controls, which below holds the
            # student's own last action -> the live inference distribution.
            obs, T = self.observation(packet)
            student_controls, student_mode = self.student(obs)
            teacher_controls = self.teacher(packet)  # DAgger label
            row = dict(v=1, match_id=self.match_id, callback=self.callbacks,
                       frame=int(packet.match_info.frame_num), elapsed=T,
                       index=int(self.index), team=int(self.team),
                       obs=obs, controls=teacher_controls,
                       student_mode=student_mode, student_controls=student_controls)
            self.file.write(json.dumps(row, allow_nan=False) + "\n")
            self.previous_elapsed = T
            self.previous_controls = student_controls
            return flat.ControllerState(*student_controls)
        except Exception as error:
            print(f"dagger callback error: {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=("reference", "smart"), default="smart",
                        help="label policy; 'smart' matches the teacher-v3 dataset")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="student checkpoint that drives the car (default model/V13.pt)")
    cli = parser.parse_args()
    DaggerRecorder(cli.teacher, cli.checkpoint).run(
        wants_ball_predictions=True, wants_match_communications=False)
