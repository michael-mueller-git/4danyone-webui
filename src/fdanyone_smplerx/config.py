"""Configuration constants mirroring SMPLer-X ``config_smpler_x_h32.py``."""

from __future__ import annotations

# model
FEAT_DIM = 1280  # ViT-H embed dim
TASK_TOKENS_NUM = 31  # 1 shape + 1 cam + 1 expr + 1 jaw + 2 hand + 25 body
DEPTH = 32
NUM_HEADS = 16
MLP_RATIO = 4
DROP_PATH_RATE = 0.55

# input / output shapes  (H, W) / (D, H, W)
INPUT_IMG_SHAPE = (512, 384)  # crop patch (H, W)
INPUT_BODY_SHAPE = (256, 192)  # model input (H, W)
OUTPUT_HM_SHAPE = (16, 16, 12)  # (D, H, W) joint volume

# virtual camera
FOCAL = (5000.0, 5000.0)
PRINCPT = (INPUT_BODY_SHAPE[1] / 2.0, INPUT_BODY_SHAPE[0] / 2.0)  # (96, 128)
BODY_3D_SIZE = 2.0
CAMERA_3D_SIZE = 2.5

# joint bookkeeping (from common/utils/human_models.py SMPLX)
BODY_POS_JN = 25  # len(pos_joint_part['body']) -> PositionNet + body tokens
BODY_ORIG_JN = 22  # len(orig_joint_part['body']) -> body_pose = (22 - 1) * 6 = 126

# checkpoint
CHECKPOINT_REPO = "caizhongang/SMPLer-X"
CHECKPOINT_FILE = "smpler_x_h32_correct.pth.tar"

# pinned SMPLer-X source revision; doubles as the MotionResult revision (40 hex)
SMPLERX_REVISION = "064baef0e4ab5277a3297691bc1d46ea5412586f"
