"""Drop-in CLI to install the SMPL-X neutral body model for 4DAnyone.

The WebUI does this for you from the Settings tab; this script is useful for
pre-seeding the models volume at container build / CI time:

    python scripts/provision_smplx.py
"""

from __future__ import annotations

import sys

from app.init_models import provision_smplx

if __name__ == "__main__":
    installed = provision_smplx()
    print(f"installed: {installed}")
    sys.exit(0)
