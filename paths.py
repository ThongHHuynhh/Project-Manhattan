"""Where the K2 cutting work finds things outside this folder (override with environment variables)."""
import os

K2_DIR = os.path.expanduser(os.environ.get("K2_DIR", "~/R-D-Projects/K2-Isaac-Sim"))
USD_DIR = os.path.join(K2_DIR, "USD")  # K2-CV-STEP-005-SDG.usd, K2-CV-STEP-006-CUT.usd
DOUGHSDG_DIR = os.environ.get("DOUGHSDG_DIR", os.path.join(K2_DIR, "StandaloneScripts", "DoughSDG"))  # dough generator
ROBOT_USD = os.path.expanduser(os.environ.get(
    "K2_ROBOT_USD", "~/sandbox-isaac/ABB-IRB-1200/abb_irb1200_support/urdf/irb1200/irb1200.usd"))
