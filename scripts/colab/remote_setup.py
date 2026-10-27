# Runs on the Colab kernel (sent with `colab exec -f`). Unpacks the code bundle and installs deps.
import os
import subprocess
import sys

os.chdir("/content")
subprocess.run(["rm", "-rf", "/content/otp"], check=True)
os.makedirs("/content/otp", exist_ok=True)
subprocess.run(["tar", "-xzf", "/content/otp_bundle.tar.gz", "-C", "/content/otp"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/mistralai/timestral.git",
                "statsmodels", "cma"], check=True)
import torch  # noqa: E402

print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True,
                     text=True).stdout)
