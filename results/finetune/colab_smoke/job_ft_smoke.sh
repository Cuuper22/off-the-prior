# Colab job: run the code cells of notebooks/finetune_colab.ipynb in TINY mode on a GPU runtime.
set -e
export OTP_DEVICE=cuda OTP_THREADS=2
nvidia-smi --query-gpu=name --format=csv,noheader
python3 - <<'PY'
import json
nb = json.load(open("notebooks/finetune_colab.ipynb"))
code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
code = code.replace("TINY = False", "TINY = True").replace("SKIP_INSTALL = False", "SKIP_INSTALL = True")
open("/content/nb_cells.py", "w").write(code)
PY
mkdir -p /content/nbrun && ln -sfn /content/otp /content/nbrun/off-the-prior
cd /content/nbrun
python3 /content/nb_cells.py
mkdir -p /content/otp/results/colab_out/ft_smoke
cp /content/otp/results/finetune/colab/run1_tiny/{costs,summary}.csv /content/otp/results/finetune/colab/run1_tiny/{provenance,colab_meta}.json /content/otp/results/colab_out/ft_smoke/
