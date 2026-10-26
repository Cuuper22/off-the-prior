# Runs on the Colab kernel. Prints the job state and the last log lines.
import os

done = os.path.exists("/content/job.exit")
print("STATE", "done" if done else "running", open("/content/job.exit").read().strip() if done else "")
if os.path.exists("/content/job.log"):
    lines = [ln for ln in open("/content/job.log", errors="ignore").read().splitlines()
             if "Fetching" not in ln and "Warning" not in ln and "warn" not in ln]
    print("\n".join(lines[-8:]))
