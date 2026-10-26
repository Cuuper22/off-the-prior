# Runs on the Colab kernel. Starts /content/otp/job.sh detached so `colab exec` returns at once.
import subprocess

subprocess.Popen(["bash", "-c", "cd /content/otp && nohup bash job.sh > /content/job.log 2>&1; echo $? > /content/job.exit"],
                 start_new_session=True)
print("started")
