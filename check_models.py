import os
import datetime
import glob

print("SEARCHING FOR MODELS...")
files = glob.glob("**/*.pth", recursive=True)
for f in files:
    mtime = os.path.getmtime(f)
    dt = datetime.datetime.fromtimestamp(mtime)
    print(f"File: {os.path.abspath(f)}")
    print(f"Time: {dt}")
    print("-" * 20)
