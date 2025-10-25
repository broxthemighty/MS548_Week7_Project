# image_generator.py
import os
import subprocess
import re
import sys
from PIL import Image
from datetime import datetime

# path to compiled stable-diffusion.cpp folder
SD_CPP_PATH = r"C:\Users\Matt-Beast\source\repos\stable-diffusion.cpp\build_cli\bin\Release"
SD_EXE = os.path.join(SD_CPP_PATH, "sd.exe") 

def generate_image(prompt, output_dir="generated_images",
                   steps=20, guidance=7.5, size=(512,512),
                   model_name="stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
                   device="cuda", progress_callback=None):

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir, f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )

    cmd = [
        SD_EXE,
        "--prompt", prompt,
        "--output", out_path,
        "--steps", str(steps),
        "--cfg-scale", str(guidance),
        "--width", str(size[0]),
        "--height", str(size[1]),
        "--model", os.path.join(SD_CPP_PATH, model_name),
        "--seed", "42"
    ]

    print("[INFO] Running:", " ".join(cmd))

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        total_steps = 0
        current_step = 0

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            # print all logs to console
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

            # look for progress indicators like [7/20]
            match = re.search(r"\[(\d+)/(\d+)\]", line)
            if match:
                current_step = int(match.group(1))
                total_steps = int(match.group(2))
                percent = int((current_step / total_steps) * 100)
                if progress_callback:
                    progress_callback(percent)

        process.wait()

        if process.returncode == 0 and os.path.exists(out_path):
            print(f"[INFO] Image saved: {out_path}")
            if progress_callback:
                progress_callback(100)
            return out_path
        else:
            raise RuntimeError("Stable Diffusion failed to produce output.")

    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        fail_path = os.path.join(output_dir, "generation_failed.png")
        Image.new("RGB", (512, 512), (255, 0, 0)).save(fail_path)
        if progress_callback:
            progress_callback(0)
        return fail_path