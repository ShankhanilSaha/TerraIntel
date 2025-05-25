import subprocess
import sys
import os

def get_venv_python():
    # Update this path to your venv folder
    venv_path = "venv"

    if sys.platform == "win32":
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_path, "bin", "python")

    if not os.path.exists(python_path):
        print(f"ERROR: Python interpreter not found in virtual environment at {python_path}")
        sys.exit(1)

    return python_path

def run_script(python_interpreter, script_name):
    print(f"\nRunning {script_name} with {python_interpreter} ...\n")

    process = subprocess.Popen(
        [python_interpreter, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",      # <-- force utf-8 decoding
        errors="replace",      # <-- replace undecodable chars with �
        bufsize=1,
        universal_newlines=True,
    )

    for line in process.stdout:
        print(line, end="")

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        print(f"\nError: {script_name} exited with code {return_code}. Stopping.")
        sys.exit(return_code)

def main():
    python_interpreter = get_venv_python()

    scripts = [
        "gee_map_call.py",
        "TerrainAnalyser.py",
        "analysis.py",
        "main.py"
    ]

    for script in scripts:
        run_script(python_interpreter, script)

    print("\nAll scripts finished successfully.")

if __name__ == "__main__":
    main()
