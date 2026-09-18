import subprocess

def get_volume():
    """Reads the current system volume (0-100)."""
    result = subprocess.run(
        ["osascript", "-e", "output volume of (get volume settings)"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())

def set_volume(level):
    # level is 0-100
    level = max(0, min(100, level))  # clamp defensively, never trust the caller blindly
    subprocess.run(["osascript", "-e", f"set volume output volume {level}"])

def get_brightness():
    """Reads the current brightness (0.0-1.0) of the main display."""
    result = subprocess.run(["brightness"], capture_output=True, text=True)
    return float(result.stdout.strip())

def set_brightness(level):
    # level is 0.0-1.0
    level = max(0.0, min(1.0, level))
    subprocess.run(["brightness", str(level)])