"""Safe local tools for AGENT 0.3.

Tools are deliberately allow-listed. AGENT never executes arbitrary shell
commands from natural-language input.
"""
from datetime import datetime
import os
import platform
import subprocess
import webbrowser


def get_time() -> str:
    return datetime.now().astimezone().strftime("%A, %B %d, %Y at %I:%M %p %Z")


def open_website(url: str = "https://www.google.com") -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opening {url}."


def open_application(name: str) -> str:
    """Open only explicitly allow-listed applications."""
    key = name.strip().lower()
    commands = {
        "chrome": {
            "Windows": ["cmd", "/c", "start", "", "chrome"],
            "Darwin": ["open", "-a", "Google Chrome"],
            "Linux": ["google-chrome"],
        },
        "google chrome": {
            "Windows": ["cmd", "/c", "start", "", "chrome"],
            "Darwin": ["open", "-a", "Google Chrome"],
            "Linux": ["google-chrome"],
        },
        "vscode": {
            "Windows": ["code"],
            "Darwin": ["open", "-a", "Visual Studio Code"],
            "Linux": ["code"],
        },
        "visual studio code": {
            "Windows": ["code"],
            "Darwin": ["open", "-a", "Visual Studio Code"],
            "Linux": ["code"],
        },
    }
    if key not in commands:
        return f"I can't open '{name}'. It is not an approved application."
    try:
        subprocess.Popen(commands[key][platform.system()])
        return f"Opening {name}."
    except (OSError, FileNotFoundError):
        return f"I couldn't open {name}. Please check that it is installed and available on your system."


def system_info() -> str:
    return f"You are running {platform.system()} {platform.release()} on {platform.machine()}."


def choose_tool(text: str):
    """Simple deterministic intent routing; no API or external model required."""
    t = text.lower().strip()
    if any(x in t for x in ["what time", "current time", "time is it", "clock"]):
        return get_time, ()
    if "open chrome" in t or "open google chrome" in t:
        return open_application, ("chrome",)
    if "open vscode" in t or "open vs code" in t or "open visual studio code" in t:
        return open_application, ("vscode",)
    if t in {"open the web", "open web", "open browser", "open google"}:
        return open_website, ()
    if "system info" in t or "what computer" in t:
        return system_info, ()
    return None
