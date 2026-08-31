"""Safe local tools for AGENT 0.4."""
from datetime import datetime
import platform
import subprocess
import webbrowser
from urllib.parse import quote_plus
from memory import remember, recall, all_memory


def get_time():
    return datetime.now().astimezone().strftime("%A, %B %d, %Y at %I:%M %p %Z")


def open_website(url="https://www.google.com"):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opening {url}."


def open_application(name):
    key = name.strip().lower()
    commands = {
        "chrome": {"Windows": ["cmd", "/c", "start", "", "chrome"], "Darwin": ["open", "-a", "Google Chrome"], "Linux": ["google-chrome"]},
        "vscode": {"Windows": ["code"], "Darwin": ["open", "-a", "Visual Studio Code"], "Linux": ["code"]},
    }
    if key not in commands:
        return f"I can't open '{name}'. It is not an approved application."
    try:
        subprocess.Popen(commands[key][platform.system()])
        return f"Opening {name}."
    except (OSError, FileNotFoundError):
        return f"I couldn't open {name}. Check that it is installed and available."


def search_web(query):
    webbrowser.open("https://www.google.com/search?q=" + quote_plus(query))
    return f"Searching the web for {query}."


def system_info():
    return f"You are running {platform.system()} {platform.release()} on {platform.machine()}."


def choose_tool(text):
    """Rule-based intent + parameter extraction. Keeps risky actions allow-listed."""
    t = text.lower().strip()

    if any(x in t for x in ["what time", "current time", "time is it", "clock"]):
        return get_time, ()

    if t.startswith(("remember that ", "remember ")):
        raw = t.replace("remember that ", "", 1).replace("remember ", "", 1)
        if " is " in raw:
            key, value = raw.split(" is ", 1)
            return remember, (key.strip(), value.strip())
        return None

    if "what do you remember" in t or t == "memory":
        return lambda: str(all_memory()) if all_memory() else "I don't have any saved memories yet.", ()

    if t.startswith("what is my ") or t.startswith("what's my "):
        key = t.replace("what is my ", "", 1).replace("what's my ", "", 1).strip(" ?")
        value = recall(key)
        return (lambda v=value, k=key: f"You told me your {k} is {v}." if v is not None else f"I don't have a memory for your {k} yet."), ()

    if "open chrome" in t or "open google chrome" in t:
        return open_application, ("chrome",)
    if "open vscode" in t or "open vs code" in t or "open visual studio code" in t:
        return open_application, ("vscode",)

    if "search for " in t or "search the web for " in t or t.startswith("google "):
        query = t.split("search the web for ", 1)[-1] if "search the web for " in t else t.split("search for ", 1)[-1] if "search for " in t else t[7:]
        return search_web, (query.strip(),)

    if t in {"open the web", "open web", "open browser", "open google"}:
        return open_website, ()

    if "system info" in t or "what computer" in t:
        return system_info, ()

    return None
