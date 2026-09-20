"""
src/agent/tools/system_executor.py — Sandboxed system command executor.
"""

import subprocess
import shlex
import os

ALLOWED_COMMANDS = {
    "date", "uptime", "ls", "dir", "touch", "uname", "ipconfig", "ifconfig", "pwd", "whoami"
}

def is_command_allowed(cmd_string: str) -> bool:
    """Check if the base command is in the allowlist."""
    if not cmd_string:
        return False
    parts = shlex.split(cmd_string)
    if not parts:
        return False
    base_cmd = parts[0].lower()
    return base_cmd in ALLOWED_COMMANDS

def execute_command(cmd_string: str, cwd: str = ".") -> str:
    """Execute a system command and return the output."""
    if not is_command_allowed(cmd_string):
        return f"Error: Command '{cmd_string}' is not allowed for security reasons."
        
    try:
        # We use shell=False and pass the split arguments for safety
        parts = shlex.split(cmd_string)
        
        # 'dir' is a shell builtin on Windows, on Linux we can map it to 'ls'
        if parts[0] == "dir" and os.name != "nt":
            parts[0] = "ls"
            
        result = subprocess.run(
            parts,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10
        )
        output = result.stdout.strip()
        if not output:
            output = "[Command executed successfully with no output]"
        return output
    except Exception as e:
        return f"Error executing command: {str(e)}"
