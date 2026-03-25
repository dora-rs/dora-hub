"""Dora node: MoQ chat-to-pick-and-place orchestrator.

Bridges a MoQ web chat to the dora dataflow for voice/text-commanded
pick-and-place.  Free-text commands are parsed by the VLM into structured
JSON, then forwarded to the grasp selector.  Execution only proceeds
after the user confirms with "ok".

State machine: Idle → Parsing → Planning → AwaitingConfirm → Executing
"""

import json
import os
import queue
import random
import time
import re
import sys
import threading

import pyarrow as pa
from dora import Node

# Force unbuffered output so dora logs captures prints immediately
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

CHAT_CHANNEL = os.getenv("CHAT_CHANNEL", "anon/openarm-chat")
CHAT_RELAY = os.getenv("CHAT_RELAY", "https://cdn.1ms.ai")
CHAT_USERNAME = os.getenv("CHAT_USERNAME", "robot")

STATE_IDLE = "idle"
STATE_PARSING = "parsing"
STATE_CHATTING = "chatting"
STATE_PLANNING = "planning"
STATE_AWAITING_CONFIRM = "awaiting_confirm"
STATE_EXECUTING = "executing"

VLM_PARSE_PROMPT = (
    "Extract the action from this robot command. "
    "Output ONLY JSON:\n"
    '- Pick and place: {{"pick": "full object description", "place": "target"}}\n'
    '- Flip: {{"pick": "full object description", "action": "flip"}}\n'
    "- If no place/flip, just: {{\"pick\": \"full object description\"}}\n"
    "IMPORTANT: Keep the FULL object description in 'pick', including location/color/size qualifiers. "
    "These help identify the correct object when there are multiple similar ones.\n"
    "Examples:\n"
    '- "flip the sausage in the pan" → {{"pick": "sausage in the pan", "action": "flip"}}\n'
    '- "pick the red cube on the left" → {{"pick": "red cube on the left"}}\n'
    '- "put the big plate in the sink" → {{"pick": "big plate", "place": "sink"}}\n'
    'If this is NOT a robot command, output exactly: {{"chat": true}}\n'
    "Command: {command}"
)

VLM_CHAT_PROMPT = (
    "You are OpenArm, a friendly open-source robot arm. You can do simple pick and place tasks. "
    "You run on the dora robotics framework and use xoq as backend for real-time communication and video streaming. "
    "Keep your reply short and friendly (1-2 sentences). "
    "If the user wants you to pick or move something, remind them to say: @robot pick the <object> and put it in the <container>\n"
    "User message: {message}"
)


def _chat_recv_thread(chat, msg_queue, stop_event):
    """Background thread: poll chat.recv() and push messages to queue."""
    print("[chat] recv thread started")
    while not stop_event.is_set():
        try:
            msg = chat.recv(timeout=0.5)
            if msg is not None:
                print(f"[chat] recv: {msg.name}: {msg.text}")
                msg_queue.put(msg)
        except Exception as e:
            print(f"[chat] recv error: {e}")
    print("[chat] recv thread stopped")


def _send_vlm_text(node, text):
    """Send a text-only request to dora-qwen-omni."""
    payload = f"<|user|>\n<|im_start|>\n{text}"
    node.send_output("vlm_request", pa.array([payload]))


def _parse_loop_steps(text):
    """Parse loop steps from comma-separated text. Returns list of step strings, or empty list."""
    lower = text.lower()
    if not (lower.startswith("loop:") or lower.startswith("loop\n")):
        return []
    body = text[text.index(":") + 1:] if ":" in text[:6] else text[4:]
    body = body.strip()
    # Split on commas, strip leading "and " from each chunk
    # Handles "a, b, and c" → ["a", "b", "c"]
    # while preserving "and" inside steps like "pick X and put in Y"
    parts = body.split(",")
    steps = []
    for p in parts:
        p = p.strip()
        if p.lower().startswith("and "):
            p = p[4:].strip()
        if p:
            steps.append(p)
    return steps


def _try_parse_json(text):
    """Extract the first JSON object from text, return dict or None."""
    # Strip markdown fences and thinking tags
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    match = re.search(r"\{[^{}]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def main():
    from xoq_chat import Chat

    chat = Chat(CHAT_CHANNEL, relay=CHAT_RELAY, username=CHAT_USERNAME)
    print(f"[chat] Connected to {CHAT_RELAY} channel={CHAT_CHANNEL} as {CHAT_USERNAME}", flush=True)

    msg_queue = queue.Queue()
    stop_event = threading.Event()
    recv_thread = threading.Thread(
        target=_chat_recv_thread, args=(chat, msg_queue, stop_event), daemon=True
    )
    recv_thread.start()

    state = STATE_IDLE
    last_user_message = ""
    last_cmd = None
    retries = 0
    remaining_cycles = 0
    total_cycles = 0
    loop_steps = []
    loop_cmds = []
    loop_index = 0
    loop_parsing_index = 0
    node = Node()
    greeted = False

    for event in node:
        if event["type"] == "INPUT":
            event_id = event["id"]

            # --- Tick: drain chat messages ---
            if event_id == "tick":
                if not greeted:
                    greeted = True
                    # Drain any replayed messages from previous sessions
                    drained = 0
                    while not msg_queue.empty():
                        try:
                            msg_queue.get_nowait()
                            drained += 1
                        except queue.Empty:
                            break
                    if drained:
                        print(f"[chat] Drained {drained} old message(s)")
                    print("[chat] Sending greeting...")
                    chat.send("At your service!")
                    print("[chat] Greeting sent")
                    continue
                while not msg_queue.empty():
                    try:
                        msg = msg_queue.get_nowait()
                    except queue.Empty:
                        break

                    # Ignore our own messages and other robot instances
                    sender = getattr(msg, "name", "") or ""
                    if sender == CHAT_USERNAME:
                        continue
                    if sender.startswith("robot-"):
                        continue

                    text = str(msg.text if hasattr(msg, "text") else msg).strip()
                    if not text:
                        continue

                    text_lower = text.lower()

                    # Only respond to messages mentioning @robot (or cancel/ok commands while active)
                    has_mention = "@robot" in text_lower
                    is_control = text_lower in ("cancel", "abort", "stop", "ok", "help")
                    if not has_mention and not is_control:
                        continue
                    # Strip the @robot mention from the command text
                    if has_mention:
                        text = re.sub(r"@robot\s*", "", text, flags=re.IGNORECASE).strip()
                        text_lower = text.lower()

                    print(f"[chat] Received from {sender}: {text}")

                    # --- Cancel (any state except idle) ---
                    if text_lower in ("cancel", "abort", "stop") and state != STATE_IDLE:
                        print(f"[chat] Cancel requested (was {state})")
                        remaining_cycles = 0
                        total_cycles = 0
                        loop_steps = []
                        loop_cmds = []
                        chat.send("Cancelled.")
                        state = STATE_IDLE
                        continue

                    # --- Help ---
                    if text_lower == "help":
                        chat.send(
                            "Commands:\n"
                            "  @robot pick the <object> — pick up an object\n"
                            "  @robot pick the <object> and put in the <target> — pick and place\n"
                            "  @robot flip the <object> — flip an object in place\n"
                            "  @robot <command> x5 — repeat a command 5 times\n"
                            "  @robot loop: <step1>, <step2>, and <step3> — loop steps forever\n"
                            "  cancel / stop — cancel current action or loop"
                        )
                        continue

                    # --- OK: confirm execution ---
                    if text_lower == "ok" and state == STATE_AWAITING_CONFIRM:
                        print("[chat] User confirmed, sending execute signal")
                        chat.send("Executing...")
                        node.send_output("execute", pa.array(["go"]))
                        state = STATE_EXECUTING
                        continue

                    # --- New command (only in idle) ---
                    if state != STATE_IDLE:
                        chat.send(f"Busy ({state}). Send 'cancel' to abort.")
                        continue

                    # Check for loop command
                    steps = _parse_loop_steps(text)
                    if steps:
                        if len(steps) < 2:
                            chat.send("Loop needs at least 2 steps. Example: loop: pick X, flip Y, and place Z")
                            continue
                        loop_steps = steps
                        loop_cmds = []
                        loop_index = 0
                        loop_parsing_index = 0
                        chat.send(f"Loop with {len(steps)} steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps)))
                        prompt = VLM_PARSE_PROMPT.format(command=loop_steps[0])
                        _send_vlm_text(node, prompt)
                        state = STATE_PARSING
                        continue

                    # Try direct JSON first
                    cmd = _try_parse_json(text)
                    if cmd and "pick" in cmd:
                        count = max(1, cmd.pop("count", 1))
                        remaining_cycles = count
                        total_cycles = count
                        cmd["command_ts"] = time.time()
                        last_cmd = cmd.copy()
                        print(f"[chat] Direct JSON command: {cmd} (×{count})")
                        cycle_suffix = f" (×{count}, cycle 1)" if count > 1 else ""
                        if cmd.get("action") == "flip":
                            chat.send(f"Got it: flip '{cmd.get('pick', '')}'{cycle_suffix}. Planning...")
                        else:
                            chat.send(f"Got it: pick '{cmd.get('pick', '')}'"
                                      + (f", place in '{cmd['place']}'" if cmd.get("place") else "")
                                      + f"{cycle_suffix}. Planning...")
                        node.send_output("command", pa.array([json.dumps(cmd)]))
                        state = STATE_PLANNING
                        continue

                    # Free-text: send to VLM for parsing
                    print(f"[chat] Sending to VLM for parsing: {text}")
                    last_user_message = text
                    prompt = VLM_PARSE_PROMPT.format(command=text)
                    _send_vlm_text(node, prompt)
                    state = STATE_PARSING

            # --- VLM response (command parsing) ---
            elif event_id == "vlm_response" and state == STATE_PARSING:
                text = event["value"][0].as_py()
                print(f"[chat] VLM parse response: {text[:200]}")
                cmd = _try_parse_json(text)

                # Loop mode: accumulate parsed commands
                if loop_steps:
                    if cmd and "pick" in cmd:
                        cmd.pop("count", None)
                        loop_cmds.append(cmd)
                    else:
                        chat.send(f"Step {loop_parsing_index + 1} ('{loop_steps[loop_parsing_index]}') isn't a valid command. Loop cancelled.")
                        loop_steps = []
                        loop_cmds = []
                        state = STATE_IDLE
                        continue

                    loop_parsing_index += 1
                    if loop_parsing_index < len(loop_steps):
                        prompt = VLM_PARSE_PROMPT.format(command=loop_steps[loop_parsing_index])
                        _send_vlm_text(node, prompt)
                        # stay in STATE_PARSING
                    else:
                        chat.send(f"All {len(loop_cmds)} steps parsed. Starting loop...")
                        loop_index = 0
                        cmd = loop_cmds[0].copy()
                        cmd["command_ts"] = time.time()
                        last_cmd = cmd
                        node.send_output("command", pa.array([json.dumps(cmd)]))
                        state = STATE_PLANNING
                    continue

                if cmd and "pick" in cmd:
                    count = max(1, cmd.pop("count", 1))
                    remaining_cycles = count
                    total_cycles = count
                    cmd["command_ts"] = time.time()
                    last_cmd = cmd.copy()
                    cycle_suffix = f" (×{count}, cycle 1)" if count > 1 else ""
                    if cmd.get("action") == "flip":
                        chat.send(f"Understood: flip '{cmd.get('pick', '')}'{cycle_suffix}. Planning trajectory...")
                    else:
                        chat.send(f"Understood: pick '{cmd.get('pick', '')}'"
                                  + (f", place in '{cmd['place']}'" if cmd.get("place") else "")
                                  + f"{cycle_suffix}. Planning trajectory...")
                    node.send_output("command", pa.array([json.dumps(cmd)]))
                    state = STATE_PLANNING
                else:
                    # Not a pick/place command — ask VLM for a natural response
                    print(f"[chat] Not a command, sending to VLM for chat response")
                    prompt = VLM_CHAT_PROMPT.format(message=last_user_message)
                    _send_vlm_text(node, prompt)
                    state = STATE_CHATTING

            # --- VLM response (conversational chat) ---
            elif event_id == "vlm_response" and state == STATE_CHATTING:
                text = event["value"][0].as_py().strip()
                print(f"[chat] VLM chat response: {text[:200]}")
                chat.send(text)
                state = STATE_IDLE

            # --- Selector status (SAM3/VLM progress) ---
            elif event_id == "selector_status":
                text = event["value"][0].as_py()
                print(f"[chat] selector_status: {text} (state={state})")
                if state == STATE_PLANNING:
                    chat.send(text)

            # --- Trajectory status from motion planner ---
            elif event_id == "trajectory_status":
                raw = event["value"][0].as_py()
                try:
                    status = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                s = status.get("status", "")
                print(f"[chat] trajectory_status: {s} (state={state})")
                if s == "ready" and state == STATE_PLANNING:
                    retries = 0
                    wp = status.get("waypoints", "?")
                    dur = status.get("duration", "?")
                    arm = status.get("arm", "?")
                    chat.send(f"Executing: {wp} waypoints, ~{dur}s on {arm} arm.")
                    node.send_output("execute", pa.array(["go"]))
                    state = STATE_EXECUTING
                elif s == "failed":
                    reason = status.get("reason", "")
                    if status.get("retry") and retries < 2 and last_cmd is not None:
                        retries += 1
                        chat.send(f"Retrying ({retries}/2): {reason}")
                        print(f"[chat] Retrying command ({retries}/2): {reason}")
                        last_cmd["command_ts"] = time.time()
                        node.send_output("command", pa.array([json.dumps(last_cmd)]))
                        state = STATE_PLANNING
                    else:
                        retries = 0
                        msg = f"Failed: {reason}" if reason else "Planning failed."
                        if loop_steps:
                            msg += f" (loop stopped at step {loop_index + 1}: '{loop_steps[loop_index]}')"
                            loop_steps = []
                            loop_cmds = []
                        cycle_done = total_cycles - remaining_cycles
                        if total_cycles > 1:
                            msg += f" (stopped at cycle {cycle_done}/{total_cycles})"
                        remaining_cycles = 0
                        total_cycles = 0
                        chat.send(f"{msg} Try a different command.")
                        state = STATE_IDLE
                elif s == "busy":
                    chat.send("Busy with previous command. Wait for it to finish.")
                    state = STATE_IDLE
                elif s == "done":
                    # Loop mode: advance to next step
                    if loop_steps:
                        step_name = loop_steps[loop_index]
                        loop_index = (loop_index + 1) % len(loop_cmds)
                        chat.send(f"Step '{step_name}' done. Next: '{loop_steps[loop_index]}'")
                        cmd = loop_cmds[loop_index].copy()
                        cmd["command_ts"] = time.time()
                        last_cmd = cmd
                        node.send_output("command", pa.array([json.dumps(cmd)]))
                        state = STATE_PLANNING
                        continue

                    cycle_just_done = total_cycles - remaining_cycles + 1
                    remaining_cycles -= 1
                    if remaining_cycles > 0 and last_cmd is not None:
                        next_cycle = cycle_just_done + 1
                        chat.send(f"Cycle {cycle_just_done}/{total_cycles} done. Starting cycle {next_cycle}...")
                        last_cmd["command_ts"] = time.time()
                        node.send_output("command", pa.array([json.dumps(last_cmd)]))
                        state = STATE_PLANNING
                    else:
                        if total_cycles > 1:
                            chat.send(f"All {total_cycles} cycles complete!")
                        else:
                            chat.send("Done!")
                        remaining_cycles = 0
                        total_cycles = 0
                        state = STATE_IDLE

        elif event["type"] == "STOP":
            break

    stop_event.set()
    print("[chat] Shutting down")


if __name__ == "__main__":
    main()
