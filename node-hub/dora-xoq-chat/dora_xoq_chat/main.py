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
import re
import threading

import pyarrow as pa
from dora import Node

CHAT_CHANNEL = os.getenv("CHAT_CHANNEL", "anon/openarm-chat")
CHAT_RELAY = os.getenv("CHAT_RELAY", "https://cdn.1ms.ai")
CHAT_USERNAME = os.getenv("CHAT_USERNAME", "robot")

STATE_IDLE = "idle"
STATE_PARSING = "parsing"
STATE_PLANNING = "planning"
STATE_AWAITING_CONFIRM = "awaiting_confirm"
STATE_EXECUTING = "executing"

VLM_PARSE_PROMPT = (
    "Extract the pick and place targets from this robot command. "
    'Output ONLY JSON: {"pick": "object description", "place": "target description"}. '
    "If no place target, omit the place key.\n"
    "Command: {command}"
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
    print(f"[chat] Connected to {CHAT_RELAY} channel={CHAT_CHANNEL} as {CHAT_USERNAME}")

    msg_queue = queue.Queue()
    stop_event = threading.Event()
    recv_thread = threading.Thread(
        target=_chat_recv_thread, args=(chat, msg_queue, stop_event), daemon=True
    )
    recv_thread.start()

    state = STATE_IDLE
    node = Node()

    for event in node:
        if event["type"] == "INPUT":
            event_id = event["id"]

            # --- Tick: drain chat messages ---
            if event_id == "tick":
                while not msg_queue.empty():
                    try:
                        msg = msg_queue.get_nowait()
                    except queue.Empty:
                        break

                    # Ignore our own messages
                    sender = getattr(msg, "name", "") or ""
                    if sender == CHAT_USERNAME:
                        continue

                    text = str(msg.text if hasattr(msg, "text") else msg).strip()
                    if not text:
                        continue

                    text_lower = text.lower()
                    print(f"[chat] Received from {sender}: {text}")

                    # --- Cancel (any state except idle) ---
                    if text_lower == "cancel" and state != STATE_IDLE:
                        print(f"[chat] Cancel requested (was {state})")
                        chat.send("Cancelled.")
                        state = STATE_IDLE
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

                    # Try direct JSON first
                    cmd = _try_parse_json(text)
                    if cmd and "pick" in cmd:
                        print(f"[chat] Direct JSON command: {cmd}")
                        chat.send(f"Got it: pick '{cmd.get('pick', '')}'"
                                  + (f", place in '{cmd['place']}'" if cmd.get("place") else "")
                                  + ". Planning...")
                        node.send_output("command", pa.array([json.dumps(cmd)]))
                        state = STATE_PLANNING
                        continue

                    # Free-text: send to VLM for parsing
                    print(f"[chat] Sending to VLM for parsing: {text}")
                    chat.send("Understanding your command...")
                    prompt = VLM_PARSE_PROMPT.format(command=text)
                    _send_vlm_text(node, prompt)
                    state = STATE_PARSING

            # --- VLM response (command parsing) ---
            elif event_id == "vlm_response" and state == STATE_PARSING:
                text = event["value"][0].as_py()
                print(f"[chat] VLM parse response: {text[:200]}")
                cmd = _try_parse_json(text)
                if cmd and "pick" in cmd:
                    chat.send(f"Understood: pick '{cmd.get('pick', '')}'"
                              + (f", place in '{cmd['place']}'" if cmd.get("place") else "")
                              + ". Planning trajectory...")
                    node.send_output("command", pa.array([json.dumps(cmd)]))
                    state = STATE_PLANNING
                else:
                    chat.send("I didn't understand that. Try: pick the <object> and put it in the <container>")
                    state = STATE_IDLE

            # --- Trajectory status from motion planner ---
            elif event_id == "trajectory_status":
                raw = event["value"][0].as_py()
                try:
                    status = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                s = status.get("status", "")
                if s == "ready" and state == STATE_PLANNING:
                    wp = status.get("waypoints", "?")
                    dur = status.get("duration", "?")
                    arm = status.get("arm", "?")
                    chat.send(f"Trajectory planned: {wp} waypoints, ~{dur}s on {arm} arm. "
                              "Reply 'ok' to execute or 'cancel'.")
                    state = STATE_AWAITING_CONFIRM
                elif s == "failed":
                    chat.send("Planning failed. Try a different command.")
                    state = STATE_IDLE
                elif s == "done":
                    chat.send("Done!")
                    state = STATE_IDLE

        elif event["type"] == "STOP":
            break

    stop_event.set()
    print("[chat] Shutting down")


if __name__ == "__main__":
    main()
