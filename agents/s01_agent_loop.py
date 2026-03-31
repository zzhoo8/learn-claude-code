#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""
import asyncio
import os
import subprocess

from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel, ChatResponse
from agentscope.tool import ToolResponse

try:
    import readline
    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# MODEL = os.environ["MODEL_ID"]

# 支持 agentscope
from agents import config
client = OpenAIChatModel(
    model_name=config.Config.OPENAI_CONFIG.get('model'),
    api_key=config.Config.OPENAI_CONFIG.get('api_key'),
    client_kwargs={
        'base_url': config.Config.OPENAI_CONFIG.get('api_base'),
    },
    generate_kwargs={
        'temperature': 0.01,
        'max_tokens': 4096
    },
    stream=False,
)


SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- The core pattern: a while loop that calls tools until the model stops --
async def agent_loop(messages: list):
    while True:
        # response = client.messages.create(
        #     model=MODEL, system=SYSTEM, messages=messages,
        #     tools=TOOLS, max_tokens=8000,
        # )
        # 支持 agentscope
        _msgs = [Msg(name='', role=message.get('role'), content=message.get('content')) for message in messages]
        _formatter = OpenAIChatFormatter()
        _prompt = await _formatter.format(msgs=[
            # 系统提示词总在最前
            Msg(name="系统", role="system", content=SYSTEM),
            *_msgs
        ])
        response: ChatResponse = await client(
            messages=_prompt,
            tools=TOOLS,
        )
        if len(response.content) < 2:
            return
        _content = response.content[1]
        # Append assistant turn
        # messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "assistant", "content": [_content]})
        # If the model didn't call a tool, we're done
        # if response.stop_reason != "tool_use":
        #     return
        if _content.get('type') != 'tool_use':
            return
        # Execute each tool call, collect results
        # results = []
        # for block in response.content:
        #     if block.type == "tool_use":
        #         print(f"\033[33m$ {block.input['command']}\033[0m")
        #         output = run_bash(block.input["command"])
        #         print(output[:200])
        #         results.append({"type": "tool_result", "tool_use_id": block.id,
        #                         "content": output})
        _input: dict = _content.get('input')
        print(_input)
        output: str = run_bash(command=_input.get('command'))
        print(output)
        # tool_response = ToolResponse(
        #     TextBlock(
        #         type="text",
        #         text=output,
        #     ),
        # )
        # tool_response.content
        result = {
            "type": "tool_result",
            "tool_use_id": _content.get('id'),
            "output": output,
        }
        messages.append({"role": "user", "content": [result]})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        asyncio.run(agent_loop(messages=history))
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
