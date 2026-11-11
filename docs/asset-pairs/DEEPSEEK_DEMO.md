# Run the same scene tool from another agent / DeepSeek

## Execution architecture

The model selects IDs and emits a `spaceman_scene` function call. Your agent runner receives the call, validates JSON and invokes `scripts/assets/scene_tool.py` (or its `execute(request)` function). The tool resolves registered assets and launches one resource-bounded Genesis process. Its JSON reply contains resolved IDs, render files, actual exit status and measured memory. Send that reply back as a tool message to the model. The model then chooses its next call or explains the result.

An API alone cannot access Genesis or the server filesystem. Run the executor in the user's Genesis environment. If the agent runs on another host, use an authenticated SSH executor restricted to this entrypoint rather than giving the model arbitrary shell access. Never send the API key as a scene argument or into the transcript.

The current Codex demo is actual Codex-selected tool execution. The calls in its saved request files are a recorded demo sequence, not proof that a different LLM autonomously selected them. The `run_tool_agent.py` adapter below uses live model-generated calls; it has not been executed against DeepSeek unless a separate API transcript is delivered.

## Tool contract

Register `scripts/assets/scene_tool.schema.json` in your agent's tool list. This schema uses the Chat Completions function-tool shape. A different harness may need an equivalent schema wrapper, but arguments and execution logic stay the same.

Actions:

- `list_assets`: returns eligible IDs; no simulator.
- `plan_scene`: returns a JSON plan; no simulator. Omitted IDs are selected by the seed.
- `render_scene`: builds and renders one scene under the existing CPU/memory/time wrapper. Requires a unique `run_id` and refuses to overwrite past runs.

Example input:

```json
{"action":"render_scene","carrier":"satellite_17","arm":"airbus_vispa","target":"gps","seed":11,"run_id":"deepseek_example_001"}
```

File CLI, from repository root:

```bash
python scripts/assets/scene_tool.py --request requests/example.json
```

Or pass JSON on stdin. Python harness dispatch:

```python
import json, sys
sys.path.insert(0, 'scripts/assets')
from scene_tool import execute

# For each function call emitted by your model:
if call.function.name == 'spaceman_scene':
    result = execute(json.loads(call.function.arguments))
    # Append result as the harness's tool-response message, preserving call.id.
```

Calls execute serially. Keep a render budget of at most four per demo and stop after a render failure. `safe_run.py` limits each process tree to a polled 8 GiB RSS / 240 seconds, two CPU threads and no GPUs, stopping when host available RAM is below 16 GiB. This does not certify hard-limit enforcement or physical capture.

## Minimal live API adapter (optional)

`scripts/assets/run_tool_agent.py` is a small standalone tool-call loop, not an installation of any particular harness. It uses the standard library and an OpenAI-compatible Chat Completions endpoint with function-tool support. Configure environment variables in the execution environment:

- `DEEPSEEK_API_KEY` or `API_KEY`: supply through your shell/secret manager; never paste it in the prompt or Git.
- `API_BASE_URL`: default `https://api.deepseek.com`; for another gateway include its API prefix, e.g. its documented `/v1` path. The adapter appends `/chat/completions`.
- `API_MODEL`: default `deepseek-chat`; change to the actual model ID supported by your API and check that function calling is supported.

Use the Genesis Python environment, from repository root:

```bash
python scripts/assets/materialize_meshes.py
python scripts/assets/run_tool_agent.py --prompt scripts/assets/demo_prompt.txt --output artifacts/deepseek_demo_trace.json --max-renders 4
```

On the existing server the runtime is `/data/Yanhan/genesis/angular_gripper_20261110/runtime`, and the interpreter is `/data/anaconda3/envs/Spaceman_env/bin/python`. Do not install packages into or modify someone else's agent environment. For a new checkout, prepare its own environment or use the known Genesis environment deliberately.

The included `scripts/assets/demo_prompt.txt` can be edited. Suggested request:

```text
先查询可用卫星和机械臂。请自主选择一个完整卫星、一个目标卫星，生成一个场景；然后只更换机械臂，保持载体和目标相同；再保持机械臂和目标，只更换载体卫星；最后调用 seed=42 的随机组合。最多生成四个场景，串行执行。为每次渲染设置唯一 run_id，并记录你为什么选择这些参数。检查每次工具返回的 ok、真实资源占用及图片路径，失败就停止渲染。最后给出四次组合对比和可打开的图片路径。不要训练，不要声称抓取成功，也不要把静态图片说成机械臂运动视频。
```

For an existing DeepSeek Harness, register this tool and read the same skill; keep the harness's own API loop instead of nesting the optional adapter inside it. The model should receive the tool schema plus skill context and actual tool results, not merely be asked to invent a shell transcript.
