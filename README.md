# When Memory Matters

Evaluation code for **When Memory Matters: A Diagnostic Benchmark for
Memory-Guided LLM Play in Zero-Sum Games**. It compares Games, Memory Item
families, and Agents in matched head-to-head games. The paper portfolio is
Nim, TicTacToe, ConnectFour, KuhnPoker, Othello, and Breakthrough.

## Install and test

Use Python 3.11 or newer. TextArena is pinned to the version used by the
benchmark. From the project directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

The tests run without API access. Install `".[plots]"` as well to render
local run summaries and enable progress bars.

## Run comparisons

Live runs make paid model calls. Set the provider API key in your shell or
in a local `.env`, using [.env.example](.env.example) as a guide. The model
ID must be available on the selected endpoint, which must support the
reasoning toggle used by the selected interface.

| Question | Command | What varies |
| --- | --- | --- |
| Memory Item families | `wmm-memory-items` | The Language Policy supplied to the same model and reasoning interface |
| Reasoning interfaces | `wmm-interfaces --comparison reasoning` | Thinking, Prompt-guided, and Non-thinking with the same model and Memory Item |
| Model identities | `wmm-models` | Model with the same Memory Item and Prompt-guided reasoning interface |
| Evolvability | `wmm-interfaces --comparison baseline` | Memory-guided versus memory-free play, with model and reasoning interface matched |
| Reference policy | `wmm-screen` | Legacy Game screening; use the reference-policy outcome counts, not its composite score |

Each command supports `--help`. For a small, illustrative Memory Item
comparison using the supplied demo group:

```sh
wmm-memory-items --game TicTacToe-v0 --provider volcengine \
  --model deepseek-v4-flash --thinking_mode non_thinking_pgt \
  --items trajectory insight reasoning_pattern \
  --memory-group examples/tictactoe_demo_group.json --num-games 2
```

The output goes under `runs/bench/` by default. Run folders contain
`run_config.json`, `summary.csv` for completed pairings, and per-game JSON
traces. Traces include prompts and model responses; review them before
sharing. A partially failed comparison exits nonzero.

See [Evaluation protocol](docs/benchmark_protocol.md) for result
orientation, aggregation, and the difference between demo runs and
reproducing the paper. The legacy LangPSRO learning experiments are
available through `run_experiment.py`; they are not the three-axis
benchmark.
