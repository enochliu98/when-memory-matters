# Evaluation Protocol

## Matched comparisons

- Each run uses a positive, even game count per pairing and swaps player
  assignments halfway through. An unordered pair is played once.
  Transposed matrix entries are views of the same games; diagonal cells
  are unplayed.
- Thinking enables model-internal reasoning. Prompt-guided disables
  model-internal reasoning and injects a deliberation instruction.
  Non-thinking uses neither. The corresponding CLI values are
  `thinking`, `non_thinking_pgt`, and `non_thinking`.
- Given `--memory-group`, the runner substitutes the authored Language
  Policy body; observation, instruction, and output format remain the
  same between sides at a fixed reasoning interface.
- `wmm-interfaces --comparison baseline` matches the model and reasoning
  interface on both sides. `--baseline-mode non_thinking_pgt` explicitly
  selects the historical Prompt-guided baseline and its original prompt.
  Do not mix those runs into matched Evolvability results.

## Memory Item groups

[The demo JSON](../examples/tictactoe_demo_group.json) documents the input
format. Use one file per Game and group, with one full Language Policy
body per family. Do not include a complete agent prompt. In a Trajectory
comparison meant to isolate episodes, omit strategic lessons and
procedural summaries. The run configuration records the group ID and
SHA-256 of the input file.

For a multi-group evaluation, run each file separately. Keep the model and
reasoning interface fixed while comparing all three pairs within a group.
For Game Axis Evolvability, use the Reasoning Pattern from each group
against the matched memory-free baseline. The built-in examples in
`Bench/memory_items.py` are historical, single-instance policies; they
are not the complete reviewed groups used to report paper results.

## Reading and aggregating scores

For a record oriented to the first-named policy, let W, D, and L denote
wins, draws, and losses, and N = W + D + L.

| Quantity | Formula | First-named policy |
| --- | --- | --- |
| Non-saturation | (W - L) / N | Reference policy |
| Evolvability | (W - L) / N | Memory-Guided policy |
| Pairwise capability | (W + 0.5 D) / N | Memory Item or Agent being scored |

`wins_p0` / `wins_p1` in Bench summaries follow the two named conditions,
not the physical player slots. Baseline summaries explicitly label both
policy roles. The legacy screening report's non-saturation counts are
from the LLM's side, so reverse wins and losses before computing
Non-saturation. Its depth-limited or sampled reference policies are
comparators, not proofs of optimal play. Do not use the screener's
historical expert-prompt matchup or weighted rating as the paper metric.

Pool recorded outcomes for the same Game and Agent when summarizing
Evolvability across Memory Items. For Memory Item comparisons, average
scores over opponents within each group and Game; summarize group-level
macro-averages with sample standard deviations. For Agent comparisons,
pool opponent counts within each Game and Memory Item, then macro-average
cells. Never replace W/D/L counts with rounded equivalents for a new N.

Completion-token cost is total completion tokens divided by model calls
for a condition. When the endpoint reports internal reasoning tokens as
part of completion, they must not be added again. Keep cost runs and
their denominators alongside the outcome records used for analysis.

## Reproduction checklist

The source bundle demonstrates the protocol and supports new runs. For
exact reproduction, also retain the original run identifiers, resolved
model and provider versions, reviewed policies and their IDs, reference
policy settings, raw outcome counts and token totals, and the mapping
from each reported cell to its source. Incomplete pairs, synthetic
examples, and unplayed matrix cells are not observed games. Provider
output may change even with the same endpoint name.
