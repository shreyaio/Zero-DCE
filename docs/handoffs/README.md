# Handoffs

Dated session records, newest first. Each is written to be read on its own.

| Date | Document | Covers |
|---|---|---|
| 2026-11-26 | `2026-11-26-phase-b-design.md` | **Current.** Phase A reinterpreted, Phase B designed, repository rebuilt and merged to `master` |
| 2026-11-11 | `2026-11-11-phase-a.md` | Phase A experiments as originally run and interpreted |

## Important

**The 26 November handoff supersedes the interpretation in the 11 November one.**

The earlier document proposes a brightness-aware cumulative-sensitivity regulariser
penalising `g²`/`J²`. That approach was subsequently shown to be structurally unable to
work — see `../the-pointwise-limit.md`. The *measurements* in the 11 November handoff
remain valid; its *conclusions about what to do next* do not.

It also refers to branch `claude/nifty-ritchie-qpqakn`, which was merged into `master`
on 26 November.

## Writing the next one

A handoff should let someone continue without the conversation that produced it. Include:

- Status at a glance, and the single next action
- What changed since the last handoff, and why
- Numbers, exact, with the context needed to read them correctly
- What was ruled out and the evidence that ruled it out
- Risks that could kill the direction
- Traps specific to this codebase
