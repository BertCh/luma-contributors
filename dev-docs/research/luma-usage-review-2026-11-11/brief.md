# Shared brief: luma.gl usage review (2026-11-11)

Repository: `<repository>` (branch `upstream-first`).
It holds GPU geospatial analysis code (`modules/analysis`), computation lenses (`modules/lenses`),
kepler.gl adapters (`modules/kepler`), a website (`website/`) and two examples (`examples/deck/*`),
all built on **published** luma.gl. Read `AGENTS.md` at the repository root first (short): it has
the upstream-first rules that bound what a recommendation may say.

## The question

Is this repository using the published luma.gl ecosystem (`@luma.gl/core`, `@luma.gl/shaders`,
`@luma.gl/graph`, and for examples, website and tests `@luma.gl/engine`) and its built-in features
as well as it can? Concretely, find:

1. **Local code that a published luma.gl feature already does**, where the local code could call
   the published feature today with no loss the callers care about.
2. **Published features nobody here uses** that would make existing code simpler, faster, more
   correct or more robust.
3. **luma.gl APIs used in a poor or non-idiomatic way** (work luma.gl would do for us done by
   hand, resources not released, caches bypassed, limits not consulted, the wrong entry point,
   a deprecated or internal-looking route when a public one exists).
4. **Stale claims**: places where `dev-docs/upstream-workarounds.md` or code comments say
   "luma.gl has no X" and the installed release does have X.

The maintainer prefers "the luma.gl way of doing things and their official implementation":
converge on published luma.gl and delegate where possible. But only where it is actually better or
at least equal: do not recommend a delegation that loses accuracy, determinism, a documented limit
or a capability a caller uses. Say what would be lost when something would.

## What is already established (do not redo it)

- Installed release: `@luma.gl/*@10.0.0-alpha.13` (bumped today from alpha.12; source typecheck and
  both rule checkers pass). The delta from alpha.12 is small: new `GPUGraphReduction` and
  `GPUGraphGroupAggregation` (semantic `GPUGraph` operations, `@luma.gl/graph/gpu-core`),
  `DeviceResource` in core (replaces the `ResourceProps` type, unused here),
  `createTextCodePointSource` in `@luma.gl/engine/text`, shared `getGPUDataBufferProps` buffer
  usage defaults inside `gpu-data`/`arrow`/`gpu-tables`, and published `GPUReduction` and
  `GPUGroupAggregation` now attach dispatch-workgroup metadata to their nodes.
- `dev-docs/upstream-workarounds.md` is the register of every known place where local code stands
  in for luma.gl. **Read it before you start** (150 lines, long table rows). Section 4 lists 27
  local kernels beside a published class, each with the reason it cannot delegate. Do not report a
  row that is already there, unless you find that its reason is wrong or no longer holds at
  alpha.13, or that delegation is possible today for part of it. That is a finding.
- `dev-docs/research/math-correctness-review-2026-11-10/luma-official.md` is the previous
  comparison (against alpha.10). It did NOT cover: `modules/lenses`, `modules/kepler`, the website,
  the examples, `@luma.gl/engine`, the implementation of `gpu-dataframe` / `gpu-sql` /
  `gpu-crossfilter` / `gpu-tables` / `arrow`, local gather and run-length code against `GPUGather`
  and `GPURunLengthEncode`, local point-to-segment distance against
  `GPUPairwisePointSegmentDistance`, and it ran nothing on a GPU. Its "Not covered" list (section 1)
  and its section 5 on conventions are worth reading for your area.
- `dev-docs/roadmap.md` section 13.7 (items FP64-01 to FP64-05) and
  `dev-docs/research/extended-precision-2026-11-11/README.md` already cover the published fp64
  shader modules.

## Inventory files (machine-made today, in the scratchpad)

Directory `<scratchpad>/`:

- `exports-installed.json`: every public export of every entry point of the four installed
  packages: `{entry: {name: {kind: 'class'|'function'|'const'|'enum'|'type', file}}}`; `file` is
  the declaring `.d.ts` relative to the package root in `node_modules/@luma.gl/<package>/`.
- `usage-report.json`: per entry point, the value exports (class, function, const) that no file in
  the repository imports (`unused`) and those imported only outside `modules/*/src` (`testOnly`:
  tests, website, examples, docs).
- `usage.json`: `usage["<specifier>#<name>"][area] = {value, type}` import counts per area
  (`analysis-src`, `lenses-src`, `kepler-src`, `tests`, `website`, `examples`, `docs`).
- `usage-summary.txt`: the same as two short tables.

The matching is by imported name, so an unused name is a lead, not a finding: the published class
may simply not be relevant. A finding needs the local code that does the same job.

## Method

- Published source to read: `node_modules/@luma.gl/<package>/dist/**/*.d.ts` for the contract
  (class documentation there is usually good) and the sibling `.js` for what the kernel really
  does. Only names exported from an entry point's `index.d.ts` are public; a file under `dist`
  that no index exports is NOT usable here (AGENTS.md).
- Work capability by capability: for each published feature in your stream, find the local code
  that does that job (`git grep` over the scope given to you), read both sides, and decide.
- Read both sides before you call two things the same. State for each finding whether you read
  both kernels ("read both"), read one and the other's documentation ("read one"), or inferred
  from names ("inferred"). Mark inferred findings as leads.
- Count things with `git grep -c` / `wc` instead of estimating.

## Rules

- **Read-only.** Do not edit, create or delete any file inside the repository. Another session
  may be working in this checkout. No `git` command that changes state (no stash, checkout, reset,
  add, commit). No `yarn install`, `yarn build`, `yarn lint-fix`.
- No GPU or browser runs (no `--project headless`). You may run existing node specs
  (`yarn vitest run --project node <file>`) and your own `node` scripts kept in the scratchpad
  directory above (under `review/<your-stream>/`).
- Use `git grep` (ripgrep may not be installed). Paths in your report: repository-relative, with
  line numbers, for both the local code and the published file.
- Do not recommend importing anything that is not a public entry-point export, forking or copying
  luma.gl source, or re-exporting luma.gl names (AGENTS.md).

## Report

Write your report to
`<scratchpad>/review/<your-stream>.md`
(at most about 300 lines). Structure:

1. **Coverage**: what you read, what you did not get to (be exact: a later pass relies on it).
2. **Findings**, most valuable first. Each one:
   - an ID `<STREAM>-NN` and a one-line title;
   - class: `adopt-now` (possible today, no loss), `adopt-with-cost` (possible today, loses
     something: name it), `misuse`, `unused-feature` (nothing local duplicates it, but it would
     improve something concrete: name what), `blocked` (needs luma.gl to change AND is not in the
     register yet: say what it needs), `stale-claim`;
   - local `file:line`, published `file:line`, evidence level (read both / read one / inferred);
   - what to change, in two or three sentences, and how many call sites it touches (counted);
   - how to verify it (which existing spec covers the local code, or what parity spec is needed);
   - risk: what could differ in results (rounding, order, validity, limits).
3. **Checked and fine**: published features you compared with local code and found correctly used
   or correctly not used, one line each with the reason. This matters as much as the findings.
4. **Register corrections**: rows of `upstream-workarounds.md` that are wrong or out of date.

Your final message back should be a short summary: counts per class and the five findings you
would act on first. The report file is the deliverable.
