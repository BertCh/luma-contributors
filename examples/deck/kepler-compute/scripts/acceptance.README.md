# Acceptance harness

`acceptance.mjs` is a Playwright harness that measures resident interactions at 1M, 5M, and 10M rows and compares them with per-hardware-tier targets. Measured results of the first runs are summarized in `dev-docs/archive/kepler-compute-integration-implementation.md` (removed from the tree on 2026-11-13; in git history at `d9d56f0f`); reports are not committed. The existing 1M smoke reports in `resident-performance.README.md` remain comparisons, not acceptance.

## Running

Start the example in one terminal:

```sh
cd examples/deck/kepler-compute
yarn start --host 127.0.0.1 --port 5318
```

From the repository root:

```sh
node examples/deck/kepler-compute/scripts/acceptance.mjs \
  --sizes 1000000,5000000,10000000 --iterations 20 --warmup 3 --frame-seconds 5 \
  --url http://127.0.0.1:5318/ \
  --output <dir>/acceptance-report.json \
  --label <machine> [--tier <id>] [--headed]
```

Datasets load through `?rows=N`, which explicitly selects the synthetic fixture even though the app now opens a public Kepler sample by default. The cap is raised to 10,000,000 rows; `?sample=synthetic` without `rows` uses 100,000. `?memoryBudgetMebibytes=N` (DEV, any query on `--url` is preserved) raises the registry's accounted dataset budget from its 512 MiB default; 1M rows as the base fixture needs it, and 10M rows needed about 8.5 GiB of registry total (including reservations and transient selection capacity) so use `memoryBudgetMebibytes=32768` for 1M, 5M and 10M. The explorer working-memory reservation is 192 bytes per row plus 8 MiB. `--verbose` logs page console output and page errors. The harness writes the JSON report and a markdown report with the same basename.

| Exit code | Meaning |
| ---: | --- |
| 0 | Pass |
| 1 | Fail |
| 2 | Incomplete or workflow error |
| 3 | Usage error |

## Files

| File | Role |
| --- | --- |
| `scripts/acceptance.mjs` | Playwright driver and report writer |
| `scripts/acceptance-targets.json` | Targets keyed tier, then rows, then metric |
| `scripts/acceptance-utils.mjs` | Pure logic (statistics, target evaluation, tier classification, report assembly) |
| `src/acceptance-instrumentation.ts` | DEV-only page instrumentation |
| `modules/kepler/test/acceptance-report.node.spec.ts` | Node spec for the pure logic |

```sh
yarn test-node --no-coverage modules/kepler/test/acceptance-report.node.spec.ts
```

## Instrumentation

In development builds `src/acceptance-instrumentation.ts` installs `window.__keplerComputeAcceptance`. It offers `beginInteraction`, `getInteraction`, frame recording, upload measures, a memory snapshot, and a device report. It also emits `performance.mark` and `performance.measure` entries named `kc:*`. The hooks observe host-submitted frames and current-summary revisions.

## Hardware tiers and targets

Tiers are `reference-desktop`, `integrated`, `low-power`, and `unclassified`. The report records adapter info and limits, and the reasons for the classification. `--tier` overrides detection; the report records it as an override, not a detection.

Metric keys per row count: `filterLatency`, `brushLatency`, `histogramLatency`, `styleLatency`, `linkedSummaryLatency`, `frameTime`, `uploadTime`, `memoryAccounted`, `warmUploadBytes`, `warmReadbackBytes`. Each target has `statistic`, `maxValue`, `unit`, and `origin`.

| Origin | Meaning |
| --- | --- |
| `plan` | Stated in `dev-docs/archive/kepler-compute-integration-plan.md` (in git history at `d9d56f0f`) |
| `derived` | Computed from a plan target |
| `provisional` | Proposed here; needs ratification before it counts as acceptance |

Only these targets come from the plan: filter p95 under 50 ms at 1M rows on reference hardware, linked compact summaries under 100 ms, a 16.7 ms playback frame budget on reference desktop, and zero warm full-dataset upload and readback. Everything else is provisional. A pass against a provisional target is not acceptance.

## Limits

- Host-submitted-frame and current-summary observations are not GPU completion or physical presentation. There are no GPU timestamp queries.
- Accounted bytes are registry accounting, not VRAM and not peak process memory.
- The harness adds `?memoryBudgetMebibytes=32768` for 1M rows and above unless the `--url` query sets it. Without it, 1M rows and above are blocked by the 512 MiB registry budget. Blocked sizes are reported as blocked with the page error, not failed and not passed. `warmReadbackBytes` is the measured-pass readback total divided by the number of measured interactions (per interaction), with the total in the run notes.
- Runs use the development server with its instrumentation. They do not represent production bundles.
- Compare reports only on the same named machine, browser mode and version, viewport, and checkout. Keep a headed run in the foreground.
- A failed or incomplete report is not a performance result.

## Recorded results

None yet. After a run on a named machine, copy the numbers from the generated report into a table like this one and link the immutable report.

Machine: `<name>`; browser: `<version, headed/headless>`; tier: `<detected or override>`; commit: `<sha>`; report: `<path>`.

| Rows | Metric | Statistic | Measured | Target | Origin | Result |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1,000,000 | filterLatency | p95 | | | | |
| 1,000,000 | linkedSummaryLatency | p95 | | | | |
| 1,000,000 | frameTime | p95 | | | | |
| 1,000,000 | warmUploadBytes | max | | | | |
| 1,000,000 | warmReadbackBytes | max | | | | |
| 5,000,000 | | | | | | |
| 10,000,000 | | | | | | |

Result is one of pass, fail, blocked, or incomplete.
