# Resident interaction measurements (KI14)

`resident-performance.mjs` publishes a deterministic Arrow point dataset through the native Kepler upload workflow. It measures the resident adapter only. It does not compare against a native CPU implementation, estimate GPU execution time, or assert a 50 ms target.

Start the example in another terminal, then run one browser workload at a time:

```sh
cd examples/deck/kepler-compute
yarn start --host 127.0.0.1 --port 5318
```

From the repository root:

```sh
node examples/deck/kepler-compute/scripts/resident-performance.mjs \
  --rows 100000 --iterations 20 --warmup 3 --batch-size 65536 \
  --url http://127.0.0.1:5318/ \
  --output /tmp/resident-performance-100k.json

node examples/deck/kepler-compute/scripts/resident-performance.mjs \
  --rows 1000000 --iterations 20 --warmup 3 --batch-size 65536 \
  --url http://127.0.0.1:5318/ \
  --output /tmp/resident-performance-1m.json
```

Use `--headed` for a visible browser and `--label` to name the machine or run. Keep a headed benchmark in the foreground: the harness rejects a hidden document. `--help` prints all bounds without launching a browser. Workloads are capped at one million source rows, 100 measured interactions, and 20 excluded warmups. Each stage and interaction has a bounded timeout. The JSON report is written on success or browser-workflow failure, including completed samples and diagnostics.

## Workload

The source has three Float32 columns (`longitude`, `latitude`, `value`), no nulls, and explicit original Arrow batches. Coordinates form a deterministic NYC grid; `value = sourceRowIndex % 1000`. Upload requires no conversion. The built-in dataset remains fixed at 256 rows and its layer is hidden before measurement. The published source uses an actual native Point layer and a native Big Number chart configured as Count with map filters enabled.

The default `--interaction create-or-update` dispatches the host’s native `createOrUpdateFilter` action. This action updates dataset, field, and value; it is broader than an existing range slider’s value-only action. The separate `--interaction value-only` mode creates the exact native range filter once before warmup, then dispatches native `setFilter` with only its value. The host rejects a missing, duplicate, or mismatched filter identity. Both modes retain the same rows, range sequence, layer, chart, and budget; the report and default filename name the chosen mode. Existing reports remain immutable and describe create-or-update behavior. Each inclusive range has a known cardinality, checked against both the current resident summary and linked chart. Interactions are sequential; their changing cardinalities avoid accepting the prior count as the next result. This measures the filter action path, not mouse dragging, event coalescing, arbitrary geometry filters, or all Kepler workflows.

## Timing definitions

Cold durations use Node's monotonic clock and include Playwright transport and UI action overhead. They separate fixture generation/IPC serialization, application startup, file transfer/decode/coordinate role selection, schema checking, publication-to-current-summary, and publication-to-loaded-layer. “Cold” means the first source publication in a fresh browser context after app readiness; OS, driver, and Vite caches are not cleared. A loaded layer is not evidence of physical presentation.

Warm durations use `performance.now()` inside one browser evaluation, starting immediately before the native filter dispatch. Four-millisecond scheduled polling can be delayed by main-thread load. Raw samples, excluded warmups, p50, p95, minimum, maximum, and mean are retained. Percentiles use the nearest-rank definition.

| Metric | Endpoint |
| --- | --- |
| `submittedFrameMilliseconds` | Revision-tagged host-observed WebGPU submission boundary for the filter's current explorer revision. |
| `nextAnimationFrameMilliseconds` | First requested browser animation callback after observing that submitted frame. |
| `residentSummaryMilliseconds` | First polling observation of the matching current revision compact resident count. |
| `linkedChartMilliseconds` | First polling observation of the native Count DOM matching that current resident count. |

Submission and the next animation callback are a presentation opportunity, **not GPU completion or physical display latency**. There is no GPU timestamp query or per-frame completion fence. The hook captures the revision at producer encoding, so later pending filter revisions cannot label an earlier frame. If `getPerformanceSnapshot` is absent, summary/DOM observations can still run, but frame metrics are explicitly unavailable. No missing metric is converted to zero.

## Evidence and limits

Reports include browser version/arguments, device information when exposed, OS/CPU/run label, Git revision and tracked-change status, source rows/batches/schema/logical bytes/IPC bytes, and the native viewport. Source buffer IDs, actual buffer object identity, device identity, and source upload accounting must remain unchanged during warm interactions. Batch boundaries must match the original IPC source.

`readbackBytes.explorer` is the published explorer's explicit CPU readback payload counter. It excludes chart, table, scope, style, and other caller readbacks. Source upload accounting describes the original registered upload, not a complete device traffic monitor. Control uploads and total device transfers are marked unavailable. Buffer identity alone cannot prove the absence of writes into existing buffers.

Memory snapshots expose registry allocated/reserved bytes and optional Chromium JS heap measurements. They are before/after observations, not peak memory, total VRAM, or total process usage. Reservations are budget accounting, not a physical allocation measurement.

Run repeated reports on the same named device, browser mode/version, viewport, source batch size, and checkout before comparing numbers. The harness runs the development server with normal instrumentation; it does not silently extrapolate to production bundles or other devices. A failed report is not a performance result. The bounded browser smoke runs are recorded below; each run still requires memory admission.

Pure helper checks:

```sh
yarn test-node --no-coverage modules/kepler/test/resident-performance.node.spec.ts
node --check examples/deck/kepler-compute/scripts/resident-performance.mjs
```


## Recorded 100k smoke

The [immutable 2026-10-29 smoke report](../../../../dev-docs/perf/kepler-resident-2026-10-29/100k-smoke.json) passed with 100,000 rows in batches `[65536, 34464]`, one excluded warmup, and five measured interactions. The run used headless Chromium 151.0.7922.34 on the exposed Apple WebGPU `metal-3` adapter; an exact GPU model and driver version were unavailable.

| Observation | p50 | p95 |
| --- | ---: | ---: |
| Dispatch to submitted host frame | 35.8 ms | 44.7 ms |
| Dispatch to next requested animation callback | 46.2 ms | 51.5 ms |
| Dispatch to current resident Count observation | 54.5 ms | 58.9 ms |
| Dispatch to linked Count DOM observation | 54.5 ms | 58.9 ms |

With five samples, nearest-rank p95 is the maximum sample. This validates the harness; it is not acceptance of a latency target. The two Count measurements coincided at the polling resolution. Publication to a current summary took 413.1 ms; the point layer was observed loaded at 417.9 ms. These cold observations include automation overhead.

Source buffer object identity, IDs, batches, and device identity remained unchanged. Registered scalar source upload accounting remained at 1,200,000 bytes; the exposed explorer readback counter increased by 20 bytes across five interactions. Registry accounting stayed at 118,241,240 bytes, including 115,815,936 bytes of GPU reservations. These figures are not peak VRAM or total device traffic.

A historical admission audit after this run found that the then-current one-million-row scenario would exceed the unchanged 512 MiB registry budget before opening the primary consumer: source storage, positions, selection scopes, and eagerly created three-field domain calculations alone have a lower-bound reservation of 549,242,880 bytes, excluding the built-in fixture. The full warm scenario then projected about 867.1 MB of registry accounting because each owner retained its full hard-limit reservation. The accounting remedy below supersedes that admission result without increasing the global budget.


## Recorded 1m smoke after memory admission remedy

The [immutable 2026-10-29 one-million-row report](../../../../dev-docs/perf/kepler-resident-2026-10-29/1m-smoke.json) passed under the unchanged **512 MiB** registry budget, with one excluded warmup and five measured interactions. It used the same exposed Chromium 151 / Apple `metal-3` environment as the 100k smoke. The source retained 16 original batches: fifteen of 65,536 rows and one of 16,960 rows.

| Observation | p50 | p95 |
| --- | ---: | ---: |
| Dispatch to submitted host frame | 93.5 ms | 120.7 ms |
| Dispatch to next requested animation callback | 94.0 ms | 143.5 ms |
| Dispatch to current resident Count observation | 159.0 ms | 179.3 ms |
| Dispatch to linked Count DOM observation | 159.0 ms | 179.3 ms |

This is five-sample smoke evidence, not acceptance of the 50/100 ms interaction targets. Publication to current summary took 1,647.1 ms; the point layer was observed loaded at 1,649.8 ms. Source buffer objects, IDs, batch boundaries, revisions, and device identity remained unchanged. Registered source upload accounting stayed at 12,000,000 bytes. The exposed explorer readback counter increased by 20 bytes across the five measured interactions.

Registry accounting was 297,350,340 bytes before and after measurement, including 273,321,676 bytes of GPU reservations, leaving 239,520,572 bytes available. The remedy retains full-cap admission during construction, then charges planned/live resident allocations and bounded inspection capacity; atomic reservation growth precedes allocation, and retired resources retain their charge until their fence completes. This changes admission and reservation accounting; it does not establish a reduction in physical GPU allocations or peak VRAM.

Command-encoder-owned transient control-upload staging remains outside these resident charges. Its transfer count and total device traffic are unavailable. A follow-up must expose a preallocation upload-byte plan and submission completion lifecycle from the command encoder, admit those bytes before allocation, and bound pending submissions. Bounded inspection staging is already reserved. No budget increase, row reduction, or changed rendering workload was used for this run.


## Optional warm-stage diagnostics

Add `--diagnostics` to the same bounded workload to record synchronous native dispatch duration, the first observed revision change, poll CPU overhead, and an independent MutationObserver timestamp for the expected linked Count DOM. This DOM timestamp can precede the summary polling observation; it is not a rendering or presentation measurement. Where the optional host hook is available, each sample also captures revision-tagged producer encoding and explicitly named summary-read stage intervals. Missing stage hooks remain null. Schema version 2 additionally records cumulative encoding count/duration deltas during dispatch and through observation, including intermediate revisions. The opt-in host retains at most 128 encode events; missing events and counter resets are explicit. The latest encoding interval is retained separately and must not be interpreted as total encoding cost. Whole-workspace encode intervals overlap attribution across sources, so they must not be summed across datasets.

A CDP V8 CPU profile starts after warmup and stops after the measured loop or 30 seconds, whichever comes first. The `.cpuprofile` sidecar next to the report preserves sampled JavaScript call stacks; the report lists the top 40 leaf frames by sample count. Its requested sampling interval is 1,000 microseconds. Counts are not exact function durations, and profiler/observer/polling overhead perturbs the run. Keep these diagnostic reports separate from baseline performance measurements.

These observations do not split GPU execution from queue waits or browser scheduling. Time between JavaScript encoding and the submitted boundary remains unclassified; asynchronous summary-read duration includes scheduling and completion waits. No GPU timestamp query or extra completion fence is introduced. Low-level range predicates support reuse, but the recorded outer create-or-update action reconstructs intermediate filter topology, as the profile below demonstrates. Summary reads also follow workspace summaries, then publication scope summaries, then publication Count; the current profile does not isolate each asynchronous wait.


## Recorded 1m diagnostic findings

A separate [five-interaction diagnostic report](../../../../dev-docs/perf/kepler-resident-2026-10-29/1m-diagnostics.json), with its [raw CPU profile](../../../../dev-docs/perf/kepler-resident-2026-10-29/1m-diagnostics.cpuprofile), passed with the same source, batch boundaries, budget, and workload. Instrumented dispatch-to-submitted p50/p95 was 104.8/150.6 ms; the gated resident Count and Count DOM observations were 177.7/189.7 ms. These do not replace the uninstrumented baseline above.

Synchronous `setPublishedRange` dispatch took 64.6/73.2 ms p50/p95. The pinned native `createOrUpdateFilterUpdater` invokes dataset, field, and value updates even for an existing filter. CPU stacks show scalar predicate construction, plot explorer construction, and scope composition compilation inside this dispatch. They also show `requestUpdate` calling `deck.redraw` synchronously while the native reducer is applying intermediate filter changes. The first optimization candidate is deferring/coalescing those redraws until the native action finishes, while preserving current revision invalidation and stale-result rejection. A separate value-only native slider workload is needed; replacing the existing action silently would invalidate the comparison.

The 1.56-second CPU profile contains 630 samples. Counting each sample once per matching call stack: 219 include `setPublishedRange`, 154 include workspace producer encoding (all inside dispatch), 170 include `requestUpdate` (all inside dispatch), 27 include scalar construction, 15 include explorer construction, and 12 include composition compilation. These categories overlap and are sample counts, not execution durations. Another 233 samples include React outside dispatch. The profile supports a significant synchronous CPU contribution; it does not establish GPU execution cost.

The snapshot’s last successful encoding interval was 0.8–1.0 ms. **This is not total encoding time:** later encodes replace earlier intervals, and the CPU stacks reveal intermediate encodes during dispatch. The named published-summary workflow took 39.0/59.6 ms p50/p95, including asynchronous scheduling and completion waits. The independent expected Count DOM mutation was observed at 160.9/176.0 ms, earlier than the gated summary observations. Polling CPU overhead was 0.5/0.7 ms. These boundaries can overlap and must not be summed into a disjoint latency breakdown.

Registry accounting remained 297,350,340 bytes, exposed explorer readback increased by 20 bytes, registered source upload accounting was unchanged, and source/device object identities were retained. The diagnostic run did not change workload or admission limits. Raw diagnostic JSON and the `.cpuprofile` sidecar should be retained together when comparing a subsequent optimization.


## Recorded 1m comparisons after coalesced redraw

Two separate uninstrumented runs used five measured interactions after one excluded warmup, the unchanged 512 MiB budget, identical original batches and native point/Count configuration. Redraw requests now preserve synchronous invalidation while coalescing notifications and rendering after the native action completes. The original create-or-update workload remains separate from the new value-only slider mode.

| Workload | Submitted p50 / p95 | Next animation callback p50 / p95 | Count p50 / p95 |
| --- | ---: | ---: | ---: |
| Original create-or-update, before coalescing | 93.5 / 120.7 ms | 94.0 / 143.5 ms | 159.0 / 179.3 ms |
| Create-or-update, after coalescing | 62.8 / 87.2 ms | 64.3 / 109.4 ms | 95.5 / 140.6 ms |
| Value-only slider, after coalescing | 39.7 / 61.9 ms | 48.7 / 71.4 ms | 83.5 / 105.5 ms |

The corresponding immutable reports are [coalesced create-or-update](../../../../dev-docs/perf/kepler-resident-2026-10-29/1m-coalesced-create-or-update.json) and [coalesced value-only](../../../../dev-docs/perf/kepler-resident-2026-10-29/1m-coalesced-value-only.json). The latter seeds the filter once before warmup and then sends only native value updates; it cannot replace the original action as a same-workload comparison. The Count column is the current-summary-gated observation, which coincided with linked DOM polling. No CPU profiler or optional stage telemetry was enabled in either new run.

Both runs retained source/device object identity, all 16 source batches, and 12,000,000 bytes of registered scalar uploads. Each read back 20 exposed explorer bytes across the measured interactions. Registry accounting remained 297,350,340 bytes before and after measurement, with 239,520,572 bytes available. These are accounting snapshots, not peak VRAM. The expected counts were 237,000, 374,000, 511,000, 648,000, and 785,000 in both modes.

These short development-server runs show improvement on the same create-or-update workload; they do not establish a causal percentage gain or production performance guarantee. Root Node checks ran concurrently, and cold driver/server caches were not controlled. Even the value-only run’s submitted and Count p95 values remain above 50/100 ms. Longer repeated runs on a quiescent named machine and additional workflows are still required for target acceptance.
