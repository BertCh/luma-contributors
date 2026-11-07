# Kepler compute acceptance report

Generated: 2026-10-31T20:05:34.407Z

## Environment

| Field | Value |
| --- | --- |
| Browser | Chromium 151.0.7922.34 (headless) |
| Hardware tier | unknown |
| Tier reasons | No device report was available |
| Git revision | cc97a1c148a737d620d903c3f4424d8d23bca74b-dirty |
| Adapter | {} |

## 1,000,000 rows (incomplete)

| Metric | Statistic | Target | Observed | Origin | Status |
| --- | --- | --- | --- | --- | --- |

Missing targets: filterLatency, brushLatency, histogramLatency, styleLatency, linkedSummaryLatency, frameTime, uploadTime, memoryAccounted, warmUploadBytes, warmReadbackBytes

- Blocked: Admission or load failed: page.waitForFunction: Error: Dataset memory budget exceeded
- Error: Dataset memory budget exceeded
    at KeplerDatasetRegistry.assertCapacity (http://127.0.0.1:5318/@fs/Users/robertchristie/Documents/GitHub/vis.gl-build/luma-contributors/modules/kepler/src/runtime/dataset-registry.ts:452:10)
    at KeplerDatasetRegistry.reserveMemoryDirect (http://127.0.0.1:5318/@fs/Users/robertchristie/Documents/GitHub/vis.gl-build/luma-contributors/modules/kepler/src/runtime/dataset-registry.ts:103:8)
    at KeplerDatasetRegistry.reserveMemory (http://127.0.0.1:5318/@fs/Users/robertchristie/Documents/GitHub/vis.gl-build/luma-contributors/modules/kepler/src/runtime/dataset-registry.ts:95:16)
    at new GPUKeplerSelectionScopes (http://127.0.0.1:5318/@fs/Users/robertchristie/Documents/GitHub/vis.gl-build/luma-contributors/modules/kepler/src/selection/gpu-selection-scopes.ts:79:38)
    at new ResidentScopeBinding (http://127.0.0.1:5318/src/resident-scope-binding.ts:59:13)
    at new ComputeRuntime (http://127.0.0.1:5318/src/compute-runtime.ts:132:19)
    at initializeDevice (http://127.0.0.1:5318/src/main.tsx:2489:13)
    at Object.onDeckInitialized (http://127.0.0.1:5318/src/main.tsx:2790:41)
    at MapContainer._onDeckInitialized (http://127.0.0.1:5318/node_modules/.vite/deps/esm-Brhr_HAQ.js?v=bfff76ac:58632:51)
    at Object.onDeviceInitialized (http://127.0.0.1:5318/node_modules/.vite/deps/esm-Brhr_HAQ.js?v=bfff76ac:58929:22)

## Overall verdict

**incomplete**

## Disclaimers

- Frame and latency observations are host-submitted-frame timings. They are not GPU completion or physical presentation latency.
- Origin `plan` targets come from the integration plan; origin `ratified` targets were scaled from it and ratified by the project owner. Any `provisional` or `derived` target is not yet accepted.
