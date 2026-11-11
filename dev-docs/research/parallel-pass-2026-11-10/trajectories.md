# trajectories report

## 1. Done
- STD-03: no bare `assert` left in my directories (`yarn check-contributor-contract --all` lists none for them).
  - `track-similarity/gpu-track-similarity-search.ts`: 14 asserts -> validateRequirement/validateCapacity/validateFiniteNumber/validateViewLength; messages start `track-similarity-search`, name the prop and limit (output.distance/lowerBound, lcssTolerance, survivorCapacity in [0, pairs], topK in [1,1024], threshold rows, topK outputs, output overlap).
  - `trajectory-estimation/prepare-trajectory-observations.ts`: 7 asserts; messages start `prepareGPUTrajectoryObservations` (no ID prop on a function).
  - `trajectory-zones/zone-events-kernels.ts`: table-gather binding count via validateMaximumCount (max 8).
  - No kernel-audit rejection cases added: the audit rejection list only holds binding/dispatch limits and none of these limits changed. Rejection cases are in the new node spec instead.
- STD-06: optional `spatialContext` on `GPUTrajectoryKalman` and `GPUTrajectoryRestriction` (files in trajectory-estimation, trajectory-restriction), resolved with `resolveGPUSpatialContext`, exposed as `readonly spatialContext`. Planar only, documented; longitude-latitude etc. rejected at construction. Kalman and restriction-with-pairs also use measure 'distance' (native metric, no lambert-equal-area frame). Omitting it is unchanged. Checked the rest: `track-similarity`, `GPUTrackSimilaritySearch` already had it (roadmap row is stale there); Hampel (values only), Split and Segments (only copy or compare coordinates for equality), isopleths (density grid), track-relative-times (times only) do not measure coordinates, so no context added.
- STD-07: additive aliases. `GPUTrajectoryEncounters.mask` and `GPUTrajectoryClosestApproach.mask` (per-track input filter; `trackValid` deprecated, both given throws), helper `resolveTrajectoryTrackMask` in `trajectory-analysis/trajectory-common.ts`; `clock-encounters` omits `mask` too. `GPUTrajectoryThinning` `output.validity` (`output.mask` deprecated, both given throws). Hampel/outlier `flags` left: they mean "is an outlier/replaced" (1 = bad), the inverse of validity.
- STD-04: documented every bit of `GPU_TRAJECTORY_KALMAN_STATUS` and `GPU_TRAJECTORY_RESTRICTION_STATUS` (incl. the 6 = invalid+predictionOnly combination) and on their output `status` / `compactStatus`. `GPUZoneEventsStatus` now composes `GPUCandidateStatusPort` and `GPUStatusScalar` (same shape). `GPUTrajectoryClosestApproach` `output.status.candidateOverflow` optional new word (OR of entry/hit overflow), kernel variant `status-candidate`, so it speaks the port vocabulary alongside the old two words.
- Docs: `docs/api/gpu-network/network-movement-coverage.md` (spatialContext, status), `docs/api/gpu-spatial-analysis/operators-lines.md` (`mask`).
- Spec: `modules/analysis/test/gpu-spatial-analysis/trajectory-analysis/trajectory-contract-completion.node.spec.ts` (6 tests).

## 2. Verified
- `npx vitest run --project node` on that spec: 6 passed. Node on trajectory-analysis, trajectory-encounters, track-similarity, trajectory-zones dirs: 13 files, 61 tests passed.
- `npx vitest run --project headless` on trajectory-estimation, -cleaning, -restriction, -encounters, -zones: 16 files, 156 passed; track-similarity: 7 files, 33 passed.
- `npx biome check --write` on my src dirs and the new spec: clean.
- `yarn check-contributor-contract --all`: no findings in my directories.
- `node scripts/typecheck-status.mjs 'track-similarity|trajectory'`: 0 errors at 18:24:56, but I could not confirm that cycle covered my last edits (a later call was still waiting for a new cycle when I wrote this).
- Note: the disk was at 100% (107 MiB free) during the work; a command was refused once with ENOSPC.

## 3. Not done
- Kernel-audit rejection cases (see above, no limit changed).
- STD-04 for closest-approach: `entryOverflow`/`hitOverflow` stay (renaming would break); the zone and closest approach flags are scalar, not renamed.

## 4. Wave 2 requests
- Rename completion: `trackValid` -> `mask` (encounters, closest approach; consumers in modules/lenses, kepler, examples, website/src/gpu: none found by grep except test oracles `parity-wp2-*.py` and the generated website inventory); `GPUTrajectoryThinning` `output.mask` -> `output.validity`. Then drop the aliases and the `status-candidate` split (make `candidateOverflow` the only word, drop `entryOverflow`/`hitOverflow` or keep as detail).
- Inventory (`website/src/catalog/generated/inventory.json`) will regenerate with the new props.

## 5. Roadmap
- STD-03: delete the trajectory/track-similarity part (none remain).
- STD-06: remove "Kalman, restriction and similarity search" from the remaining list (similarity search already had it).
- STD-07: trajectory remaining: Hampel/outlier `flags` are outlier flags (not validity), others done as aliases; complete in wave 2.
- STD-04: trajectory per-row status words documented; closest-approach still has ad hoc `entryOverflow`/`hitOverflow` next to the port word.

## 6. Noticed
- `GPUTrajectoryThinning` validation messages name `output.mask` even when the caller passed `output.validity`.
- Roadmap STD-06 text claimed similarity search had no spatialContext; it has.
