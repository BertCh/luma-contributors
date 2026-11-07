# Final inventory and acceptance handoff

Do not execute the source audit below until the integration owner announces final source freeze. The existing copied-input snapshot remains valid for its recorded timestamp; subsequent source fixes are outside its fingerprint. Preparing this runbook does not rerun or replace that snapshot.

From the repository root, after the final freeze and the integration owner's required checks:

```sh
python3 dev-docs/research/showcase-overhaul-2026-10-25/write-source-manifest.py \
  dev-docs/research/showcase-overhaul-2026-10-25/functionality-final-before.json
node --test scripts/showcase-functionality-inventory.test.mjs
node scripts/report-showcase-coverage.mjs --format=json --functionality \
  > dev-docs/research/showcase-overhaul-2026-10-25/functionality-after.json
python3 dev-docs/research/showcase-overhaul-2026-10-25/write-source-manifest.py \
  dev-docs/research/showcase-overhaul-2026-10-25/functionality-final-after.json \
  --compare dev-docs/research/showcase-overhaul-2026-10-25/functionality-final-before.json
python3 dev-docs/research/showcase-overhaul-2026-10-25/reconcile-functionality.py \
  dev-docs/research/showcase-overhaul-2026-10-25/functionality-after.json
```

A mismatch rejects the snapshot. Re-establish the freeze before another attempt; do not certify a scan whose source or dependency fingerprints changed. The utility's documented joint-hash encoding is independent of the earlier copied-input manifest, so compare manifests produced by the same utility rather than assuming the old joint digest uses identical encoding.

The reconciler writes `functionality-concurrent-decisions.json` and its Markdown companion. Before promoting them to the final source-decision inventory, review its source counts and remaining decisions, refresh the 23 manual dispatch anchors against the frozen source, and retain these distinctions:

- A static source call is not GPU execution. Legacy raw fields named `runtimeCovered` remain lexical evidence only.
- A manual or composed path needs its actual scene, control, compile/encode path and configured output. A module import alone earns no execution credit.
- Infrastructure, ownership and equivalent authoring entry points need an explicit scope rationale. No source call is claimed for intentionally uninvoked methods.
- Every option contract remains inventoried. Static literal candidates and dynamic references do not establish an entire union, numeric range, or combination of options.
- Source freeze hashes identify analyzed code. Production bundle hashes identify tested artifacts. Neither substitutes for the other.

## Durable acceptance structure

`acceptance-ledger.json` is the consolidated evidence index; it remains provisional while any final suite is queued, running, failed, partially accepted, or awaiting a display correction. Each suite records:

1. Bounded accepted/planned record counts and precise status; counts are not public API coverage percentages.
2. Tested production bundle SHA256 and before/after equality.
3. Independent reference method and tolerance, with self-consistency, smoke, display and numerical-only evidence labeled separately.
4. Limitations, including unresolved outputs, model assumptions and untested option variants.
5. A reproducible command and required environment where relevant.
6. A durable domain packet, raw report path and report SHA256. Completed summaries remain in the repository; screenshots and large raw logs may remain in `/tmp`.

A passing retry supplements rather than erases an earlier diagnosed failure. Preserve the rejected grid precision baseline, partial geometry results, corrected harness/reference contracts, and explicit unresolved cases. Cases repeated on a later bundle must not be summed as new independent coverage. Promote a suite only after the completed report and bundle evidence agree, and keep numerical acceptance separate from pending visual corrections.

Final source inventory publication and final runtime acceptance are separate decisions. The integration owner performs the required root build, lint, test and website gates; this evidence tooling does not replace them.

## Executed final wave 22 source audit

Before/after source fingerprints match all 4,172 source/analyzer inputs and
174 published stories. Source SHA256:
`7d3648ef59ab450e79eaa37c06749e5f59c832ae8167f8907b3439e3d6ccf24e`.
The inventory fixture passed one test with zero failures/skips/TODOs. All 91 exact source anchors
across 23 manually traced paths match. Callable, method and option counts are unchanged.

The prior wave 21 source snapshot is archived in [source-audit-wave21.tar.gz](source-audit-wave21.tar.gz),
SHA256 `c790eb554ab5823e1da16e7f401bd34e5c12d15662bfd11078ca251656437f9d`.
Its [archive record](source-audit-wave21-archive.json) lists the raw inventory, decisions, manifests
and manual anchors. Earlier snapshots remain preserved.

Manifest scope is module/showcase TypeScript and the two inventory analyzers. Browser verifiers,
CSS, screenshots and reports retain separate production-bundle and domain-packet provenance.
No source reachability count implies successful GPU execution or whole-option correctness.

The [complete wave-21 sweep](atlas-wave21-sweep.json) visited 174 stories and 994 steps, with 173
passing stories and one mobile legend failure. Wave 22 contains narrow mask-display and responsive
map-chart corrections. Their focused rebuilt checks and the full repository test command pass.
[Final integration evidence](integration-final.json) closes the bounded queue, preserves both test-only
fixture diagnoses and nonfatal warnings, and does not claim another full atlas sweep.
