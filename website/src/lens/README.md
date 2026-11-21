# Generating lens website artifacts

From `website/`, run `node --experimental-strip-types src/lens/generate-view-models.ts`.
Without scenario arguments this replaces the complete model set and removes obsolete JSON models.
With scenario IDs it replaces only the selected models, retaining the other model files byte for
byte. Both modes refresh the full summary and lenses page metadata from the proposed complete set.
A selected update therefore requires existing models for every source record; run full generation
first if that set is incomplete.

Before replacing any artifact, generation decodes the requested recordings, checks each envelope's
scenario ID against its filename, checks that every requested ID exists, parses retained models,
and calculates page metadata with all required featured scenarios. Rejected input preserves all
previous artifacts. Correct the named source, filename, selection or retained model and retry.
This preflight guarantee does not cover disk-write failure or process termination during publication.

Both entry points require equal source/model filename sets and compare each retained model's parsed
content with fresh extraction from its source. ID-only and stale models are rejected with a
regeneration remedy; harmless JSON whitespace or object property order can remain unchanged.
Model generation reads each source once, then derives its model, counts and UTF-8 size from that
same cached snapshot, so a later file version cannot supply contradictory metadata in the run.

Histogram bins and median kernel counts use original compute commands before display grouping.
Display groups can contain both compute and copy commands. A model's `recordedNodeCount` counts
all graph commands and is displayed as graph nodes. File sizes and summary byte totals use UTF-8
bytes, including on selected updates.
The histogram has an explicit zero-compute bin for copy-only recordings. Empty chart distributions
render zero-height bars, and the median still counts original compute commands.

For point captures with a published active count, the map withholds the point layer when the count
exceeds the declared key-space size. If that size is absent, exact logical buffer bytes in the
capture's own graph establish capacity (four bytes per scalar component). Estimated or lower-bound
byte sizes do not establish a maximum. The captured window still limits the rows actually drawn.

The summary's `inventoryKnown` is false when any graph omitted its resource inventory. In that
case the numeric count and byte fields sum only the known inventories, their aggregate byte
precision is `unavailable`, and the page displays unavailable inventory and size. Explicit empty
inventories support an exact zero. This preserves the distinction between missing and empty data.
The summary fields are `resourceCount`, `resourceBytes` and `resourceBytesKind`: they include
buffers, textures and external textures. Counts are labeled resources in shared graph and workflow
displays too. If any included resource has unavailable bytes, the total is unavailable even when
another legacy resource has unknown precision; unknown precision cannot make an unavailable size
appear measured.

`node --experimental-strip-types src/components/meta/pages/lenses/generate-data.ts` refreshes only
page metadata from existing models and decoded sources. It uses the same metadata calculation and
identity checks as model generation and validates before writing.

Regression checks run in the ordinary Node project:

```
node_modules/.bin/vitest run --project node --maxWorkers=2 modules/lenses/test/computation-lenses/website-generated-metadata.node.spec.ts modules/lenses/test/computation-lenses/website-view-model-parity.node.spec.ts modules/lenses/test/computation-lenses/website-graph-count-labels.node.spec.ts
```

Run that command from the repository root. The generator tests use the real decoder, extractor
and metadata calculation in an in-memory filesystem; they do not replace committed artifacts.
