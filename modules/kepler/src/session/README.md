# Arrow stream publication

`KeplerArrowStreamPublisher` admits original Arrow record batches into an existing
`KeplerComputeSession`. Every admitted batch is normalized explicitly and uploaded
once. Published dataset revisions contain borrowed aggregate views of the preserved
batch allocations. Appending never packs batches or uploads previous rows again.
Empty batches remain visible and source row offsets are cumulative original counts.

```ts
const stream = new KeplerArrowStreamPublisher({
  session,
  id: 'observations',
  normalization: {float64: {mode: 'exact'}},
  maximumQueuedBytes: 16 * 1024 * 1024,
  maximumQueuedBatches: 2,
  maximumBatchRows: 250_000,
  dictionaryEvolution: 'append-unordered'
});

// Pulls at most one batch ahead and finishes only after the iterator ends.
const complete = await stream.consume(recordBatches);
const explorer = session.openExplorer(complete.id, {
  workingMemoryBudgetBytes: 32 * 1024 * 1024,
  dimensions: []
});
```

For push producers, `enqueue(batch)` returns that batch's publication promise.
Admission beyond either queue limit throws `KeplerStreamBackpressureError` without
retaining the rejected batch. Await the shared `stream.ready` wake-up signal and
retry explicitly. A batch that alone exceeds the byte or row capacity is rejected
permanently, rather than waiting for capacity it can never obtain. The byte limit
includes the active batch and counts retained source backing-buffer capacities;
shared backing allocations across batches are conservatively counted per batch.

The source schema, field metadata, nullability, and selected normalization policy
are fixed. Unordered dictionaries may append labels while retaining every earlier
code. `dictionaryEvolution: 'fixed'` rejects new labels after the first batch.
Ordered dictionary labels/order always remain fixed. Earlier revision metadata is
not mutated. A normalization/upload failure rejects pending admissions and marks
the already published prefix `failed`.

Every publication carries `population: {kind: 'stream', streamId, state}`. Registry
leases, analysis requests, and session explorers reject unfinished populations by
default. A prefix preview requires `openExplorer(...,
{allowIncompletePopulation: true, ...})` or an explicitly permitted registry lease.
Appending invalidates old preview handles; callers create a preview for the new
revision and fence the previous handle before closing it. The old GPU buffers stay
alive through leases and are also referenced by the current prefix. Finishing
publishes a `complete` revision without uploading or allocating source data.

`finish()` stops further admission, drains accepted batches in order, then publishes
completion. `cancel()` rejects queued admissions and retains an explicitly cancelled
prefix; it never silently declares that prefix complete or frees a renderer's
buffers. `dispose(lastSubmissionFence)` removes the publisher's own dataset and
releases resources only after that fence and existing leases settle. Replacement by
an unrelated publisher is detected and never overwritten or removed.

CPU admission covers retained source/normalized typed storage; GPU admission covers
actual source buffers including empty allocations. JavaScript object/string overhead
and temporary normalization allocations are excluded, as for ordinary session
imports. The explicit row capacity bounds normalization input size independently of
compact Arrow input formats. There is no GPU source readback in the publisher.

Device loss stops queued work. The registry retains the original Arrow source plus
stream completeness and normalization metadata. Completed streams recover through `session.recoverDevice` using retained prepared
chunks, preserving dictionary codes, batch boundaries, exact time storage and the
complete population marker. Unfinished, cancelled and failed streams are skipped:
an application must explicitly reimport/rebind the retained source and decide
whether a prefix may continue. Recovery never labels a partial stream complete
implicitly.

After a host successfully attaches the completed snapshot, `detachCompleted()`
transfers cleanup ownership to that host without changing the registry revision or
uploading buffers. Disposing the detached publisher cannot remove the host's dataset
or a replacement-device recovery revision. Failed attachment must finish its own
fenced consumer rollback before publisher disposal.

The Kepler example's **Prepare data → Stream Arrow IPC** workflow reads a sequential
IPC stream from `File.stream()` with a bounded decoder input, one admitted batch per
pull, a first-batch pause and a bounded original-source preview. It rejects the
random-access Arrow file format instead of silently materializing it. The user
explicitly finishes at end of input, then the host attaches the completed snapshot
without reimporting it. Unfinished previews remain outside native analysis/export.
