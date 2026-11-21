# Spatial weights helper conventions

The internal weights stages use the same mathematical conventions as their public contributors.
The focused `weights-helpers-audit` Node and headless specs compare their results with independent
NumPy/SciPy definitions or explicit small closed forms. GPU execution is a separate qualification;
CPU interpretation of emitted WGSL does not certify GPU compilation or cooperative barriers.

Island fallback selects the closest finite, distinct row by Euclidean distance, breaking equal
distances by the lowest row index. `unresolved` selects queries, not representative candidates.
Unselected attachment entries are preserved. The optional uncertainty count is the number of
selected queries entering fallback, including when that number is zero. Conservative dispatch
counts are the Cartesian product of query and representative batches; unused batches do no work.

Key pairs sort lexicographically by `(high, low)`, with equal keys retaining original item order.
The pair `(0xffffffff, 0xffffffff)` is invalid and sorts last; every invalid item gets its own
group. Group indices are one-based inclusive run counts. Group boundaries contain each run's
start plus the final item count; entries beyond that defined prefix are unspecified. Narrowed
sort keys are valid only when every valid half fits below the narrowed all-ones sentinel.

Slot grouping uses a stable column-key sort. The unused key equals the number of columns and is
excluded from the final grouped-slot count. Repeated keys preserve slot order; empty columns
have repeated offsets. Unlike key-pair grouping, unused slot keys form one excluded tail.

Statistics binding preflight traverses graph-view metadata, including nested chunks and status
ports, while ignoring typed-array contents and breaking object cycles. Storage limits include
the aligned prefix of each view. A physical buffer must be imported once and shared through
logical subviews; importing it under two graph handles is refused.

Cooperative statistics dispatchers admit whole workgroups to barriers. Their optional control
uniform must contain at least 64 uint32 words, start at an aligned binding boundary, belong to
the graph, fit the device's uniform binding limit, and use a different physical buffer from
writable storage outputs. Use a 256-byte-aligned 64-word view when packing control records.
An unaligned prefix cannot be interpreted as part of the control record; short or oversized
bindings are refused before compilation. No-control dispatches retain their previous graph.

Count arithmetic preserves integer count bits above 2^24 until the final nearest/even binary32
rounding, rather than rounding the count first. The audit includes division by a count,
multiplication by a count, a ratio of counts and division of a count by a float; its Python oracle
compares exact rational distances between adjacent binary32 values. Underflow, overflow, signs
and odd counts near the uint32 limit are asserted as literal bit patterns. The Node interpreter
cannot faithfully execute the large-integer vector/struct path; the headless test covers it directly.

Population count resolution preserves the supplied graph view: `includedCount` is the preferred
spelling and `count` is its existing compatibility alias. Supplying both is ambiguous and refused,
even if they refer to the same view. Omitting both requests no population-count output.

Row dispatch adds the dispatch's global row base exactly once. `WEIGHTS_ROW_START` identifies
the first CSR row of the bound chunk; `CAPACITY` is the logical neighbor length after subtracting
the aligned binding prefix. Prefix words and unused backing-buffer tails do not become neighbors.
Output rows outside the requested dispatch remain untouched.

Writable ports must always have disjoint logical bytes. Ports written together also need disjoint
aligned binding ranges. Ports written in separate dispatches may share an aligned range when
their logical bytes are disjoint. Separate imports of the same physical buffer are refused.

CSR stages count complete rows, scan their counts, clamp offsets to capacity, then emit the
retained prefix. `requiredCount` remains the full requirement. An initializing stage resets
overflow to zero or one; a later stage may raise an inherited flag but never clear it. Merge
visitors implement ordinary union/intersection/difference/symmetric difference of listed IDs,
including zero-weight slots. The optional row sort reorders only neighbor IDs; all other
attributes must be constant within the row, as for binary higher-order frontiers.

Loss aggregation treats every nonzero source as true and ORs matrix/chunk sources in batches
of at most seven read-only words plus one writable result. Source identities are deduplicated
separately for retained-slot and candidate loss. `merge: true` retains an operation's existing
output flag; omit that output from the input list. A read-only loss source sharing the writable
output's physical buffer is refused, because WebGPU cannot use both access modes in one pass.
Without merge, a new encoding resets the result; an empty merged input leaves it untouched.

Row-local algebra uses caller row identities unchanged. A subgraph removes masked focus rows and
masked neighbors; self-weight insertion replaces an existing diagonal instead of duplicating it,
with distance zero. Per-row self weights are indexed by global row even for chunked CSR. Binary
rules apply their combination only where both operands list a neighbor; one-sided neighbors keep
their operand's weight, except `binary`, which always writes one. Distances prefer the left slot
when present. Chunk capacities and requirements are local to each chunk; matrix loss is their OR.

Space-time weights use the temporal source period `t - lag`, dropping periods outside the range
without wrapping. Period-major output is the sum of temporal/spatial Kronecker products;
cell-major output is its simultaneous row/column permutation. For an `includeSelf` lag, replace
the spatial diagonal by the unit identity (including spatial islands), rather than adding to an
existing diagonal. Other lags preserve the supplied diagonal. This capability is packed-CSR only;
the chunked row-local wrapper has no space-time form. Distance outputs are unsupported.

Suppression handles `output`, `statistics`, `summary`, `counts`, `cardinality`, and
`permutationCount`. Any nonzero matrix or chunk loss makes the logical float prefix NaN and
integer prefix zero. Entries beyond the logical limit are preserved. A complete frame leaves
producer values untouched. Fields passed together must be co-bindable; callers with packed
fields used in separate dispatches suppress those fields separately.

Cooperative kernel guards exclude every padded workgroup and all workgroups when the logical
invocation count is zero. The guard is uniform within a workgroup and executes before any barrier
or caller body. Independent-row dispatches retain their ordinary per-invocation guard.

Weights contributors capture their declared props, nested port maps, topology metadata,
provenance and coordinate tuples at construction. Later caller changes to those maps do not
retarget a validated contributor. Graph views retain their identities: updating their underlying
buffer contents between encodings remains supported. Coordinate-frame getters are read once;
undeclared object properties are not copied into validation bindings. Validation's generic
`props.weights` and `outputWeights` preserve the input subtype's fields by identity, including
chunk-weight fields and inherited getter values; lowering uses a separate fixed-field snapshot.
The captured output status getter is read once, and named validators capture their enclosing
input/output maps once as well. The validation output remains a one-word `invalidCount`.

Packed transpose refuses destination status ports and distances without source distances. It
inherits input loss while omitting input count/requiredCount, which may change when columns are
dropped. The terminal output offset is the retained count. Summary and transpose writable fields
must have disjoint logical bytes; fields bound together must also have disjoint aligned ranges.
The disjoint counts/cardinality layout used in separate summary dispatches remains supported.
All matrix/chunk status fields, including `count`, are preserved against output overlap. Rectangular
transpose clears double-standardization metadata when filtering can remove mass. Binary transforms
can share a buffer with disjoint unbound distance views; actual read/write bindings require physical
separation. Chunk alias checks group physical allocations once and sort logical ranges within each
group, avoiding all-pairs scans across separately allocated chunks.

Every kernel transform publishes unknown symmetry because numerical input symmetry does not
guarantee reciprocal stored slots or distances. Double standardization detects an unrepresentable
scale from `frexp` mantissas and integer exponents before dividing. Its exceptional path uses
bounded mantissa division and integer exponent scaling, then multiplies by the target. For large
symmetrization operands, integer exponent subtraction halves each positive normal word before
addition. These paths avoid overflowing intermediates and cannot be factored into the original
overflowing expressions. Ordinary arithmetic is retained outside those exceptional ranges.

Capacity growth caps optional headroom at `0xffffffff`. A larger safe-integer requirement sets
`limited: true` and retains the exact requirement in the plan; use chunked CSR for such matrices.
Contiguity's derived pair growth uses the same uint32 cap, exposing its uncapped estimate and
`limited` flag so the returned pair capacity can be passed back to the constructor.
