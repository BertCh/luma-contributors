# Terrain status and count evidence

Status, count and validity assertions use their own expected values. A numerical interpolation
reference does not establish an overflow flag or a stopping decision. Terrain status oracles compare
complete uint32 words exactly, poison caller-owned output buffers, and re-encode inputs through the
same compiled contributor so stale nonzero values must clear.

The focused suites below add evidence where a named test observed only one pixel, only a fresh zero,
or a stopping flag without a scheduling-independent reason. Their expected numbers are literal
hand derivations in the fixture comments. CPU operations use those same flag/count literals where a
production twin exists; GPU-only contributors have no invented CPU twin. Headless execution of
these assertions is a separate WebGPU qualification step.

| Contributor | Independent fixture and expected status |
| --- | --- |
| Geomorphons | A 5×5 raster with a masked centre has eight valid interior pixels; the excluded outer ring and centre are zero. Nonfinite scale clears all prior ones. |
| Profile peaks | Rows 2 and 7 are isolated positive-prominence maxima. With suppression radius zero, one round decides both; zero rounds leave both undecided. Capacity one stores one of two peaks and overflows, while capacity three stores two without overflow. Candidate incompleteness is independent of final capacity overflow. |
| Breach | On `[2,5,0]`, pit 0 has the outlet path `0→1→2` with middle height lowered to 1.5: status 1, unresolved count 0. `[2,5,2]` has no lower original outlet within two steps: status 2, count 1. An out-of-grid pit has status 0, count 1. With both pits `[0,2]`, `[2,5,2]` gives statuses `[2,2]` and count 2; replaying `[2,0,2]` gives `[1,1]` and count 0, excluding a Boolean count implementation. |
| Cumulative viewshed | Two identical observers at `x=0,z=1` face a wall at `x=6,z=0.5`. Target-plane clearance is `1/3` and `1/4` at columns 4/5, `7/10` at target 6, and `j/12-1` at 7–11. With tolerance 0.5 m, columns 4/5 and 7–11 are marginal twice; target 6 is visible twice. Even without the wall, flat-ground clearance `1/(j-1)` makes columns 4–11 marginal. A separate flat, zero-tolerance replay clears every marginal count. Exact rational geometry in `terrain-analysis-status-oracles.oracle.py` supplies all twelve visible/marginal count literals; both march and pyramid are checked. |
| Horizon | In a 3×3 raster with only the centre missing, the full validity mask is `[1,1,1,1,0,1,1,1,1]`. Missing neighbours remove samples, rather than invalidating a finite centre. An all-missing replay clears every previous one. March and sweep are checked. |
| Spike repair | With one edge joining initial labels 0 and 1, round one must change label 1. A one-round cap therefore gives convergence 0, limit 1, heights `+Infinity` and usability validity 0 at both valid input cells. Only the labels carry `0xffffffff`. A height gap above `jump` removes that edge, making the initial labels stable: convergence 1, limit 0 and ordinary validity 1. |
| Stream network | The two disjoint reaches `0→1` and `3→4` need two reach slots and four vertex slots. One reach slot stores one reach/two vertices with overflow 1. Two reach slots with three vertex slots also store only one complete reach, giving `(reachCount,vertexCount,overflow)=(1,2,1)`; four vertices give `(2,4,0)`. Removing every stream clears counts and overflow. |
| Stream order | A rootless `0↔1` receiver cycle gives convergence 0; replacing it with outlets gives 1. The default Euler-tour route has no capped pull iteration, so its limit word must be written as zero even from poisoned storage. |
| Tiled flow | A `0↔1` cycle cannot publish either cell, while independent outlet 2 must publish in the first resident sweep. At budget one the progressing resident solver reports convergence 0 and limit 1; all outlets give 1/0. The exact boundary D8 route uses graph ranks and reports limit 0. Auto/resident D8, boundary D8, and auto/resident branching are checked. |
| Watersheds | On `0→1→2`, the first pointer round must change pointer 0 from 1 to 2. Labels already identify terminal 2, but no unchanged confirmation has occurred: convergence 0, limit 1, count 1. Three independent outlets confirm convergence in one round: 1/0/1. The cycle `0↔1` plus outlet 2 at budget two gives labels `[0xfffffffe,0xfffffffe,2]` and flags/count `(0,0,2)` despite apparent self-pointers. ESRI, Whitebox and TauDEM cases also test that a pour point at 0 legitimately cuts the cycle. |

For a singleton terrain flow, all fill, flat and accumulation dependencies are already stable.
The GPU counts one fill confirmation, three separate flat confirmations and one Euler-tour
confirmation. With explicit positive-epsilon fill and flat budgets, the CPU twin counts one fill
and three flat confirmations; its exact donor queue counts zero relaxation rounds. These counters describe
executed algorithm rounds, so their CPU/GPU values need not coincide across different algorithms.

The 3×3 plateau of tens additionally checks accumulation with its upstream flat status omitted:
a flat budget of one gives accumulation convergence 0 and nine NaNs; two gives convergence 1 and
`[1,1,1,1,1,2,1,1,1]` (only centre cell 4 drains to tie-first east neighbour 5).
A 3×3 bowl with rim 2, centre 0 and epsilon 0.25 starts the centre at infinity. The first fill
round changes it to 2.25; budget one gives fill convergence/count `0/1` and suppresses filled
elevation and accumulation. Budget two confirms `1/2` and publishes centre 2.25. These cases run
on both operation forms and directly bound GPU contributors, including replay through stable inputs.

Further hydrology cases distinguish whole-stage completion from locally stable pointers or fields:

- An unresolved watershed direction clears convergence at default and explicit budgets; a pour
  point at that cell restores convergence. An invalid neighbour remains the independent NONE code.
- Positive-epsilon corridors test right/left/down/up sweeps. Unit rise per escape step independently
  gives the complete interior sequence `0,1,…`; the 1616×3 rightward case must complete at cap 99,
  and 1664-cell reflected/rotated cases distinguish the other sweep phases at caps 100/101/102.
- Flat routing with its fill flag omitted must report convergence 0 and nine unresolved directions
  when fill cap one truncates the bowl; at cap two, centre direction 1 and convergence 1 return.
- D8 HAND cycle/stream replays assert unresolved drainage versus a genuine stream stop in all three
  encodings, with and without the drainage-cell output. Heights `[5,3]` give exact drops `[0,-2]`
  once stream 0 intercepts the cycle. No HAND CPU operation twin exists.
- GPU iteration counts are bounded directly by their own stage budgets. The one-tile 5×5 epsilon
  bowl requires exactly two rounds (one change, one confirmation), independently of CPU counts.

The published spike-repair record and formula card also have a Node regression that renders the
owning lens view and requires validity 0 beside `+Infinity`, reserving `0xffffffff` for labels:
[published lens assertion](https://github.com/visgl/luma-contributors/blob/master/modules/lenses/test/computation-lenses/terrain-spike-repair-status-oracles.node.spec.ts).

A separate visibility frame raises the wall at column 6 to 2 m and places observers at columns
0 and 10. With tolerance 0.5 m, target 10 gives `(visible,marginal)=(1,0)` and target 4 gives
`(0,1)`. Each has a hidden ray, so marginal count cannot be computed as `2-visibleCount`.
The rational oracle derives all 24 count words for this frame on march and pyramid.

The Node suites execute CPU operations against independent literals. The GPU operation suites
also construct the directly bound contributor from its declared ports, poison those outputs, replay
input frames, and separately exercise mixed graph execution. They do not use CPU output as a GPU
expectation.

The focused files are:

- [Processing status oracles](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing-status-oracles.spec.ts)
  and [CPU processing oracles](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing-status-oracles.node.spec.ts).
- [Hydrology status oracles](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-terrain/hydrology/terrain-hydrology-status-oracles.spec.ts)
  and [CPU hydrology oracles](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-terrain/hydrology/terrain-hydrology-status-oracles.node.spec.ts).
- [Visibility and horizon validity](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-terrain/terrain-analysis/terrain-analysis-status-oracles.spec.ts).
- [Exact rational visibility oracle](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-terrain/terrain-analysis/terrain-analysis-status-oracles.oracle.py)
  derives the complete tolerance-band count arrays independently of shader code.
- [Spike-repair stopping and validity](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-terrain/terrain-decode/terrain-decode-status-oracles.spec.ts).
- [Profile-peak status oracles](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-terrain/terrain-features/profile-peaks-status-oracles.spec.ts)
  and [CPU profile-peak oracles](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-terrain/terrain-features/profile-peaks-status-oracles.node.spec.ts).
- [Geomorphon validity](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-terrain/geomorphons/geomorphons-status-oracles.spec.ts).
- [Branching-flow status oracles](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-terrain/terrain-processing/branching-flow-status-oracles.spec.ts)
  and [CPU branching-flow oracles](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-terrain/terrain-processing/branching-flow-status-oracles.node.spec.ts).
