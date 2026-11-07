# Habitat circuit implementation ledger

## Contract

This scene treats circuit connectivity as an explicitly chosen land-cover resistance scenario.
Circuitscape documents raster four/eight-neighbor choices and resistance-versus-conductance averaging;
it also describes unit-current pairwise solves and voltage/current maps ([options](https://docs.circuitscape.org/Circuitscape.jl/stable/options/), [usage](https://docs.circuitscape.org/Circuitscape.jl/latest/usage/)). This implementation fixes its own precise edge contract:

`g(a,b) = 2 / (R(a) + R(b)) / physicalEdgeLengthMetres`

The raster recipe applies that formula to the original 20 m grid, including `20 * sqrt(2)` m diagonal
edges when eight-neighbor connectivity is selected. A unit injection is a normalized solver scenario,
not a measured ampere value. Effective resistance is reported in relative resistance·metre units;
class weights are analyst scenarios, not calibrated ohms.

The current-density recipe reports `max(sum incoming edge current, sum outgoing edge current) / cell area`.
It is therefore current per square metre for the unit-injection scenario, not a separately normalized
flux-area statistic. Circuit edges are isotropic; the DEM slope term belongs only to the existing
walking-cost scenario. Circuitscape describes alternate-path current and voltage mapping; the
walking-cost layer instead propagates a least-cost surface.

The source and sink are deterministic valid cells selected nearest to local subset targets (3, 8) and
(12, 8), respectively, with row-major tie breaking. They are scenario terminals, not observed origins.
Unknown class codes become barriers. Optional WorldCover permanent water can also be treated as a
barrier; otherwise it receives the existing mild/strong table value.

## Source and reproducibility inputs

- Dataset: `dixie-fire`, ESA WorldCover 2021, CC BY 4.0; source provenance and build notes are in
  `examples/deck/spatial-analysis-showcase/scripts/data/dixie-fire/README.md`.
- Crop: 64 × 64 values read from the actual bundled `worldcover.png` class raster, centered on
  `(-120.9511, 40.1396)` and clamped to the 750 × 750 source WorldCover extent. `startColumn` and
  `startRow` are derived from the dataset UTM frame at load time as `clamp(floor(centerCell - 32),
  0, 750 - 64)`, not synthesized. The implementation ledger records these offsets at runtime via
  that deterministic calculation; the interactive terminal readout gives the exact local vertex
  indices and coordinates for the loaded crop.
- Grid: original 20 m analysis cell size; 64 × 64 crop dimensions; matching WorldCover and DEM
  offsets retained from the existing scene.
- Class mapping (mild): 10→1, 20→1.5, 30→2, 40→3, 50→8, 60→4, 70→5, 80→20, 90→4, 95→2,
  100→3.
- Class mapping (strong): 10→1, 20→2, 30→6, 40→15, 50→80, 60→10, 70→20, 80→120, 90→15,
  95→3, 100→4.
- Main solve: 64 × 64 raster, selected four/eight-neighbor directions, graph source/sink indices
  mapped from the exact central subset terminals, unit injection, configured bounded iteration budget
  and absolute current-conservation tolerance.
- Oracle subset: 16 × 16 center crop at `(rowOffset, columnOffset) = (24, 24)` in the 64 × 64 crop;
  256 vertices. Dense and sparse recipe calls share the same uploaded CPU-assembled edge and
  conductance arrays. This assembly uses actual crop resistance values and the declared direction
  order `[right, down]` or `[right, down, down-right, down-left]`. Boundary placeholders have zero
  conductance. Diagonal length is `20 * sqrt(2)` m.
- Terminal vertex selection: within that 16 × 16 subset, choose the valid cell nearest `(column,
  row) = (3, 8)` as source and nearest `(12, 8)` as sink; squared grid distance and row-major scan
  order break ties. The local vertex IDs are row-major `row * 16 + column`; full-crop IDs add
  `(24, 24)` before `row * 64 + column`. The map marks those exact full-crop cells with gold (source)
  and magenta (sink), and the terminal readout reports IDs and coordinates.

The GPU oracle compares only when the dense and sparse status both report reachable and converged.
Potential/current maximum deltas and energy differences come from actual GPU output readbacks, not a
CPU analytical solve. A solver status that does not converge or cannot reach the terminals withholds
the map and accepted scalar diagnostics. The configured iteration budget is reported as a budget;
the sparse recipe's status does not promise an actual iteration count. Residual diagnostics remain
available only when they are defined for the reported status.

## Verification scope

The authoring pass may run scoped Biome formatting and the showcase package TypeScript check. The
parent planner owns independent NumPy oracles and any actual GPU test run. No browser or GPU run is
claimed by this ledger.

## Planner review before wave 12

Fixed the readback revision offset after adding circuit columns (it was incorrectly reading the final diagnostic word), separated the rebuild controller from the owned compute instance so repeated compile-option changes cannot write destroyed buffers, and gated both potential/current maps on accepted reachable convergence. Generation guards suppress stale readbacks. Matched the potential legend to its cividis map. Independent float64 NumPy PCG on actual WorldCover source row506/column555 covers both16²and64² domains, both4/8neighbor graphs, both resistance tables and water handling; source/sink IDs must be matched against the runtime readout before accepting these oracle comparisons. Runtime and GPU acceptance remain pending.
