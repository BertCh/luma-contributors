# Flight matrix pair and route-record weights

## Finding and source meaning

The flight matrix now switches between two measures on the same undirected airport-pair CSR:

- **Airport pairs** answers how many connected airport pairs share a matrix cell.
- **Airline-route records** sums the bundled OpenFlights `count` field for those pairs. The source build describes this field as airline-route records on each pair across both directions. It is not a count of flight departures, timetable frequency, seats, passengers or distinct airlines. Distinct airlines are stored in the separate `airlineCount` field.

The source is the OpenFlights community airport and routes dataset, frozen around 2014; see the [official OpenFlights data page](https://openflights.org/data.php) and the bundled dataset build notes in `examples/deck/spatial-analysis-showcase/scripts/data/openflights/README.md`. The already aggregated pair table cannot recover direction. Both matrix measures, and the existing map/coarsening, retain the undirected interpretation. Pair-topology chance comparisons are disabled for route-record weights.

## Numerical reference

`examples/deck/spatial-analysis-showcase/src/scenes/flows/flight-matrix-weight-reference.mjs` independently reads the shipped binary columns, reconstructs symmetric CSR slots, assigns each slot the pair's `count` value, and bins with scalar CPU arithmetic. It checks the source lengths, doubled slot count, exact integer weight sum and `uint32` bound. Its cases cover input and degree orders at 128 and 256 bins plus a centered zoom window. The class counts use the same published integer breaks (`1`, `2`, `3–5`, `6–10`, `11–20`, `21+`) so the browser check can compare their displayed labels at the published display precision.

Reference execution on the checked-in bundle produced 18,930 pairs over 3,257 airports, 66,933 route records before symmetric expansion and 133,866 after expansion. The maximum full-matrix route-record cell is 3,102 for degree order at 128 bins and 938 at 256 bins; input order gives maxima 100 and 76 respectively. SHA-256 hashes of the five input columns are emitted by the reference script for reproducibility.

The first zoom oracle incorrectly clipped to the requested 271 positions, yielding 58 pair slots and 108 record units. The published whole-cell layout instead uses three airports per cell and draws 91 cells spanning 273 positions; the 128 compiled cells span 384 positions. The corrected scalar reference bins the compiled support, sums only visible cells, and computes the maximum over all compiled bins as the readout's help specifies. Its visible totals are 60 pair slots and 112 route-record units, with 59 occupied cells and global maxima 2 and 5. A 271/128 known-result fixture checks the 3/91/273/384 layout. The failed zoom report is preserved; this is an input-support correction, not a tolerance adjustment.

## Implementation and checks

`GPUAdjacencyMatrix` receives the exact per-pair route-record integer in both CSR directions, with `weightScale: 1`. Before graph construction the scene checks integer `uint16` source values and verifies that the actual expanded global sum is at most `2^32 - 1`; because every cell is a subset of that sum, no cell atomic can wrap. Both `weightSums` and `maximumWeightSum` are real GPU outputs: the selected raster reads the chosen count buffer, the legend and cell histogram classify that same measure, and the readouts and matrix tooltip report selected values. The full-window symmetric total visibly makes the two-slot conservation convention explicit. Readbacks are revision checked against the active matrix, order, window and measure.

The browser harness is `examples/deck/spatial-analysis-showcase/scripts/check-flight-matrix-weights-browser.mjs`. It launches the repository WebGPU Playwright configuration, changes the All controls options, waits for ready UI and generated readouts/charts, then compares GPU totals, maxima, occupied cells and the six visible class bins with the independent reference. Its output records pass status, failure reason and console errors separately and closes the browser in `finally`.

**Current checks:** CPU reference execution, known-result whole-cell fixture, Node syntax checks and showcase TypeScript check pass. Wave 16 bundle `31e545703f8e9c2990ab47eb3c8ec879c9e3532b9a386c16da552c59782e7eaf` passed all ten browser cases with exact integer total/maximum/occupancy checks, including the corrected padded zoom support. Chart-bin labels are compared at displayed precision: counts at least 1,000 intentionally use whole-thousand strings. The accepted report is `/tmp/luma-showcase-overhaul-20261025/movement-wave16/flight.json`; durable hashes and provenance are in `movement-acceptance-final.json`. Earlier failed and partial reports are preserved. Tooltip behavior and exact per-bin readback values are not independently asserted by this harness.
