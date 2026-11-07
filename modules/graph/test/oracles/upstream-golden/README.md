# Pinned primary reference fixtures

These files are **unmodified upstream regression inputs and golden expected outputs**. The
Circuitscape Julia and PDAL M3C2 binaries were not executed locally. `manifest.json` records the
exact Git commit, source path, download URL and SHA-256 for every retained file. The accompanying
node test verifies byte integrity and checks that derived JSON expected values are copied directly
from the golden columns, rather than produced by the GPU or an independent formula.

- Circuitscape.jl tree `4e0d29d3a7f2ed542a9f2ade782ea554c54b4ae7` identifies package version 6.1.0.
  Its `sgNetworkVerify2` configuration is a legacy fixture declaring version 3.5.5. The precise
  historical binary used to create the golden files is not encoded; we claim comparison with the
  current pinned upstream regression fixtures, not freshly executed Julia 6.1.0 outputs. Six
  source/sink pairs test the seven-node branched network's voltages, absolute branch currents and
  effective resistance derived from the golden voltage difference for a unit source. MIT license
  and copyright are retained in [Circuitscape.jl/LICENSE.md](Circuitscape.jl/LICENSE.md).
- PDAL tree `0e64af327cba0a4c54e501db0d61b62390b69a79` contains the `verifyPoints` M3C2 test,
  two small LAS epochs (829 and 687 points), 103 core results and their parameters. All 91 golden
  rows with at least two samples from both epochs are compared. The other 12 rows have no reported
  reference solution and are explicitly listed, because the GPU sample-variance contract requires
  at least two samples. The upstream copyright and BSD terms are retained in
  [PDAL/LICENSE.txt](PDAL/LICENSE.txt); upstream filter/test sources document the precise conventions.

Run `../upstream-golden-references.py` with NumPy 2.2.6 to regenerate the derived JSON fixtures.
It uses a small LAS decoder and NumPy eigendecomposition solely to prepare caller-supplied normals
at PDAL's normal radius. Expected changes, uncertainty, roughness and counts come from the upstream
text file. Coordinates are recentered before f32 upload. The first core is run separately with the
first source point omitted to match PDAL's explicit first-point exclusion. PDAL reports population
roughness, whereas the GPU reports sample roughness; the test converts this convention explicitly
when comparing roughness and the detection limit. These differences are not silently normalized
into a claim of identical APIs or full M3C2 pipeline parity.
