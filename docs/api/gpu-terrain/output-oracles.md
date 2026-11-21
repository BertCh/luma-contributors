# Terrain output oracles

Texture publications are caller-owned outputs. Requesting a texture without its corresponding
buffer still computes and publishes channel zero; requesting both publishes the same defined
quantity to each. The texture-only tests use poisoned storage and independently expected values
on a nine-cell flat raster, including the last invocation of a partial workgroup.

| Output | Flat-fixture expectation |
| --- | --- |
| Horizon solid-angle sky view | 1 |
| Relief hillshade, one 30-degree light | 0.5 |
| Relief with exposure 0.5 | 0.25 |
| Solar hours, one two-hour daylight sample | 2 hours |
| Solar energy, DNI 100 and diffuse 10 at 30 degrees for two hours | 120 Wh/m² |
| Shadow illumination, ambient 0.2 and sun 0.6 at 30 degrees | 0.5 |
| Cast-shadow visibility, sun above flat ground | 1 |

These are mathematical expectations, not captured GPU results. GPU qualification is recorded
separately by the local gate; adding a spec does not establish that the GPU executed it correctly.

## Counts and status

Summit `candidateOverflow` describes loss before final publication. An exact summit mask has
no candidate loss, so this flag is zero even when a zero-capacity final ID list reports
`count=0`, `requiredCount=1`, and `overflow=1`. Do not substitute final-capacity overflow for it.

Iteration counts describe executed GPU rounds, including the unchanged confirming round.
On a one-cell fixpoint, relaxation fill with positive epsilon, downstream pointer jumping and
Euler-tour ranking each count one round. Flat resolution counts three rounds because it confirms
three separate relaxations. CPU operation forms normally report zero for these GPU-specific
diagnostics. With an explicit watershed `maximumIterations`, however, the CPU form counts
synchronous pointer-jumping rounds, including confirmation. For `[1,1,0]` with budget one,
pointers `[1,2,2]` become `[2,2,2]`: labels are all 2, but convergence is 0, limit reached is 1,
and iteration count is 1. Budget two permits the unchanged confirmation and reports `[1,0,2]`.
Requesting only filled elevation and flat diagnostics still runs flat resolution: the singleton
at elevation 7 reports convergence 1 and three GPU rounds even without a routing raster output.
An explicit CPU flat budget also counts those three confirming stages. On a 3×3 raster of tens
with zero fill epsilon, the lower stage changes the center from infinity to 2 in its first round.
Budget one therefore reports flat convergence 0 and count 3 in both operation forms; budget two
permits the lower confirmation and reports convergence 1 and count 4. Incomplete flats suppress
directions, classes, accumulation and streams with their unresolved values, while filled elevation
stays usable. CPU flat stages use frozen tile halos, so cross-tile GPU scheduling can affect which
flat budget suffices. Without an explicit flat budget the CPU retains exact complete traversals.

An explicit watershed budget is in `[1,1024]` on both operation forms and the CPU function.
Each explicitly budgeted GPU round copies its input pointers before updating them. The chain
`[1,1,1,0]` at budget one thus gives `[UNRESOLVED,3,3,3]` and status `[0,1,1]` on both forms,
independent of invocation order. The default GPU route retains its in-place jumping schedule.

Either flat diagnostic can be selected alone, including an explicitly undefined companion key.
Flat convergence and count share one finalizer dispatch, so pooled output words must occupy
disjoint storage binding windows: offsets 0 and 4 are rejected; offsets 0 and 256 are accepted.
The pooled singleton oracle poisons both words with 123 and requires `[1,3]`, retaining padding.
Output selection and compile-time options are captured at construction; later caller mutation
does not change lowering.

Point horizon `samples` counts evaluated lattice samples, separately for each ray. Declaring
`minimumDistance=maximumDistance=1` gives the singleton lattice `{1}`; an interior eye has exactly
one evaluated sample in each cardinal direction in both march and pyramid traversals. Ground 10
and eye height 2 give tangent -2 at that sample without curvature. On a rectangular 3-by-4 grid,
flow-fraction lengths are `[3,5,4,5,3,5,4,5]` in the east-first clockwise direction order.

## Secondary values and operation forms

The scalar oracle specs use these independently derived fixtures:

- A radius-one ring of `z=-(x²+y²)` has four differences of one and four of two; gain two and
  z factor three give curvature nine. Ring windows crossing the raster edge are invalid.
- Eye `(0,1)`, wall `(5,2)`, target `(10,0)` give line-of-sight clearance -3 metres. Clearance
  is a vertical margin at the target, not the slope difference alone.
- The cumulative-view fixture has eyes at height 1, a wall of height 0.5 at column 6 and a target
  at column 10. The wall rises 1/10 metre above the sight line, projecting to 1/6 metre at the
  target. Each observer contributes one marginal count under a 0.5 metre tolerance and zero
  visible counts there. The unobstructed target at column 2 has visibility code 1 in the
  line-of-sight fixture and visible count 2 in the cumulative fixture, with optional outputs
  present and absent for both march and pyramid traversals.
- Relief color with tint disabled is elevation color multiplied by relief, packed in RGBA order;
  the flat fixture gives linear RGB `[0.2,0.1,0.05]` with opaque alpha. Color, relief and hillshade
  are tested separately as well as together.

Operation fixtures compare named outputs directly with independent expectations in both forms.
The CPU operations execute the same fixture assertions in Node without a GPU. Coverage includes
parabola refinement and capacity-one peak counts; all five branching-flow recurrences; breach
depths, masks and complete paths; merge-tree birth/spill levels, ownership and stage curves;
catchment hypsometry and profile status; fill and drainage convergence/counts; peak prominence;
and the committed published-NOAA solar-position reference. These assertions do not obtain their
expected values by running the CPU twin. The texture-shade publication uses committed SciPy
convolution values with the same numeric budget as its buffer-reference test.

The breach fixture `[2,5,0]` with decrement 0.5 requires a cut of `5-1.5=3.5`.
A cut budget of four resolves it; a budget of three leaves elevations unchanged, publishes
no path (`pathLengths=0`, path cells are `0xffffffff`), `pitStatus=2` and `unresolvedCount=1`.
Solar-position fixtures include NOAA row 27 at night alongside daylight rows, including each
single-output selection. Azimuth must be finite and in `[0,360)` before circular comparison;
an otherwise equivalent angle plus 360 is outside the output contract.
