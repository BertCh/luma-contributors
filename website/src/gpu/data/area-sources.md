# Chicago tract sample

The areal runs use whole Census tracts around the Near West Side of Chicago.
The complete fixture covers 156 tracts, 364 undirected shared-edge adjacencies and 523,303
residents in the 2018–2022 five-year population estimates. Smaller runs take a connected prefix
of the same sample. This geographic window was chosen for illustration and is not a citywide
statistical sample.

## Sources and terms

- Boundaries and GEOIDs come from the U.S. Census Bureau's 2022 Illinois cartographic boundary file
  at 1:500,000 scale (`cb_2022_17_tract_500k`). [Census cartographic boundaries](https://www.census.gov/geographies/mapping-files/2022/geo/carto-boundary-file.html)
- The CDC/ATSDR Social Vulnerability Index 2022 Illinois tract table supplies ACS 2018–2022 estimates
  of total population (`E_TOTPOP`), population aged 65 and older (`E_AGE65`), percent below 150%
  of the poverty threshold (`EP_POV150`), and civilian unemployment percent (`EP_UNEMP`).
  These U.S. government source columns and boundaries are public domain. [SVI downloads](https://atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html) · [2022 data dictionary](https://www.atsdr.cdc.gov/place-health/media/pdfs/2024/10/SVI2022Documentation.pdf)

## Preparation and use

Run the following command at the repository root to reproduce `chicago-tracts.json` from the
checked-in showcase geography and four federal estimate columns.

```sh
node examples/deck/spatial-analysis-showcase/scripts/data/chicago-tracts/extract-card-tracts.mjs
```

The extractor selects populated, single-shell tracts with complete estimates. It starts at the
tract nearest 87.67° W, 41.878° N, growing the window through shared boundary segments in order
of distance from its center. Corner-only contacts do not count toward rook adjacency. Coordinates
use a local equirectangular projection rounded to whole meters. Earcut triangulates the outlines
for rendering. The GPU polygon operations receive the original vertex ordering and ring offsets
in the same map frame as the rendering mesh. The `createTracts` adapter uniformly fits the
selected window and computes neighbor distances in those world units.

`values` scales the poverty percentage to the full sample range (4.4–76.1%). Raw poverty and
unemployment fractions and population counts are available separately. Dot density splits estimated
residents into ages 65+ and under 65. Its default draws one dot per 100 residents. The continuous
sampling demo uses the union of whole tract triangles as its sampling frame. Generated dots,
sampling sites, regression responses and slider-controlled rate scenarios are modeled quantities.

## Chicago employment overlays

Zonal statistics evaluates observed job counts (`C000`) at the internal points of 1,528
Census blocks in the 64-tract window. Block internal points locate geographic reporting
units. They are not employee addresses. Each compact row contains east and north in the
local meter frame, jobs, and the Census crosswalk tract row. Point-in-polygon on the generalized
1:500,000 display geometry agrees with the official tract for 1,527 of 1,528 internal points.
The zonal operation uses the displayed geometry. Raster join uses these same 1,528 workplace
block points and 64 tract polygons. It assigns each block's 2021 job count to the tract at
the center of its raster cell. Changing cell size alters join totals and the boundary-cell
mask. The tract fill shows the logarithmically scaled sum of jobs, and teal dots identify
blocks whose cell touches a tract boundary. The slider draws a deterministic prefix in
hashed block-ID order. The line-length run draws the strongest selected home-to-work
flows between the centroids of the displayed tracts. These straight lines illustrate
commuting demand. Their lengths describe the map geometry, not road travel distance.
[U.S. Census Bureau LODES data](https://lehd.ces.census.gov/data/) ·
[LODES technical documentation](https://lehd.ces.census.gov/doc/help/onthemap/LODESTechDoc.pdf)

Both extracts come from U.S. Census Bureau LEHD LODES8, a public-domain U.S. government
dataset. The workplace counts use Illinois 2021 all-jobs WAC and the LODES Illinois
geography crosswalk. The OD counts use the `chicago-lodes-od` showcase's 2021 Illinois
non-self tract flows. Rebuild both after generating `chicago-tracts.json` using
`python3 website/scripts/data/extract-chicago-workplaces.py <downloads-directory>`.
Use these two compressed input files.

| File | SHA-256 |
| --- | --- |
| [Illinois WAC all jobs 2021](https://lehd.ces.census.gov/data/lodes/LODES8/il/wac/il_wac_S000_JT00_2021.csv.gz) | `3c1375ac4786dc437796e3c0bd2366d5ffe11ee83d217a83d98968d76028acc6` |
| [Illinois LODES8 crosswalk](https://lehd.ces.census.gov/data/lodes/LODES8/il/il_xwalk.csv.gz) | `f9919440f105a27c4b97cd2001b8e3506b59eee34ddd57bf4e6e77ade7f0ebcf` |

## Amsterdam building frontages

Five urban-form operations use Overtoom frontages in Amsterdam Oud-West. The fixture contains 43
whole building shells and roof heights. Each record retains its BAG building identifier.
**© 3DBAG by tudelft3d and 3DGI.** The data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
[3DBAG source and credit requirements](https://docs.3dbag.nl/en/copyright/) ·
[Height attribute definitions](https://docs.3dbag.nl/en/schema/layers/)

To reproduce the extract, save the feature collection below and run
`python3 website/scripts/data/extract-amsterdam-buildings.py <download.json>`.
The bounding box uses EPSG:7415. The extractor keeps simple, single-shell LoD0 footprints with
valid height attributes, translates and rounds vertices to local centimeters, and subtracts the
ground elevation (`b3_h_maaiveld`) from the 70th-percentile roof elevation (`b3_h_dak_70p`).
The resulting height is in meters. [CityJSON feature collection](https://api.3dbag.nl/collections/pand/items?bbox=119345,485905,119455,485980&limit=100)

The map rotates the local RD meter frame to follow Overtoom and scales both horizontal and vertical
dimensions uniformly. Each operation selects 6–24 frontages. Building tessellation selects separate
shells so that boundary sample sites have unambiguous owners. Adjacency uses the surveyed party-wall
geometry and a setback scenario on every nth building. Set the setback to zero to restore all
original outlines. Street profiles sample a digitized straight transect, use the source roof heights
and allow modeled street widening. Isovist viewpoints, visibility sites, and the footprint scale
control are demonstration inputs.
