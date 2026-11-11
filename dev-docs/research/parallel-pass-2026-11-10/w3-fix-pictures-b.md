# w3-fix-pictures-b

## Done
- 2.1 map-matching: lines renamed "matched positions, joined in order", thinned (radius 1.2, alpha 0.7), comment corrected. Not drawn as road edges (matchedEdges exists but no road geometry is bound to it here).
- 2.2 terrain-dem-coregistration: label/titles/caption say "simulated change"/"simulated elevation change"; SCENARIOS entry added in website/build/demo-data-sources.ts (departure: file not in my list, one small Edit).
- 2.3 adjacency-matrix(-order): note now "tract links, either direction (1 in 6 cross-band kept)"; caption of adjacency-matrix corrected; SCENARIOS entries for both (thinning kept, disclosed). Crosshair rest moved to [0.66, 0.34] so it no longer covers "band 3".
- 2.4 pointcloud-statistical-filter: CPU estimate preview, candidate buffers and legend entry removed; readout says "release to apply"; caption colours corrected (kept blue, rejected red).
- 2.5 pointcloud-height: probe takes the tallest return within a tenth of the frame; readouts now 50, 61, 2, 0, 45 m (was 0 at all five). Legend "top of the ramp: N m" (theme-neutral).
- 2.6 spectral-unmixing: endmembers now pure pixels of the scene (darkest, highest SWIR minus NIR, highest NIR minus red) instead of hand-set; panels now vary; key says bright or dark by theme. Per-panel stretch not done (one raster layer, one range); range 0 to 1 is stated.
- 2.7 pointcloud-icp: caption rewritten to the drawn colours and to say closest-point ICP approaches the scan gradually. See Noticed.
- 2.8: lon-lat legend inset 48 clear of date line, swatch text "colour per piece"; geometry-orientation ARROW_SCALE 1, shorter legend; computation-metrics y labels left of bars, peer dashed outlines removed; computation-record lifetime rows lowered (area.y1 0.43) clear of the title; point-horizon-profile title moved right, "0° horizon" label dropped (grey guide line named in title); dasymetric ring now dashed (checked: total readable at rest).
- 2.10 encounters: bars titled in all three; clock-encounters (skip 9 tracks) and trajectory-closest-approach (skip 4) now use different tracks (new optional skipTracks in lib/fleet.ts). Clock scene is still small.
- 2.11: normalize-geometry caption, voxel-merge caption and centroid legend entry, hot-spot legend "ramp tops out at +4", track-quality caption says simulated, raster-registration "large dot / dark dot", SCENARIOS line for glcm-texture.
- 2.13: addZoneSegments now lights only edges with both ends inside (&&); legends for network-kernel-density and catchment-accessibility name the lit roads as derived from the input distances/cost matrix.

## Verified
- verify-runs on 26 ids, dark and light: 50 of 50 passed (before the final pointcloud-height and computation-metrics tweaks, each re-run: pass). Warnings: track-quality maximumGap inert, raster-registration upsampleFactor inert (reviewer 2.12: keep), two "differs after return".
- Looked at full-size screenshots (both themes where it mattered) of spectral-unmixing, map-matching, icp, adjacency-matrix, height, lon-lat, orientation, metrics, record, horizon-profile, encounters, dasymetric, dem, kde, catchment.
- check-catalog: 0 errors. tsc website: no errors in my folders. biome check --write on my files.
- Not run: snapshot-runs, card-size views, voxel-merge/track-quality/statistical-filter screenshots reviewed only through verifier output.

## Not done
- 2.9 gestures that change nothing: dem-coregistration readout is constant by design (the bias does not depend on the pointer), line-loop-removal, minimum-clearance, network-snapping, geometry-orientation, lon-lat, neighborhoods untouched.
- terrain-critical-points / weiss swatch dimming, pointcloud-neighborhoods count key, cell-geometry legend, voxel-merge one-run lag (2.14), pair-links coupling (2.15): minor, left.

## Work list for others
Rest picture changed (regenerate posters): map-matching, terrain-dem-coregistration, adjacency-matrix, adjacency-matrix-order, pointcloud-statistical-filter (caption only, rest unchanged), pointcloud-height, spectral-unmixing, pointcloud-icp (caption only), geometry-orientation, lon-lat-polygon-preparation, computation-metrics, computation-record, point-horizon-profile, clock-encounters, trajectory-encounters, trajectory-closest-approach, dasymetric-redistribution-recipe, hot-spot-analysis-recipe (legend), network-kernel-density, catchment-accessibility, pointcloud-voxel-merge, raster-registration (label).
Shared edit outside my folders: website/build/demo-data-sources.ts (SCENARIOS: terrain-dem-coregistration, adjacency-matrix, adjacency-matrix-order, glcm-texture).

## Roadmap
None touched.

## Noticed
- pointcloud-icp: at 8 degrees and 12 iterations RMSE falls 0.044 to 0.023 (0.039 to 0.022 at other corners); only the 0 degree case reaches 0.000. Point-to-point closest-point matching on a dense, nearly flat cloud with no targetGrid converges linearly; I could not check against a CPU reference.
- spectral-unmixing with hand-set endmembers gave water 0.52-0.57 everywhere on the Dixie scene; the operation was behaving, the endmembers did not fit.
