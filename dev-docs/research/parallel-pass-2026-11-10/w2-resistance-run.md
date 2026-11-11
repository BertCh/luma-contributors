# w2-resistance-run

Files: website/src/gpu/runs/entries/recipes-trajectories/raster-resistance-connectivity-recipe.ts and .parameters.ts; one catalog entry inserted before sparse-resistance-connectivity-recipe in website/src/catalog/entries/recipes-trajectories.ts.

Picture: 48x32 constructed resistance raster (Modeled scenario: no data source, card shows "Modeled scenario"; caption says so). Input (muted grey): rough background, a road with one crossing (resistance = barrier), a lake of NaN cells (no-data, removed from circuit). Output (bright, fade): vertexCurrents. Current concentrates through the crossing and bends around the lake; five patches on the left, one ground on the right.
Gesture: place. Pointer snaps (chooseSticky) to one of five habitat patches; the source is compile-time, so it goes through `settle` (buildCache 10). Sliders: source, neighborhood (4/8; compile-time), Data: barrier resistance (per frame). Readout: "effective resistance 2.70" (diagnostics[0], with "not converged" fallback).
Verifier (dark+light, --out website/.reports/w2-resistance): 2/2 pass, WebGPU, ~1 s start, sliders change 22.5% / 19.6% / 9.0%, pointerRebuilds 0, restDifference 0. One warning (dark): only 2 distinct pictures at 5 pointer positions (five snapped patches, the five probe positions hit two). Readouts 2.70 and 3.04.
Screenshots read: dark, light, pointer-1, pointer-2 (source patch 2 vs 4: current climbs around the lake to the crossing).
tsc: no errors in my files. check-catalog recipes-trajectories.ts: 0 errors.
Recipe notes: cellSize 1 used so effective resistance is in cell units (with ground cell size s, R_eff scales by s, documented). Catalog check forces listing contributors/intermediates/status/provenance/estimatedCommandNodeCount as outputs, leaving little room for edge currents (8-output cap). Seed 9241 used (427 was taken in another file). Not run: snapshot-runs, builds.
