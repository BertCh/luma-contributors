# Spectral texture display and transect implementation ledger

## Scope and source

- Scene: `raster/spectral-texture`, backed by the existing `dixie-spectral` dataset.
- Both existing acquisitions remain available: 2021-07-13 (ignition day) and 2021-09-21 (later scene). The scene continues to use the existing Copernicus Sentinel attribution and dataset provenance.
- The source is a 250 × 250, north-first, 60 m raster. The scene uses the same 128 × 128 crop for both dates: source column 60 through 187 and row 50 through 177 (zero-based, inclusive); projected crop bounds are `[666400, 4445360, 674080, 4453040]` in EPSG:32610.
- A crop cell is supported only when the dataset's strict native QA byte is `1` and both NIR and SWIR₂ values are finite. Invalid cells stay invalid through the display, profile, and transport paths. The existing `sourceSupport` readout reports the selected acquisition's actual valid cell count.
- Static inspection of bundled source bytes found 16,246 valid crop cells on July 13 (NIR min/max `0.1705388874` / `0.4677083194`) and 9,516 on September 21 (`0.0406388901` / `0.3568777740`). No strict-valid NIR values in either crop lie outside literal `[0, 0.6]`.
- The original filtering, texture, and focal-pairs paths still consume calibrated source NIR/SWIR₂. The transformed display buffer is separate and is not connected to those contributors.

## Display transform

- `GPURasterHistogram` and `GPURasterContrast` are added to the selected acquisition's existing compiled command graph; no CPU histogram is supplied to the equalizer.
- Domains: literal calibrated `[0, 0.6]`, or the valid finite source min/max reduced on the GPU. The automatic domain is written to a caller-owned two-float buffer by the validity-aware raster histogram and reused by the selected histogram, contrast transform, and transformed-value histogram.
- Source histogram counts for equalization use 64 or 256 bins according to the control. Histogram bins are half-open intervals derived from `floor(((value - min) / (max - min)) * binCount)`, with the maximum included in the last bin. `GPURasterHistogram` excludes invalid/non-finite samples and valid values outside its domain; it does not clamp them into edge bins. The bundled selected-date NIR crop was statically inspected: every strict-valid source value is inside literal `[0, 0.6]`; GPU min/max includes all valid finite values by construction. `GPURasterContrast` independently clamps normalized samples to the domain before transformation.
- The transformed distribution chart compares the source and transformed GPU histograms with 256 bins over the selected domain. In GPU-domain mode, the source comparison uses the GPU auto-domain histogram; in literal mode, it uses the `[0, 0.6]` histogram. The chart counts come from GPU readback.
- Linear and gamma contrast center and clamp normalized values. The contributor's gamma convention is `pow(normalized, 1 / gamma)`. Equalization uses the inclusive histogram CDF; it subtracts the first-bin cumulative count and divides by `validCount - firstBinCount`, with an empty histogram invalidated. Contrast and gamma are applied after the CDF result. Values are mapped back into the selected scalar domain and retain an independent output validity mask.
- Display mode defaults to linear; contrast and gamma to `1`; bin count to `64`; domain to calibrated. These new settings do not change the existing source, date, filter, texture, or focal defaults.
- Method context: [scikit-image exposure transforms](https://scikit-image.org/docs/stable/api/skimage.exposure.html) describe distribution transforms for display; this scene does not treat them as recovered observations.

## GPU profile transect

- The fixed polyline is specified by east/south distances `[450,390]`, `[3930,3690]`, `[7290,7290]` from the crop northwest corner. GPU profile positions use east/north coordinates relative to `[666400,4453040]`: `[450,-390]`, `[3930,-3690]`, `[7290,-7290]`. Planner review corrected the sign so the sampler and rendered line use the same cells. In source pixel-center coordinates these are `(column,row) = (7,6)`, `(65,61)`, and `(121,121)`. The path has two segments and crosses the source crop. A static inspection of the bundled QA masks shows both endpoints invalid on September 21 and valid on July 13; the intervening line crosses both supported and invalid cells on the later date.
- Grid row zero remains north: the profile uses the source crop's west/north affine origin and a negative Y row step. Local-origin coordinates avoid float32 precision loss from absolute UTM positions. The cyan map line uses those same vertices transformed into the scene's established raster frame.
- Default spacing is 120 m; supported settings are 60, 120, and 240 m. The polyline length is 9,720.2615 m, so required sample counts are 164, 83, and 42 respectively (regular spacing samples below the endpoint plus the exact endpoint). Default interpolation is nearest; bilinear is available. Strict missing-data behavior is the default; renormalization is available for bilinear support.
- `GPURasterProfile` emits distance and measured reflectance columns, exact line length, count, required count, overflow, minimum, and maximum. Capacity is 512 samples. The scene surfaces both emitted and required counts and the overflow word; the chart only includes emitted rows. Contributor sampling includes the exact final endpoint. No CPU point-loop interpolation is used.
- The profile reports measured NIR only. It does not call positive or negative reflectance increments elevation ascent/descent.

## Buffer/texture/buffer transport

- The selected-date calibrated source NIR is float32. It is written into one selected R/G/B/A channel of a caller-owned 128 × 128 `rgba32float` storage texture, then gathered from that same canonical graph texture view into a float32 output buffer.
- Both conversion legs set `applyCalibration: false`. No synthetic DN or calibration demonstration is claimed. The non-selected channels are zero-filled by the contributor; the explicit validity buffer remains canonical.
- Storage support is checked with `device.getTextureFormatCapabilities('rgba32float').store` before graph construction. The returned buffer is selectable as a rendered map layer. The readout compares actual GPU-readback returned samples against the source array and reports maximum absolute finite-sample error, finite comparison count, and validity mismatch count.

## Verification ledger

Implemented and statically checked in this change:

- Planner review consolidates the inquiry into six steps: the original observation step now includes display transforms, paired-fit interpretation includes the former support-limits step, and a final transect step covers sampling and transport. Baseline observation/scale/edges/texture/paired IDs and defaults remain; support-limits was consolidated and the draft display-transforms step removed.
- GPU contributors are present in the executable graph and their outputs feed raster rendering, charts, or readouts. GPU histogram counts are read back; the profile uses the contributor's count/required/overflow outputs; the transport values and validity are read back from its returned buffer.
- Scoped Biome formatting/check passed for the three owned TypeScript files.
- `node scripts/check-story-steps.mjs raster` passed, including the spectral scene's controls/readout references.
- `node scripts/check-wgsl-identifiers.mjs src/scenes/raster` reports three pre-existing constant-NaN-bitcast issues in the dirty `tiled-observations.compute.ts`; it reports no issue in the owned spectral-texture files.
- Package `typecheck` was run. Existing errors in unrelated shared dirty scenes were reported across geometry, hydrology, points, and tiled observations; the spectral-texture compute, scene, and display helper had no reported TypeScript errors in that run.
- Strict cartography check reports only the scene's eight total steps (recommended 5–6). The original scene already had six and all six IDs/step values are preserved; the two added steps are the requested maximum. The display map and legend use `cividis`.

Still pending and requiring the parent-coordinated browser/GPU slot:

- Actual WebGPU pipeline compilation and execution on a browser device, all three contrast modes, both histogram bin counts/domains, both profile interpolation/missing-data modes and spacings, and all four texture channels.
- Numerical independent-oracle comparison for GPU transformed values, profile samples, and texture roundtrip, including actual max error and validity mismatch counts.
- Visual review of the new maps, cyan transect alignment, legend labels, and rapid graph swaps. No static class declaration or successful TypeScript check is presented as GPU execution coverage.
