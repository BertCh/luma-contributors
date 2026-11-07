# Äußeres Hochebenkar: genuine repeated-epoch lidar

This is a small derived sample of real terrestrial laser scans from 2017 and 2018A at the Äußeres
Hochebenkar rock glacier, Obergurgl, Tirol, Austria. It is not a synthetic displacement experiment.

Source: Winiwarter, Lukas; Anders, Katharina; Zahs, Vivien; Hämmerle, Martin; Höfle, Bernhard (2021),
*M3C2-EP: Pushing the limits of 3D topographic point cloud change detection by error propagation
[Data and Source Code]*, [heiDATA, V1](https://doi.org/10.11588/data/XHYB10), CC BY 4.0.
The bounded source clouds are distributed by the authors' group in
[py4dgeo-test-data](https://github.com/3dgeo-heidelberg/py4dgeo-test-data), release 2026-01-29.
The source license and required attribution are retained in `public/data/ahk-rock-glacier/`.

## Preparation

Run `build.py` with Python 3.12+, NumPy, laspy, lazrs and pyproj. Its archive checksum is pinned;
the downloaded 29 MB archive remains in `/tmp` and never ships with the app.

The preparation reads only the two named LAZ epochs, the provided core points and normals, and the
provided `tfM.csv` / `redPoint.csv` registration parameters. It applies the provider's affine
transformation to the comparison cloud around the provided reduction origin, exactly as described
in [the py4dgeo M3C2-EP documentation](https://py4dgeo.readthedocs.io/en/latest/m3c2ep.html).
It then retains the first original row in each occupied 0.6 m 3D voxel, restoring original source
row order. Every eighth provided core point is retained with its corresponding original normal.
No points are interpolated; original LAZ row IDs are shipped for audit.
Three retained core normals contain source NaN values. They are preserved and must be rejected
as invalid orientations, separately from cylinders with insufficient point support. The other
255 supplied normals have unit length to float32 precision.

| Column group | Source rows | Retained rows |
| --- | ---: | ---: |
| 2017 reference | 1,195,012 | 5,192 |
| 2018A comparison | 1,324,416 | 5,407 |
| Provided cores and normals | 2,064 | 258 |

Computational XYZ values are native UTM32N metres minus `[652920, 5189155, 2485]` to preserve
float32 precision. The provider names UTM32N but the bounded LAZ samples do not specify a complete
horizontal or vertical datum. EPSG:32632 is assumed only for display longitude/latitude conversion;
analysis uses the untouched native metric axes. The common translation does not change the given
normal vectors. The full applied affine transform, reduction origin, native bounds, input and
output SHA-256 hashes and sampling policy are in `metadata.json` and `manifest.json`.

The companion vertical-frame study uses `demReference`, `demComparison`, their `Count` columns,
and geographic `demPosition` cell centres. These are arithmetic mean elevations on a fixed,
north-up 38 × 37 grid of 1 m native-XY cells. Each epoch needs at least three retained returns
in a cell; missing cells remain NaN. There are 718 paired supported cells. The affine transform
is `[1, 0, -18, 0, -1, 18]` relative to the native origin, with area pixels. No surface
interpolation or hole filling occurs. These sampled cell means are not independent survey DEMs;
different within-cell sampling can contribute to their elevation difference. No source stable
mask or calibrated cell-difference variance is supplied. A reader-selected control region and
measurement-error covariance must be described as assumptions, not observations.

## Scientific limits

The supplied alignment is retained; fitting another ICP transform over this moving crop can absorb
surface movement into a registration correction. Such a fit is a sensitivity demonstration, not
evidence of a better survey registration. The original study used stable areas for alignment.

Sampling alters cylinder support and roughness. The showcase's M3C2-style mean difference and
detection threshold do not reproduce the original M3C2-EP sensor and 12-parameter alignment-error
propagation, and results must not be described as the published study's findings. The two file
labels identify acquisition epochs; the compact distribution does not establish exact elapsed
days, so the showcase should report displacement in metres, not annual velocity.
