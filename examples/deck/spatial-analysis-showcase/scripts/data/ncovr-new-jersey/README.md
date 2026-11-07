# NCOVR New Jersey county units

This folder builds the 21 New Jersey units from the National Consortium on Violence Research
(NCOVR) county data with its supplied historical common-support polygons. The source page describes
the national file as 3,085 observations and 69 variables for 1960, 1970, 1980 and 1990, and attributes
it to Messner, Anselin, Hawkins, Deane, Tolnay and Baller. GeoDa states that the data are provided
“as is” with no warranties. This project does not assert a separate license for the source data.

Source: <https://geodacenter.github.io/data-and-lab/ncovr/>  
Archive: <https://geodacenter.github.io/data-and-lab/data/ncovr.zip>

## Rebuild

Download the archive and its `codebook.pdf`, then run:

```sh
python3 build.py --archive /path/to/ncovr.zip --codebook /path/to/codebook.pdf
```

The builder uses only Python's standard library. It selects every row with state FIPS `34`, sorts
by the source `FIPSNO`, and matches each row to the source GeoJSON feature by the same identifier.
It fails unless exactly 21 rows and 21 matching geometries are present. The original GeoJSON
coordinates and source values are retained. The builder pins the source archive and codebook
SHA-256 hashes and fails if they drift. `--allow-unverified-source` explicitly permits a different
input pair and records that override in the manifest. The manifest records hashes for every generated
file except the manifest itself.

The verified archive hash is `e8cb04e6da634c6cd21808bd8cfe4dad6e295b22e8d40cc628e666887719cfe9`;
the verified codebook hash is `229aac156c4005e70c2b43d15c69c73c0aacf4884a6af55f57487dbffe6b3de7`.

## Variables and limits

The three analysis periods are 1970, 1980 and 1990. `UE` is percent of the civilian labor force
unemployed; `FP` is percent of families below the poverty threshold for the corresponding census
vintage; `DNL` is source log population density; `PO` contains actual decennial population counts;
and `POL` is source log population. The `FP59`/1960 field is omitted because the codebook says that
its $3,000 threshold differs from later poverty definitions. Nominal income fields are also omitted.
These are three decennial snapshots, not an annual or dynamic panel. The county units reflect the
archive's Horan–Hargis historical common-support template. The template label alone does not imply
that present county identities changed; the source geometry is retained as published.

The scene projects coordinates into a spherical azimuthal-equidistant planar-meter frame centered
at `[-74.55, 40.05]`. This local metric frame is used by the analysis and inverted for map display;
the bundled GeoJSON remains in its original longitude/latitude coordinates. No current-boundary
overlay or cross-frame coordinate mix is used.
