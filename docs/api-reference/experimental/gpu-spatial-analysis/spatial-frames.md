# Spatial frames

A `GPUSpatialFrame` describes the projected planar coordinate space of a set of rows. It is carried
by `GPUSpatialContext.frame` and is how planar operators and projection graphs reject data that was
projected differently.

## Frame model

A frame is an identity plus optional math.gl CRS metadata:

```ts
type GPUSpatialFrame = {
  id: string; // identity; equal ids mean interchangeable coordinates
  kind: 'local-azimuthal-equidistant' | 'local-lambert-equal-area' | 'crs';
  origin?: [number, number]; // [longitude, latitude] in degrees
  crs?: string; // identifier as written, for kind 'crs'
  signature?: string; // execution signature of resolved math.gl CRS metadata
  unit?: GPUSpatialFrameUnit; // math.gl CRSUnit: name, metersPerUnit, degreesPerUnit
  axisOrder?: string; // for example 'enu'
  areaOfUse?: readonly [west: number, south: number, east: number, north: number];
  anchor?: 'projection-center' | 'relative-origin';
  precision?: 'float32' | 'float32-origin-relative' | 'float64-split';
};
```

Compatibility between two contexts is decided by comparing frame `id` only. Every other field is
descriptive metadata and never changes whether two frames are compatible. Two frames with
different ids are incompatible even if they describe the same CRS; build both with the same
constructor and options to get the same id.

- `anchor: 'projection-center'` means the projection is centered on `origin` (local frames).
- `anchor: 'relative-origin'` means coordinates are the CRS's own, stored relative to `origin` to
  keep float32 precision.

### Signature and semantic equivalence

`signature` is the math.gl `getCRSSignature` of resolved CRS metadata. It compares execution
conventions only (the parameters that determine how coordinates are computed). It is not a proof
that two CRS definitions are semantically the same CRS, and it is not used for compatibility.
Use it to tell whether two frames execute identically; use `id` to decide whether data may be
mixed. Datum or area-of-use differences that do not change execution are not reflected in it.

## Units and precision

`unit` is math.gl's `CRSUnit`. The library checks declared units and never converts them.
`GPUSpatialContext.units` is derived from the frame unit: `'meters'` for `metersPerUnit === 1`,
`'kilometers'` for `1000`, otherwise undeclared (native).

Local frames are in meters, centered on their origin and held origin-relative in float32
(`precision: 'float32-origin-relative'`). The origin is subtracted in binary64 on the CPU, so float32
frame coordinates keep about 1e-7 of the frame extent (millimeters at 10 km, centimeters at 100 km),
instead of the 0.4 to 1 m quantization of float32 longitude/latitude.

## Distance frames and area frames

Local frames come in two kinds, listed in `GPU_LOCAL_FRAME_KINDS`:

| `kind` | Frame kind | Projection | Use for |
| --- | --- | --- | --- |
| `'azimuthal-equidistant'` | `local-azimuthal-equidistant` | Spherical AEQD on the mean Earth radius (or `sphereRadius`) | Distance bands, neighbor radii, buffers. Distances and bearings from the origin are exact great-circle values; other distances are within about 0.1% up to 200 km. |
| `'lambert-equal-area'` | `local-lambert-equal-area` | Ellipsoidal (WGS84) LAEA, `+proj=laea +ellps=WGS84` | Densities, areal weights, dot sampling. Areas are exact ellipsoidal areas at any scale. |

`GPU_LOCAL_FRAME_KINDS` maps each local kind to its `GPUSpatialFrameKind`. Distance operators
reject Lambert frames and area operators reject AEQD frames.

## Creating frames

### createGPULocalFrame

```ts
import {createGPULocalFrame} from '@luma-contributors/analysis/gpu-spatial-analysis';

const frame = createGPULocalFrame({origin: [-122.4, 37.8], kind: 'azimuthal-equidistant'});
const [x, y] = frame.project(-122.39, 37.81); // meters, binary64
const positions = frame.projectArray(lngLats, 2); // interleaved float32 [x, y]
// frame.spatialContext carries units 'meters' and the frame identity.
```

`createGPULocalFrame(props)` returns a binary64 CPU-side frame with `project`, `unproject`,
`projectArray` and `unprojectArray`. Props: `origin` (required, finite, `|latitude| <= 90`), `kind`
(default `'azimuthal-equidistant'`), `sphereRadius` (AEQD only; default mean Earth radius),
`id` and `datum`. The `datum` is copied to `spatialContext.datum` and does not change the frame id.
Antipodes project to NaN.

### createGPULocalSpatialFrame

Creates only the `GPUSpatialFrame` descriptor (meters, `axisOrder: 'enu'`,
`anchor: 'projection-center'`, `precision: 'float32-origin-relative'`) without a transform. The
default id is `local-azimuthal-equidistant(lng,lat)` or `local-lambert-equal-area(lng,lat)`; an AEQD
frame with a non-default `sphereRadius` appends `,R=<radius>`.

### createGPUSpatialFrameFromCRS

```ts
import {createGPUSpatialFrameFromCRS} from '@luma-contributors/analysis/gpu-spatial-analysis';

const frame = createGPUSpatialFrameFromCRS('urn:ogc:def:crs:EPSG::32633');
// frame.id === 'EPSG:32633'
```

Creates a `'crs'` frame synchronously, without a CRS provider. The id is the normalized authority
identifier (`epsg:3857` and `urn:ogc:def:crs:EPSG::3857` both give `EPSG:3857`), else the metadata
signature, else the input string. `crs` keeps the input as written. Options: `metadata` (resolved
math.gl metadata, for example from `projectionEngine.getCRS`, giving the signature, unit, axis
order and area of use), `id` and `precision`. The unit comes from `metadata`, then the PROJ string
`+units`/`+to_meter`, then a known registry family; it is omitted when unknown.

### createGPUSpatialFrameFromIdentifier and getKnownGPUSpatialFrameUnit

`createGPUSpatialFrameFromIdentifier(crs)` creates a `'crs'` frame with no math.gl runtime code:
the id and `crs` are the input as written. It is what `GPUProjection` uses for a plain `targetCrs`
string, so spellings are not normalized.

`getKnownGPUSpatialFrameUnit(identifier)` returns the unit of a recognized family: `EPSG:4326` and
`OGC:CRS84` (degrees), `EPSG:3857` and WGS84 UTM `EPSG:326xx`/`EPSG:327xx` (meters). It returns
`undefined` for anything else.

## What math.gl provides

The frame facade delegates to `@math.gl/geospatial/crs`: `normalizeCRSIdentifier`,
`getCRSSignature`, `parsePROJString` and `linearUnits`, plus the `CRSMetadata` and `CRSUnit`
types. Local frame transforms use `projectionEngine` from `@math.gl/geospatial/projection`. The
`gpu-project` execution bundle imports math.gl types only, so it stays free of math.gl CRS runtime
code; only `createGPUSpatialFrameFromCRS` pulls the runtime in.

## Projection spatial context

`getGPUProjectionSpatialContext(targetCrs?, frame?)` is internal (`@internal`) and builds the planar
native context for rows projected into a target CRS. It returns `undefined` when neither argument
is given. A `frame` wins over `targetCrs`; otherwise a frame is made with
`createGPUSpatialFrameFromIdentifier(targetCrs)`. `units` is declared when the frame unit is known.

`GPUProjection`, `CoordinateProjectionGraph`, and `prepareCRSProjection` accept optional
`sourceFrame` and `targetFrame` props next to the CRS string props. A frame's `crs` takes
precedence over the matching string, and results expose `spatialContext` built from the target
frame. A target frame origin becomes the compiled plan's `destinationOrigin` when its anchor is
`'relative-origin'`, or when its precision is `'float32-origin-relative'` without a
`'projection-center'` anchor. An explicit `destinationOrigin` compile option still wins, and
behavior without frames is unchanged. `sourceFrame` currently only supplies the source CRS.

## deck.gl adapter

```ts
import {getDeckCoordinatePropsFromSpatialFrame} from '@deck.gl-community/gpu-layers';

const {coordinateSystem, coordinateOrigin} = getDeckCoordinatePropsFromSpatialFrame(frame);
```

Maps a planar frame to deck.gl layer coordinate props:

- Local frames (AEQD and LAEA) become `COORDINATE_SYSTEM.METER_OFFSETS` with
  `coordinateOrigin: [longitude, latitude, 0]` at the frame origin. It throws if the frame has no
  origin.
- `'crs'` frames become `COORDINATE_SYSTEM.CARTESIAN`, treating coordinates as unprojected scene
  units; the caller owns the projection.

deck's `METER_OFFSETS` is a local east/north/up tangent approximation, not an exact AEQD or LAEA
projection. The two agree at the origin and diverge by roughly `d^3 / (6 R^2)`:

| Distance from origin | Approximate error |
| --- | --- |
| 1 km | under 1 mm |
| 10 km | under 0.1 m |
| 100 km | a few meters (about 4 m) |

The unit test asserts bounds of 1 mm, 0.1 m and (loosely) 100 m at these distances, against exact
spherical AEQD at 45 degrees azimuth; the 4 m figure is the analytic estimate, not a test bound.
Use the adapter for neighborhood-scale scenes only.
