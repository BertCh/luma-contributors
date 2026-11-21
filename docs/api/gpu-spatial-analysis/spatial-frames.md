# Spatial frames

A `GPUSpatialFrame` describes the projected planar coordinate space of a set of rows. It is carried
by `GPUSpatialContext.frame` and is how planar operators and projection graphs reject data that was
projected differently.

## Frame model

A frame is an identity plus optional math.gl CRS metadata:

```ts
import type {GPUSpatialFrameUnit} from '@luma-contributors/analysis/analysis-core';

type GPUSpatialFrame = {
  id: string; // identity; compatibility also checks relative storage origins
  kind: 'local-azimuthal-equidistant' | 'local-lambert-equal-area' | 'crs';
  origin?: [number, number]; // local center: degrees; relative storage: target CRS units
  crs?: string; // identifier as written, for kind 'crs'
  signature?: string; // execution signature of resolved math.gl CRS metadata
  unit?: GPUSpatialFrameUnit; // math.gl CRSUnit: name, metersPerUnit, degreesPerUnit
  axisOrder?: string; // for example 'enu'
  areaOfUse?: readonly [west: number, south: number, east: number, north: number];
  anchor?: 'projection-center' | 'relative-origin';
  precision?: 'float32' | 'float32-origin-relative' | 'float64-split';
};
```

Compatibility compares frame `id` and effective relative storage origins, including when an
explicit id is supplied. Two frames with different ids are incompatible even if they describe
the same CRS; build both with the same constructor and options to get the same id. Relative
origins default to a captured, published `[0, 0]` tuple and also apply when precision is `float32-origin-relative` unless
the anchor is `projection-center`.

- `anchor: 'projection-center'` means the projection is centered on `origin` (local frames).
- `anchor: 'relative-origin'` means coordinates are the CRS's own, stored relative to `origin` to
  keep float32 precision. Storage origins use target CRS units (meters for UTM); local projection
  centers use `[longitude, latitude]` in degrees.

### Signature and semantic equivalence

`signature` is the math.gl `getCRSSignature` of resolved CRS metadata. It compares execution
conventions only (the parameters that determine how coordinates are computed). It is not a proof
that two CRS definitions are semantically the same CRS, and it is not used for compatibility.
Use it to tell whether two frames execute identically; use `id` to decide whether data may be
mixed. Datum or area-of-use differences that do not change execution are not reflected in it.

## Units and precision

`unit` is math.gl's `CRSUnit`. The library checks declared units and never converts them.
For manually described frames without `unit`, recognized registry CRS identifiers still supply
their known unit for validation and length compatibility.
`getGPUSpatialFrameContextUnits` returns `'meters'` for `metersPerUnit === 1` and otherwise leaves
`GPUSpatialContext.units` undeclared (native), as required by the native planar context validator.
Declaring `units: 'meters'` for a known non-meter frame throws. A known angular frame
cannot use the planar native metric: project its positions into a linear frame first. Provider-free
PROJ `longlat`, `latlong`, `lonlat` and `latlon` resolve as degrees. Raw unlabelled XY alone cannot
determine units; callers must declare the coordinate space and frame from their data source.

Kilometers and other units remain in `frame.unit`; coordinates and thresholds continue to use
those native frame units without conversion.

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
import {createGPULocalFrame} from '@luma-contributors/analysis/analysis-core';

const frame = createGPULocalFrame({origin: [-122.4, 37.8], kind: 'azimuthal-equidistant'});
const [x, y] = frame.project(-122.39, 37.81); // meters, binary64
const lngLats = Float64Array.of(-122.39, 37.81, -122.38, 37.82);
const positions = frame.projectArray(lngLats, 2); // interleaved float32 [x, y]
// frame.spatialContext carries units 'meters' and the frame identity.
```

`createGPULocalFrame(props)` returns a binary64 CPU-side frame with `project`, `unproject`,
`projectArray` and `unprojectArray`. Props: `origin` (required, finite, `|latitude| <= 90`), `kind`
(default `'azimuthal-equidistant'`), `sphereRadius` (AEQD only; default mean Earth radius),
`id` and `datum`. The `datum` is copied to `spatialContext.datum` and does not change the frame id.
Unknown local-frame kinds throw with accepted values `azimuthal-equidistant` and
`lambert-equal-area`. Spatial context units accept only `native`, `meters` and `kilometers`.
Antipodes project to NaN. `projectArray` requires complete rows (`length % stride === 0`);
`unprojectArray` requires complete XY pairs (`length % 2 === 0`). Partial inputs and short
destinations throw before any destination write. Overlapping typed-array views are supported
by snapshotting the source. Extra destination elements remain unchanged.

### createGPULocalSpatialFrame

Creates only the `GPUSpatialFrame` descriptor (meters, `axisOrder: 'enu'`,
`anchor: 'projection-center'`, `precision: 'float32-origin-relative'`) without a transform. The
default id is `local-azimuthal-equidistant(lng,lat)` or `local-lambert-equal-area(lng,lat)`; an AEQD
frame with a non-default `sphereRadius` appends `,R=<radius>`.

### createGPUSpatialFrameFromCRS

```ts
import {createGPUSpatialFrameFromCRS} from '@luma-contributors/analysis/analysis-core';

const frame = createGPUSpatialFrameFromCRS('urn:ogc:def:crs:EPSG::32633');
// frame.id === 'EPSG:32633'
```

Creates a `'crs'` frame synchronously, without a CRS provider. The id is the normalized authority
identifier (`epsg:3857` and `urn:ogc:def:crs:EPSG::3857` both give `EPSG:3857`), else the metadata
signature, else the input string. `crs` keeps the input as written. Options: `metadata` (resolved
math.gl metadata, for example from `projectionEngine.getCRS`, giving the signature, unit, axis
order and area of use), `id`, `precision`, `origin` and `anchor`. Invalid anchors and precisions
throw with their accepted spellings; supplied origins must contain two finite numbers. Origin tuples are captured at
construction. Relative storage origins are included in the derived id. The unit comes from `metadata`, then the PROJ string
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

## Frames and the published projection classes

The published projection entry points (`prepareCRSProjection` of `@luma.gl/graph/gpu-project/crs`,
and the plans and programs of `@luma.gl/graph/gpu-project`) take CRS strings and an optional
`destinationOrigin`. They do not take a `GPUSpatialFrame`. To project into a frame, pass the frame's `crs` as the target CRS. Pass the
frame's `origin` as `destinationOrigin` only when that origin is a point in the target CRS that
the coordinates are stored relative to: `anchor: 'relative-origin'`, or
`precision: 'float32-origin-relative'` with an anchor other than `'projection-center'`. A frame
anchored at `'projection-center'` (a local frame) holds its origin as longitude and latitude, the
centre of the projection, and its coordinates are already relative to it: pass no
`destinationOrigin` for such a frame. Without a `destinationOrigin` the compiled plan keeps the
origin the projection compiler chooses.

## Rendering frames

Configure the renderer's coordinate frame, projection and origin explicitly from the stored data
coordinates. This package exports frame descriptors and projection helpers; it does not export a
deck.gl layer adapter. The former `@deck.gl-community/gpu-layers` import is not a published
dependency of this workspace. The [Kepler example](https://github.com/visgl/luma-contributors/blob/master/examples/deck/kepler-compute/README.md)
describes its local rendering integration and pinned deck.gl version.

A local AEQD or LAEA projection is distinct from an east/north/up tangent offset approximation.
Reusing the same longitude/latitude origin alone does not make their coordinates interchangeable.
Project the data into the renderer's frame, or choose a rendering projection that preserves the
analysis frame; keep the declared origin and units consistent in both paths.

## Comparing metric results

Position compatibility retains the coordinate contract. This mode compares **lengths and distance
thresholds**, including the length radius; it does not certify area results. When comparing lengths or thresholds,
request `purpose: 'metric-results'`: it checks effective result units, sphere radius and ellipsoid,
including their defaults. Omitted sphere radius matches explicit `6371008.8` meters, but differs
from a unit sphere. Native planar units resolve through the known frame scale, so an omitted unit
in a meter frame matches explicit meters. Kilometers differ from meters even when positions share
the same frame.

For areas, compare the resolved area model and area radius separately. Default linear-edge
spherical areas use the authalic radius `6371007.180918475` meters; explicitly setting
`sphereRadius: 6371008.8` overrides that area radius even though the length radii match.

```ts
import {
  assertCompatibleGPUSpatialContexts,
  type GPUSpatialContext
} from '@luma-contributors/analysis/analysis-core';

const source: GPUSpatialContext = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'};
const target: GPUSpatialContext = {...source, units: 'meters', sphereRadius: 6371008.8};
assertCompatibleGPUSpatialContexts('length comparison', source, target, {purpose: 'metric-results'});
```

The same optional purpose argument is accepted by `getGPUSpatialContextCompatibility`.
