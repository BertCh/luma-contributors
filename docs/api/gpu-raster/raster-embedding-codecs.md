# Embedding raster codecs

Import from `@luma-contributors/analysis/raster-analysis`. These codecs convert caller-supplied,
pixel-major arrays; they do not run a model or include weights or datasets. Related analysis operators
are described in [Embedding rasters](./raster-embeddings.md).

## `GPURasterEmbeddingDequantize`

Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding
records stay together. Reference covariance-family dimensions remain 128.

The existing decoder accepts `format` (compile-time, default `'alphaearth-int8'`):

| Format | Packed `quantized` input | Decoded component | Invalid pixel |
|---|---|---|---|
| `alphaearth-int8` | Four signed bytes per uint32, low byte first | `sign(code) * (code / 127.5)^2` | Any byte is -128 |
| `tessera-int8` | Four signed bytes per uint32, low byte first | `float32(code) * scales[pixel]` | Scale or decoded component is non-finite |
| `float16` | Two IEEE binary16 components per uint32, low half first | IEEE binary16 to float32 | Any component has exponent 31 (NaN or infinity) |

Every format also accepts a per-pixel uint32 `mask`; zero invalidates the entire pixel. Invalid pixels
write NaN to every output component and zero to optional `output.validity`. Other pixels write validity
one, including all-zero vectors. TESSERA uses all signed byte codes: -128 is a value, not a sentinel.
Zero and negative finite scales are multiplied as supplied. A TESSERA scale is one float32 per pixel;
per-component scales are not supported. `scales` is required only for TESSERA and rejected otherwise.

The TESSERA formula matches the authors' [GeoTessera `dequantize_embedding`](https://github.com/ucam-eo/geotessera/blob/main/geotessera/core.py#L30-L61)
(verified 2026-11-14). Binary16 follows the [WGSL floating-point representation](https://www.w3.org/TR/WGSL/#floating-point-types):
normal magnitudes are `(1024 + fraction) * 2^(exponent - 25)` and subnormals are `fraction * 2^-24`.
The decoder reads integer fields, preserving half-float subnormals without `shader-f16`.

```ts
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPURasterEmbeddingDequantize} from '@luma-contributors/analysis/raster-analysis';

// Packed input has pixelCount * 128 / 4 words; output has pixelCount * 128 floats.
declare const pixelCount: number;
declare const packedSignedBytes: GraphDataView<'uint32'>;
declare const mask: GraphDataView<'uint32'>;
declare const validity: GraphDataView<'uint32'>;
declare const pixelScales: GraphDataView<'float32'>;
declare const vectors: GraphDataView<'float32'>;
// pixelScales, mask and validity each have pixelCount rows, in distinct buffers.
graph.add(new GPURasterEmbeddingDequantize({
  pixelCount,
  dimensions: 128,
  format: 'tessera-int8',
  quantized: packedSignedBytes,
  scales: pixelScales,
  mask,
  normalize: true,
  output: {vectors, validity}
}));
```

`normalize: true` divides finite nonzero vectors by their length, scaling by the largest magnitude
first to avoid overflow. A zero vector stays zero. The default preserves decoded magnitudes. Each word
array has exactly `ceil(pixelCount * dimensions / 4)` words for int8 or
`ceil(pixelCount * dimensions / 2)` for binary16. Unused lanes of the final word are ignored, even when
they contain a sentinel or NaN.

## `GPURasterEmbeddingQuantize`

Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding
records stay together. Reference covariance-family dimensions remain 128.

{/* lens:GPURasterEmbeddingQuantize */}

::lens{scenario=gpu-raster.raster-embedding-quantize rung=formula}

The synthetic lens encodes three three-component pixels, including saturation, a masked pixel, a nonfinite pixel and zeroed unused final bytes.

Encodes float32 vectors to the AlphaEarth int8 format, including vectors produced by
`GPURasterEmbeddingPyramidLevel`. The formula is
`clip(roundToEven(sign(value) * sqrt(abs(value)) * 127.5), -127, 127)`, from
[AlphaEarth Foundations, supplementary S22.1](https://arxiv.org/html/2507.22291v2#S22.SS1)
(power 2, scale 127.5; verified 2026-11-14). Rounding follows nearest-even, including negative ties.
Finite components outside [-1, 1] saturate. The encoder does not normalize vectors.

```ts
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPURasterEmbeddingQuantize} from '@luma-contributors/analysis/raster-analysis';

declare const pixelCount: number;
// Overview input has pixelCount * 64 float32 components, with a per-pixel mask.
declare const overviewVectors: GraphDataView<'float32'>;
declare const overviewValidity: GraphDataView<'uint32'>;
declare const packedSignedBytes: GraphDataView<'uint32'>;
declare const validity: GraphDataView<'uint32'>;
// Packed output has pixelCount * 64 / 4 words; validity has pixelCount rows.
graph.add(new GPURasterEmbeddingQuantize({
  pixelCount,
  dimensions: 64,
  vectors: overviewVectors,
  mask: overviewValidity,
  output: {quantized: packedSignedBytes, validity}
}));
```

`vectors` has exactly `pixelCount * dimensions` float32 entries. `mask` and `output.validity` are
optional, each one uint32 per pixel. A masked pixel or any non-finite input component writes -128 in
every band and validity zero. Otherwise validity is one. `output.quantized` has exactly
`ceil(pixelCount * dimensions / 4)` uint32 words; unused final bytes are zero. The validity pass uses
one invocation per pixel, then the packing pass uses one per word, so odd vector lengths cannot cause
concurrent writes to the same word. Omitting `output.validity` allocates a pixelCount-word scratch.

## Limits and verification

Both classes accept positive integer pixel counts and 1–1,024 dimensions; total pixelCount is at most 4,294,967,295
across all chunks. Chunking cannot bypass this total uint32 row limit. The flattened component
count is at most 2^32 - 1 per chunk for uint32 addressing; the total across chunks may exceed it.
All supplied pixel fields must have matching chunk counts and complete-pixel boundaries. Packed
int8/binary16 words cannot cross a chunk boundary; only the final nonempty chunk may have word padding.
Each chunk binding, including encoder scratch, is checked against the device's storage binding limit
before nodes are created. A 262,144-pixel, 128-component float32 chunk occupies 128 MiB; split larger
arrays into aligned chunks on default devices. Widths 192, 256, 384 and 1,024 are covered by the
declaration and shader-audit scale spec; the 128-dimension limit belongs to reference covariance
operators, not these codecs. Decode takes one dispatch and encode takes two per nonempty chunk.
The per-contributor command budget is `1 + dispatchesPerChunk × nonemptyChunks ≤ 32,768`:

| Codec | Dispatches per chunk | Maximum nonempty chunks | First rejected chunk count |
| --- | --- | --- | --- |
| Encode | 2 | 16,383 | 16,384 |
| Decode | 1 | 32,767 | 32,768 |

Empty chunks do not consume dispatches; every binding and native packed-word boundary must still fit.
Shapes, chunk topology, formats, normalization and optional view presence are
compile-time; all view contents are per-frame. No parameter packer is needed. These raster operations
remain GPU-only, as recommended by the CPU suitability survey.

The independent NumPy fixture generator is
`modules/analysis/test/oracles/emb01-embedding-codecs-numpy.py`; run it with `/usr/bin/python3`.
The headless specs compare decoded/normalized vectors and exact packed output on signed extrema,
nearest-even rounding, saturation, zero and non-finite scales, masks, half-float subnormals and
exceptional values, partial workgroups and final-word padding. The existing AlphaEarth suite remains
regression coverage.

Presto uint16 decoding is **NOT DONE**: the [author's pipeline page](https://nasaharvest.github.io/presto-embeddings/)
and [dataset card](https://huggingface.co/datasets/izvonkov/Togo_Presto_Embeddings) confirm storage type,
but provide no scale/offset formula. The linked generation script redirects to Google sign-in. A
verified quantization formula is required before adding a Presto format; no scaling is inferred here.
