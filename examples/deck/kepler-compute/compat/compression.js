// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Preserve both generations of the API using the real implementations. Kepler's loaders 4
// imports the combined codec classes; luma's loaders 5 uses separate compressor/decompressors.
export * from '../../../../node_modules/@loaders.gl/compression/dist/index.js';
export {
  Compression,
  NoCompression,
  DeflateCompression,
  GZipCompression,
  BrotliCompression,
  SnappyCompression,
  LZ4Compression,
  ZstdCompression
} from '@kepler-compat/compression';
