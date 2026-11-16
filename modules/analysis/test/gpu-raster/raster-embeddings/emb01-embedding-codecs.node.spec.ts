// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPURasterEmbeddingDequantize,
  GPURasterEmbeddingQuantize
} from '@luma-contributors/analysis/raster-analysis';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';

it('EMB-01 validates formats, view lengths, optional scales, masks and named dimensions at construction', () => {
  const context = createKernelAuditContext('emb01-validation');
  const {view, device} = context;
  try {
    const props = {
      pixelCount: 3,
      dimensions: 3,
      quantized: view('uint32', 3),
      output: {vectors: view('float32', 9)}
    };
    expect(() => new GPURasterEmbeddingDequantize({...props, format: 'tessera-int8'})).toThrow(
      /raster-embedding-dequantize scales/
    );
    expect(() => new GPURasterEmbeddingDequantize({...props, scales: view('float32', 3)})).toThrow(
      /scales must be supplied exactly/
    );
    expect(
      () =>
        new GPURasterEmbeddingDequantize({
          ...props,
          format: 'tessera-int8',
          scales: view('float32', 2)
        })
    ).toThrow(/scales must hold pixelCount/);
    expect(() => new GPURasterEmbeddingDequantize({...props, format: 'float16'})).toThrow(
      /quantized must hold ceil.*2/
    );
    expect(() => new GPURasterEmbeddingDequantize({...props, mask: view('uint32', 2)})).toThrow(
      /mask must hold pixelCount/
    );
    expect(() => new GPURasterEmbeddingDequantize({...props, format: 'unknown' as never})).toThrow(
      /format must be/
    );
    expect(() => new GPURasterEmbeddingDequantize({...props, pixelCount: 0})).toThrow(
      /pixelCount must be an integer in \[1,/
    );
    expect(
      () =>
        new GPURasterEmbeddingQuantize({
          pixelCount: 1,
          dimensions: 1025,
          vectors: view('float32', 1025),
          output: {quantized: view('uint32', 257)}
        })
    ).toThrow(/raster-embedding-quantize dimensions.*\[1, 1024\]/);
    expect(
      () =>
        new GPURasterEmbeddingQuantize({
          pixelCount: 1,
          dimensions: 3,
          vectors: view('float32', 3),
          output: {quantized: view('uint32', 2)}
        })
    ).toThrow(/output.quantized must hold ceil/);
    expect(
      () =>
        new GPURasterEmbeddingQuantize({
          pixelCount: 1,
          dimensions: 3,
          vectors: view('float32', 3),
          output: {quantized: view('uint32', 1), validity: view('uint32', 2)}
        })
    ).toThrow(/output.validity must have 1 rows/);
    expect(
      () =>
        new GPURasterEmbeddingDequantize({
          ...props,
          format: 'float16',
          quantized: view('uint32', 5)
        })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('EMB-01 refuses oversized bindings before creating nodes or scratch', () => {
  const context = createKernelAuditContext('emb01-binding-limit');
  const {graph, view, device} = context;
  try {
    const pixelCount = Math.floor(device.limits.maxStorageBufferBindingSize / (128 * 4)) + 1;
    const vectors = view('float32', pixelCount * 128);
    const quantized = view('uint32', (pixelCount * 128) / 4);
    const decode = new GPURasterEmbeddingDequantize({
      pixelCount,
      dimensions: 128,
      quantized,
      output: {vectors}
    });
    const encode = new GPURasterEmbeddingQuantize({
      pixelCount,
      dimensions: 128,
      vectors,
      output: {quantized}
    });
    expect(() => decode.getCommandNodes(graph)).toThrow(
      /raster-embedding-dequantize output.vectors.*limit/
    );
    expect(() => encode.getCommandNodes(graph)).toThrow(/raster-embedding-quantize vectors.*limit/);
  } finally {
    device.destroy();
  }
});

it('EMB-01 binding limits include the aligned prefix of offset views', () => {
  const context = createKernelAuditContext('emb01-offset-binding-limit');
  const {graph, view, device, buffer} = context;
  try {
    const byteLength = device.limits.maxStorageBufferBindingSize;
    const pixelCount = byteLength / (128 * 4);
    const vectors = graph.createDataView(buffer(byteLength + 4), {
      format: 'float32',
      length: pixelCount * 128,
      byteOffset: 4
    });
    const quantized = view('uint32', (pixelCount * 128) / 4);
    expect(() =>
      new GPURasterEmbeddingDequantize({
        pixelCount,
        dimensions: 128,
        quantized,
        output: {vectors}
      }).getCommandNodes(graph)
    ).toThrow(/output.vectors.*limit/);
    expect(() =>
      new GPURasterEmbeddingQuantize({
        pixelCount,
        dimensions: 128,
        vectors,
        output: {quantized}
      }).getCommandNodes(graph)
    ).toThrow(/vectors.*limit/);
  } finally {
    device.destroy();
  }
});
