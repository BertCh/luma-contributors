// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPURasterEmbeddingPyramidLevel} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-pyramid-level';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';

function expectPublishedSumOrder(source: string): void {
  const store = 'vectorSums[vectorSumsOffset + index * 2u + band] = total[band]';
  expect(source).toContain(store);
  expect(source).toContain('let meanLengthValue');
  expect(source.indexOf(store)).toBeLessThan(source.indexOf('let meanLengthValue'));
}

it('R7 the aggregation assertion rejects deletion of the parent sum store', () => {
  const source =
    'vectorSums[vectorSumsOffset + index * 2u + band] = total[band];\nlet meanLengthValue = 1.0;';
  expect(() => expectPublishedSumOrder(source.replace(/vectorSums[^;]+;/, ''))).toThrow();
});

it('R2 captures the validated aggregation and output maps before caller mutation', () => {
  const context = createKernelAuditContext('pyramid-mutation');
  try {
    const source = context.view('float32', 4);
    const sums = context.view('float32', 2);
    const aggregation = {vectorSums: source, validCount: context.view('uint32', 2)};
    const output = {vectors: context.view('float32', 2), vectorSums: sums};
    const props = {sourceWidth: 2, sourceHeight: 1, dimensions: 2, scale: 2, aggregation, output};
    const contributor = new GPURasterEmbeddingPyramidLevel(props);
    aggregation.vectorSums = sums;
    output.vectorSums = source;
    props.dimensions = 128;
    expect(contributor.props.aggregation!.vectorSums).toBe(source);
    expect(contributor.props.output.vectorSums).toBe(sums);
    expect(contributor.props.dimensions).toBe(2);
    expect(() => contributor.getCommandNodes(context.graph)).not.toThrow();
  } finally {
    context.device.destroy();
  }
});

it('a parent binds child sums and support counts instead of normalized directions', () => {
  const context = createKernelAuditContext('embedding-pyramid-parent');
  try {
    const contributor = new GPURasterEmbeddingPyramidLevel({
      sourceWidth: 2,
      sourceHeight: 1,
      dimensions: 2,
      scale: 2,
      aggregation: {
        vectorSums: context.view('float32', 4),
        validCount: context.view('uint32', 2)
      },
      output: {
        vectors: context.view('float32', 2),
        vectorSums: context.view('float32', 2),
        validCount: context.view('uint32', 1),
        meanLength: context.view('float32', 1)
      }
    });
    const [kernel] = getGPUCommandNodeKernelSources(contributor.getCommandNodes(context.graph));
    expect(kernel.source).toContain('sourceSums[sourceSumsOffset + pixel * 2u + band]');
    expect(kernel.source).toContain('sourceValidCount[sourceValidCountOffset + pixel]');
    expect(kernel.source).toContain('validPixels += pixelCount;');
    // Publish raw sums before computing the displayed mean length.
    expectPublishedSumOrder(kernel.source);
    expect(() =>
      expectPublishedSumOrder(
        kernel.source.replace('vectorSums[vectorSumsOffset + index * 2u + band] = total[band]', '')
      )
    ).toThrow();
    expect(auditWGSLModule(kernel.source)).toEqual([]);
  } finally {
    context.device.destroy();
  }
});

it('raw child levels publish unnormalized sums including zero sums with nonzero support', () => {
  const context = createKernelAuditContext('embedding-pyramid-child');
  try {
    const contributor = new GPURasterEmbeddingPyramidLevel({
      sourceWidth: 3,
      sourceHeight: 1,
      dimensions: 2,
      scale: 2,
      vectors: context.view('float32', 6),
      output: {
        vectors: context.view('float32', 4),
        vectorSums: context.view('float32', 4),
        validCount: context.view('uint32', 2)
      }
    });
    const [kernel] = getGPUCommandNodeKernelSources(contributor.getCommandNodes(context.graph));
    expect(kernel.source).toContain(
      'vectorSums[vectorSumsOffset + index * 2u + band] = total[band]'
    );
    expect(kernel.source).toContain('validCount[validCountOffset + index] = validPixels;');
  } finally {
    context.device.destroy();
  }
});

it('validates sum/count layouts and rejects mixed source modes at construction', () => {
  const context = createKernelAuditContext('embedding-pyramid-validation');
  try {
    const aggregation = {
      vectorSums: context.view('float32', 4),
      validCount: context.view('uint32', 2)
    };
    const props = {
      sourceWidth: 2,
      sourceHeight: 1,
      dimensions: 2,
      scale: 2,
      aggregation,
      output: {vectors: context.view('float32', 2)}
    };
    expect(
      () => new GPURasterEmbeddingPyramidLevel({...props, vectors: context.view('float32', 4)})
    ).toThrow(/exactly one/);
    expect(
      () =>
        new GPURasterEmbeddingPyramidLevel({
          ...props,
          aggregation: {...aggregation, validCount: context.view('uint32', 1)}
        })
    ).toThrow(/aggregation.validCount/);
    expect(
      () =>
        new GPURasterEmbeddingPyramidLevel({
          ...props,
          aggregation: {...aggregation, vectorSums: context.view('float32', 2)}
        })
    ).toThrow(/aggregation.vectorSums/);
    expect(
      () =>
        new GPURasterEmbeddingPyramidLevel({
          ...props,
          output: {...props.output, vectorSums: context.view('float32', 1)}
        })
    ).toThrow(/output.vectorSums/);
  } finally {
    context.device.destroy();
  }
});
