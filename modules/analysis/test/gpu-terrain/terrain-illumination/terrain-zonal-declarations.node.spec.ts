// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createHash} from 'node:crypto';
import {
  createTransientView,
  GPUCommandGraph,
  GraphDataView,
  GraphTextureView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {GPUTerrainHorizon} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {GPURasterZonalStatistics} from '../../../src/raster-analysis/raster-zonal-statistics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function getDeclaration(nodes: readonly GPUCommandNode[]) {
  return {
    nodes: nodes.map(node => ({
      id: node.id,
      resources: node.resources?.map(resource => {
        if ('buffer' in resource) {
          const view = resource.buffer;
          return {
            usage: resource.usage,
            buffer:
              view instanceof GraphDataView
                ? {
                    id: view.buffer.id,
                    format: view.format,
                    byteOffset: view.byteOffset,
                    byteStride: view.byteStride,
                    length: view.length,
                    rowByteLength: view.rowByteLength
                  }
                : {id: view.id, byteLength: view.byteLength}
          };
        }
        if ('texture' in resource) {
          const texture = resource.texture;
          return {
            usage: resource.usage,
            texture:
              texture instanceof GraphTextureView
                ? {
                    id: texture.texture.id,
                    baseMipLevel: texture.baseMipLevel,
                    mipLevelCount: texture.mipLevelCount,
                    baseArrayLayer: texture.baseArrayLayer,
                    arrayLayerCount: texture.arrayLayerCount
                  }
                : {id: texture.id}
          };
        }
        return {usage: resource.usage, externalTexture: resource.externalTexture.id};
      })
    })),
    pipelines: getGPUCommandNodeKernelSources(nodes).map(
      ({id, source, constants, shaderLayout, entryPoint}) => ({
        id,
        source,
        constants: Object.entries(constants).sort(([first], [second]) =>
          first < second ? -1 : first > second ? 1 : 0
        ),
        shaderLayout,
        entryPoint
      })
    )
  };
}

// URT-03 added population variance after these sample-variance declaration pins were recorded.
// For these fixtures no population output is selected. Check every exact added WGSL fragment
// before projecting that known extension back to the pinned form. The legacy hash still pins
// all other source bytes, ordered resources, binding layouts, entry points and constants.
// Independent divisor contract: population = corrected SSD / total for n >= 1;
// sample = corrected SSD / (total - sum(w^2)/total) for n >= 2 and a positive divisor.
const POPULATION_VARIANCE_EXTENSION = [
  ['  var populationVariance = 0x7fc00000u;\n', ''],
  [
    'if (valueCount >= 1u && total > 0.0) {',
    'if (valueCount >= 2u && denominator > 0.0 && total > 0.0) {'
  ],
  [
    `    populationVariance = bitcast<u32>(numerator / total);
    if (valueCount >= 2u && denominator > 0.0) {
      let result = numerator / denominator;
      variance = bitcast<u32>(result);
      deviation = bitcast<u32>(sqrt(result));
    }`,
    `    let result = numerator / denominator;
    variance = bitcast<u32>(result);
    deviation = bitcast<u32>(sqrt(result));`
  ],
  [
    '  }\n  \n  variances[variancesOffset + index] = variance;',
    '  }\n  variances[variancesOffset + index] = variance;'
  ]
] as const;

function projectPopulationVarianceExtension(source: string): string {
  for (const [current, legacy] of POPULATION_VARIANCE_EXTENSION) {
    // Missing, duplicated or changed extension text is a failure, not ignored source drift.
    expect(source.split(current)).toHaveLength(2);
    source = source.replace(current, legacy);
  }
  return source;
}

function getDigest(nodes: readonly GPUCommandNode[], legacySampleVariancePin = false): string {
  const declaration = getDeclaration(nodes);
  if (legacySampleVariancePin) {
    const finalizers = declaration.pipelines.filter(pipeline => pipeline.id.endsWith('-variances'));
    expect(finalizers).toHaveLength(1);
    for (const finalizer of finalizers)
      finalizer.source = projectPopulationVarianceExtension(finalizer.source);
  }
  return createHash('sha256').update(JSON.stringify(declaration)).digest('hex');
}

// Original WGSL/ID pins were captured before RT-113 extraction. Round one extends the pin to
// ordered views/ranges, shader binding layouts, entry points and actual override constants.
// The geographic WGSL intentionally changes for the B3 opposite-pole precision fix.
// The URT-03 population-variance extension is checked explicitly above; these pins stay unchanged.
const EXPECTED: Record<string, string> = {
  'march-float32': '20102b846c855d7f3f924252fb55ae649c00ac2586004ec50af0a6344125b0c8',
  'sweep-unorm16': '40a55a00fcf98f068b8f7e1cb45eac4443adc4fc7194746b1c5ac59f3e4267c2',
  'zonal-sorted': '922b24e558c9569bc3a4d4d64ac9ea6ac6335c5bc9b90bb3fc231341e608cda4',
  'zonal-atomic': '1152a889ea04b7942361a58a995ddccc7d95c7d6dd0bb3f1d3003a9cc98616c2',
  'zonal-geographic': 'c498c73bcf57616be68723e3504bfe58f13981b0453d07b512b1b6accfd9053a',
  'zonal-dictionary': 'd91a560d708b36caa0a2fcd79410e70e94f7941df8503b8489798fc509b16e6e'
};
for (const route of [
  'march-float32',
  'sweep-unorm16',
  'zonal-sorted',
  'zonal-atomic',
  'zonal-geographic',
  'zonal-dictionary'
]) {
  it(`RT-113 preserves graph declarations: ${route}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format, length = 12) =>
      createTransientView(graph, name, format, length);
    const grid = makeGPURasterMetadata({width: 4, height: 3});
    const elevation = {
      id: 'values',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: view('values', 'float32')},
      validity: view('validity', 'uint32')
    };
    const contributor = route.startsWith('zonal')
      ? new GPURasterZonalStatistics({
          grid,
          zones: view('zones', 'uint32'),
          values: elevation,
          zoneCapacity: 2,
          sumOrder: route === 'zonal-atomic' ? 'atomic' : 'sorted',
          ...(route === 'zonal-geographic'
            ? {cellSizeMode: 'geographic' as const, parameters: view('parameters', 'float32', 7)}
            : {}),
          ...(route === 'zonal-dictionary'
            ? {zoneDictionary: view('dictionary', 'uint32', 2)}
            : {}),
          output: {
            cellCounts: view('counts', 'uint32', 2),
            sums: view('sums', 'float32', 2),
            means: view('means', 'float32', 2),
            variances: view('variances', 'float32', 2),
            ...(route === 'zonal-dictionary'
              ? {
                  sourceZoneIds: view('source-ids', 'uint32', 2),
                  denseZoneRows: view('dense-rows', 'uint32', 2),
                  status: {invalidCount: view('invalid', 'uint32', 1)}
                }
              : {})
          }
        })
      : new GPUTerrainHorizon({
          grid,
          elevation,
          maximumRadius: 2,
          directionCount: 8,
          algorithm: route === 'march-float32' ? 'march' : 'sweep',
          horizonFormat: route === 'march-float32' ? 'float32' : 'unorm16',
          parameters: view('parameters', 'float32', 12),
          output: {
            horizon:
              route === 'march-float32'
                ? view('horizon', 'float32', 96)
                : view('horizon', 'uint32', 48),
            skyViewFactor: view('svf', 'float32'),
            negativeOpenness: view('negative', 'float32'),
            anisotropicSkyViewFactor: view('asvf', 'float32')
          }
        });
    try {
      const nodes = contributor.getCommandNodes(graph);
      const hash = getDigest(nodes, route.startsWith('zonal'));
      expect(hash).toBe(EXPECTED[route] ?? 'PIN');
    } finally {
      device.destroy();
    }
  });
}

it('t1 population extension projection rejects a changed population divisor', () => {
  const extension = POPULATION_VARIANCE_EXTENSION.map(([current]) => current).join('\n');
  expect(() => projectPopulationVarianceExtension(extension)).not.toThrow();
  expect(() =>
    projectPopulationVarianceExtension(
      extension.replace('numerator / total', 'numerator / denominator')
    )
  ).toThrow();
});

it('S3 structural pin detects swapping two same-format caller output views', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const first = createTransientView(graph, 'first-output', 'float32', 1);
  const second = createTransientView(graph, 'second-output', 'float32', 1);
  const nodes = (swapped: boolean) => [
    createWGSLKernelNode(graph, {
      id: 'pin-bindings',
      operation: 'PinBindings',
      invocationCount: 1,
      bindings: [
        {name: 'first', view: swapped ? second : first, type: 'f32', access: 'read_write'},
        {name: 'second', view: swapped ? first : second, type: 'f32', access: 'read_write'}
      ],
      body: 'first[index] = 1.0; second[index] = 2.0;'
    })
  ];
  expect(getDigest(nodes(false))).not.toBe(getDigest(nodes(true)));
});

it('S3 structural pin detects changed pipeline override values', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const output = createTransientView(graph, 'output', 'float32', 1);
  const nodes = (scale: number) => [
    createWGSLKernelNode(graph, {
      id: 'pin-overrides',
      operation: 'PinOverrides',
      invocationCount: 1,
      bindings: [{name: 'result', view: output, type: 'f32', access: 'read_write'}],
      declarations: 'override SCALE: f32;',
      constants: {SCALE: scale},
      body: 'result[index] = SCALE;'
    })
  ];
  expect(getDigest(nodes(1))).not.toBe(getDigest(nodes(2)));
});
