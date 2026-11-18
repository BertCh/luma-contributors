// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPUSpatialErrorGMOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm-operation';
import {GPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares-operation';
import {GPUSpatialWeightsTracesOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces-operation';
import type {GPUContributorOperationViewGetter} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `merge-cap-${serial++}`, format, length);
  const weights = (rows: number) => ({
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', rows),
    weights: view('float32', rows)
  });
  return {device, graph, view, weights};
}

for (const family of ['error-gm', 'weights-traces'] as const) {
  it(`${family} snapshots chunk boundaries and weight view maps before validation`, () => {
    const {device, graph, view, weights} = createContext();
    try {
      const rows = 8192;
      const chunks = [
        {rowStart: 0, rowCount: 2, weights: weights(2)},
        {rowStart: 2, rowCount: rows - 2, weights: weights(rows - 2)}
      ];
      const originalOffsets = chunks[0].weights.offsets;
      const input = {rowCount: rows, chunks};
      const common = {weights: input, tileRowCount: 2};
      const contributor =
        family === 'weights-traces'
          ? new GPUSpatialWeightsTraces({
              ...common,
              maximumPower: 1,
              mode: 'stochastic',
              probeCount: 1,
              output: {powerTraces: view('float32', 1), powerSums: view('float32', 1)}
            })
          : new GPUSpatialErrorGM({
              ...common,
              estimator: 'kelejian-prucha',
              predictorCount: 1,
              predictors: view('float32', rows),
              response: view('float32', rows),
              output: {
                table: view('float32', 12),
                summary: view('float32', 32),
                status: {code: view('uint32', 1)}
              }
            });
      chunks[0].weights.offsets = view('uint32', 3);
      chunks.splice(
        0,
        chunks.length,
        {rowStart: 0, rowCount: 1, weights: weights(1)},
        {rowStart: 1, rowCount: rows - 1, weights: weights(rows - 1)}
      );
      const nodes = contributor.getCommandNodes(graph);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      const normTiles = kernels.filter(kernel => kernel.id.includes('norm-tiles'));
      expect(normTiles).toHaveLength(2);
      expect(normTiles[0].source).toContain('const WEIGHTS_ROW_COUNT: u32 = 2u;');
      expect(normTiles[1].source).toContain('const CHUNK_TILE_COUNT: u32 = 4095u;');
      expect(
        nodes.some(node =>
          node.resources?.some(
            resource =>
              'buffer' in resource &&
              (resource.buffer === originalOffsets || resource.buffer === originalOffsets.buffer)
          )
        )
      ).toBe(true);
    } finally {
      device.destroy();
    }
  });
}

for (const Contributor of [GPUSpatialLagMaximumLikelihood, GPUSpatialErrorMaximumLikelihood]) {
  for (const impossible of [false, true]) {
    it(`${Contributor.name} resolves nested trace tiles before resources: impossible=${impossible}`, () => {
      const {device, graph, view, weights} = createContext();
      try {
        const counts = impossible
          ? new Array<number>(4097).fill(1)
          : [...new Array<number>(4095).fill(1), 33];
        let rowStart = 0;
        const chunks = counts.map(rowCount => {
          const chunk = {rowStart, rowCount, weights: weights(rowCount)};
          rowStart += rowCount;
          return chunk;
        });
        const props = {
          id: 'nested-traces',
          weights: {rowCount: rowStart, chunks},
          tileRowCount: 64,
          predictorCount: 1,
          predictors: view('float32', rowStart),
          response: view('float32', rowStart),
          maximumPower: 2,
          traceMode: 'stochastic' as const,
          traceProbeCount: 1,
          output: {
            table: view('float32', 12),
            summary: view('float32', 32),
            status: {code: view('uint32', 1), converged: view('uint32', 1)}
          }
        };
        const allocate = vi.spyOn(graph, 'createTransientBuffer');
        if (impossible) {
          expect(() => new Contributor(props)).toThrow(/nested-traces.*tileCount.*4096/);
          expect(allocate).not.toHaveBeenCalled();
        } else {
          const contributor = new Contributor(props);
          const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
          const traceTiles = kernels.filter(kernel => kernel.id.includes('traces-norm-tiles'));
          expect(traceTiles).toHaveLength(4096);
          expect(traceTiles[4095].source).toContain('const CHUNK_TILE_COUNT: u32 = 1u;');
        }
      } finally {
        device.destroy();
      }
    });
  }
}

for (const family of ['error-gm', 'two-stage-least-squares', 'weights-traces'] as const) {
  for (const rows of [4096, 4097]) {
    it(`${family} operation lowering preserves the tile cap: ${rows}`, () => {
      const {device, view, weights} = createContext();
      try {
        const common = {
          id: 'lowered-merge',
          rowCount: rows,
          tileRowCount: 1,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          predictorCount: 1,
          predictors: 'predictors',
          response: 'response',
          output: {table: 'table', summary: 'summary', status: {code: 'code'}}
        };
        const operation =
          family === 'weights-traces'
            ? new GPUSpatialWeightsTracesOperation({
                ...common,
                maximumPower: 1,
                mode: 'stochastic',
                output: {powerTraces: 'powerTraces', powerSums: 'powerSums'}
              })
            : family === 'error-gm'
              ? new GPUSpatialErrorGMOperation(common)
              : new GPUSpatialTwoStageLeastSquaresOperation(common);
        const ports = {
          ...weights(rows),
          predictors: view('float32', rows),
          response: view('float32', rows),
          ...Object.fromEntries(
            Object.entries(operation.outputs).map(([name, output]) => [
              name,
              view(output.format as 'float32' | 'uint32', output.length)
            ])
          )
        };
        const getView = ((name: keyof typeof ports) =>
          ports[name]) as GPUContributorOperationViewGetter;
        const lower = () => operation.createContributor(getView);
        if (rows === 4096) expect(lower).not.toThrow();
        else expect(lower).toThrow(/^lowered-merge tileCount must be at most 4096$/);
      } finally {
        device.destroy();
      }
    });
  }
}

it('chunked traces operation lowering counts tiles at chunk boundaries', () => {
  const {device, view, weights} = createContext();
  try {
    const rows = 4096 * 64;
    const operation = new GPUSpatialWeightsTracesOperation({
      id: 'lowered-chunks',
      rowCount: rows,
      tileRowCount: 64,
      maximumPower: 1,
      mode: 'stochastic',
      weights: {
        rowCount: rows,
        chunks: [
          {
            rowStart: 0,
            rowCount: 1,
            weights: {offsets: 'firstOffsets', neighbors: 'firstNeighbors', weights: 'firstWeights'}
          },
          {
            rowStart: 1,
            rowCount: rows - 1,
            weights: {
              offsets: 'secondOffsets',
              neighbors: 'secondNeighbors',
              weights: 'secondWeights'
            }
          }
        ]
      },
      output: {powerTraces: 'powerTraces', powerSums: 'powerSums'}
    });
    const first = weights(1);
    const second = weights(rows - 1);
    const ports = {
      chunk0Offsets: first.offsets,
      chunk0Neighbors: first.neighbors,
      chunk0Weights: first.weights,
      chunk1Offsets: second.offsets,
      chunk1Neighbors: second.neighbors,
      chunk1Weights: second.weights,
      powerTraces: view('float32', 1),
      powerSums: view('float32', 1)
    };
    const getView = ((name: keyof typeof ports) =>
      ports[name]) as GPUContributorOperationViewGetter;
    expect(() => operation.createContributor(getView)).toThrow(
      /^lowered-chunks tileCount must be at most 4096$/
    );
  } finally {
    device.destroy();
  }
});

for (const family of ['error-gm', 'two-stage-least-squares', 'weights-traces'] as const) {
  for (const tileCount of [4096, 4097]) {
    it(`${family} accepts 4096 tiles and refuses 4097: ${tileCount}`, () => {
      const {device, view, weights} = createContext();
      try {
        const common = {
          id: 'bounded-merge',
          weights: weights(tileCount),
          tileRowCount: 1,
          predictorCount: 1,
          predictors: view('float32', tileCount),
          response: view('float32', tileCount)
        };
        const create = () => {
          if (family === 'weights-traces')
            return new GPUSpatialWeightsTraces({
              ...common,
              mode: 'stochastic',
              maximumPower: 1,
              output: {powerTraces: view('float32', 1), powerSums: view('float32', 1)}
            });
          const props = {
            ...common,
            output: {
              table: view('float32', 12),
              summary: view('float32', 32),
              status: {code: view('uint32', 1)}
            }
          };
          return family === 'error-gm'
            ? new GPUSpatialErrorGM(props)
            : new GPUSpatialTwoStageLeastSquares(props);
        };
        if (tileCount === 4096) expect(create().tileCount).toBe(4096);
        else expect(create).toThrow(/^bounded-merge tileCount must be at most 4096$/);
      } finally {
        device.destroy();
      }
    });
  }

  it(`${family} retains the scalable default layout at 2^24 rows`, () => {
    const {device, view, weights} = createContext();
    try {
      const rows = 2 ** 24;
      const common = {weights: weights(rows)};
      const contributor =
        family === 'weights-traces'
          ? new GPUSpatialWeightsTraces({
              ...common,
              maximumPower: 1,
              mode: 'stochastic',
              output: {powerTraces: view('float32', 1), powerSums: view('float32', 1)}
            })
          : (() => {
              const props = {
                ...common,
                predictorCount: 1,
                predictors: view('float32', rows),
                response: view('float32', rows),
                output: {
                  table: view('float32', 12),
                  summary: view('float32', 32),
                  status: {code: view('uint32', 1)}
                }
              };
              return family === 'error-gm'
                ? new GPUSpatialErrorGM(props)
                : new GPUSpatialTwoStageLeastSquares(props);
            })();
      expect(contributor.tileCount).toBe(family === 'weights-traces' ? 1024 : 4096);
    } finally {
      device.destroy();
    }
  });
}

for (const family of ['error-gm', 'weights-traces'] as const) {
  it(`${family} counts tiles separately across chunk boundaries`, () => {
    const {device, view, weights} = createContext();
    try {
      const rows = 4096 * 64;
      const common = {
        id: 'chunked-merge',
        tileRowCount: 64,
        weights: {
          rowCount: rows,
          chunks: [
            {rowStart: 0, rowCount: 1, weights: weights(1)},
            {rowStart: 1, rowCount: rows - 1, weights: weights(rows - 1)}
          ]
        }
      };
      const create = () =>
        family === 'weights-traces'
          ? new GPUSpatialWeightsTraces({
              ...common,
              maximumPower: 1,
              mode: 'stochastic',
              output: {powerTraces: view('float32', 1), powerSums: view('float32', 1)}
            })
          : new GPUSpatialErrorGM({
              ...common,
              estimator: 'kelejian-prucha',
              predictorCount: 1,
              predictors: view('float32', rows),
              response: view('float32', rows),
              output: {
                table: view('float32', 12),
                summary: view('float32', 32),
                status: {code: view('uint32', 1)}
              }
            });
      expect(create).toThrow(/^chunked-merge tileCount must be at most 4096$/);
    } finally {
      device.destroy();
    }
  });
}
