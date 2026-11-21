// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {CPUSpatialCrossValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-spatial-cross-validation-operation';
import {
  GPUSpatialCrossValidationOperation,
  type SpatialCrossValidationOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation-operation';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {constructSamplingContributor} from '../point-process/validation-sampling-operation-test-utils';
import fixtures from './spatial-cross-validation-operation-fixtures.json';
import {
  BLOCK_AUDIT_POSITIONS,
  BLOCK_AUDIT_BLOCKS,
  BLOCK_AUDIT_MASK,
  BLOCK_AUDIT_EXPECTED
} from './spatial-design-audit.test-utils';

it.each([
  'cpu',
  'gpu'
] as const)('W5-25 %s retains close standardized groups and constant columns', async form => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const standardize of [true, false]) {
    for (const featureCount of [1, 2]) {
      const low = Math.fround(0.1),
        high = low + 2 ** -25;
      const props: SpatialCrossValidationOperationProps = {
        rowCount: 8,
        positions: 'positions',
        blocks: 'blocks',
        foldCount: 3,
        seed: 0,
        design: {
          mode: 'environmental',
          features: 'features',
          featureCount,
          standardize,
          diagnostics: 'diagnostics'
        },
        output: {
          folds: 'folds',
          training: 'training',
          testing: 'testing',
          counts: 'counts',
          status: 'status'
        }
      };
      const actual = await runOperationGraph(device, {
        inputs: {
          positions: {
            format: 'float32x2',
            values: Float32Array.from({length: 16}, (_, index) => (index % 2 === 0 ? index / 2 : 0))
          },
          blocks: {format: 'uint32', values: new Uint32Array(8)},
          features: {
            format: 'float32',
            values: Float32Array.from(
              [3, low, high, 3, 3, high, high, high].flatMap(value =>
                featureCount === 1 ? [value] : [value, 7]
              )
            )
          }
        },
        operations: [
          form === 'cpu'
            ? new CPUSpatialCrossValidationOperation(props)
            : new GPUSpatialCrossValidationOperation(props)
        ],
        outputs: ['folds', 'training', 'testing', 'counts', 'status', 'diagnostics']
      });
      // Seeds 3, low, high give three constant clusters: SSE=0 and all 8 rows retained.
      for (const [port, expected] of Object.entries({
        folds: [0, 1, 2, 0, 0, 2, 2, 2],
        counts: [5, 3, 0, 7, 1, 0, 4, 4, 0],
        training: [0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
        testing: [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
        status: [0, 0, 0],
        diagnostics: [0, 2, 0, NaN]
      }))
        expect(
          Array.from(actual[port]),
          `${port}; standardized ${standardize}; columns ${featureCount}`
        ).toEqual(expected);
    }
  }
});

it.each([
  'cpu',
  'gpu'
] as const)('review B1/B2 %s retains residual centroids and tiny seed separations', async form => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixtures = [
    {
      features: [0, 1, 1 + 2 ** -23, 1 + 2 ** -22, 1 + 2 ** -22, 1 + 2 ** -22, 1 + 2 ** -21],
      folds: [0, 2, 2, 1, 1, 1, 1],
      counts: [6, 1, 0, 3, 4, 0, 5, 2, 0],
      objective: 7 * 2 ** -47
    },
    {
      features: [0, 2 ** -78, 1, 0, 2 ** -78, 1],
      folds: [0, 2, 1, 0, 2, 1],
      counts: [4, 2, 0, 4, 2, 0, 4, 2, 0],
      objective: 0
    }
  ];
  for (const fixture of fixtures) {
    for (const standardize of [false, true]) {
      const rows = fixture.features.length;
      const props: SpatialCrossValidationOperationProps = {
        rowCount: rows,
        positions: 'positions',
        blocks: 'blocks',
        foldCount: 3,
        seed: 0,
        design: {
          mode: 'environmental',
          features: 'features',
          featureCount: 1,
          standardize,
          diagnostics: 'diagnostics'
        },
        output: {
          folds: 'folds',
          training: 'training',
          testing: 'testing',
          counts: 'counts',
          status: 'status'
        }
      };
      const actual = await runOperationGraph(device, {
        inputs: {
          positions: {
            format: 'float32x2',
            values: Float32Array.from({length: rows * 2}, (_, index) =>
              index % 2 === 0 ? index / 2 : 0
            )
          },
          blocks: {format: 'uint32', values: new Uint32Array(rows)},
          features: {format: 'float32', values: Float32Array.from(fixture.features)}
        },
        operations: [
          form === 'cpu'
            ? new CPUSpatialCrossValidationOperation(props)
            : new GPUSpatialCrossValidationOperation(props)
        ],
        outputs: ['folds', 'training', 'testing', 'counts', 'status', 'diagnostics']
      });
      // Hand Lloyd means: (0,1+2.5u,1+.5u), SSE=3.5u²; tiny fixture has three constant groups.
      expect(Array.from(actual['folds'])).toEqual(fixture.folds);
      expect(Array.from(actual['counts'])).toEqual(fixture.counts);
      expect(Array.from(actual['status'])).toEqual([0, 0, 0]);
      expect(actual['diagnostics'][0]).toBe(0);
      expect(Number.isFinite(actual['diagnostics'][2])).toBe(true);
      if (!standardize)
        expect(Math.abs(actual['diagnostics'][2] - fixture.objective)).toBeLessThanOrEqual(
          fixture.objective * 2e-6
        );
      for (let fold = 0; fold < 3; fold++) {
        expect(Array.from(actual['testing'].slice(fold * rows, (fold + 1) * rows))).toEqual(
          fixture.folds.map(value => Number(value === fold))
        );
        expect(Array.from(actual['training'].slice(fold * rows, (fold + 1) * rows))).toEqual(
          fixture.folds.map(value => Number(value !== fold))
        );
      }
    }
  }
});

it.each([
  'cpu',
  'gpu'
] as const)('review 2 R1 %s preserves three distinct packed environmental groups', async form => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [low, high] of [
    [1, 1 + 2 ** -22],
    [1 + 2 ** -21, 10 + 2 ** -20]
  ]) {
    const props: SpatialCrossValidationOperationProps = {
      rowCount: 8,
      positions: 'positions',
      blocks: 'blocks',
      foldCount: 3,
      seed: 0,
      design: {
        mode: 'environmental',
        features: 'features',
        featureCount: 1,
        standardize: false,
        diagnostics: 'diagnostics'
      },
      output: {
        folds: 'folds',
        training: 'training',
        testing: 'testing',
        counts: 'counts',
        status: 'status'
      }
    };
    const actual = await runOperationGraph(device, {
      inputs: {
        positions: {
          format: 'float32x2',
          values: Float32Array.from({length: 16}, (_, index) => (index % 2 === 0 ? index / 2 : 0))
        },
        blocks: {format: 'uint32', values: new Uint32Array(8)},
        features: {
          format: 'float32',
          values: Float32Array.of(10, low, high, low, high, 10, low, high)
        }
      },
      operations: [
        form === 'cpu'
          ? new CPUSpatialCrossValidationOperation(props)
          : new GPUSpatialCrossValidationOperation(props)
      ],
      outputs: ['folds', 'training', 'testing', 'counts', 'status', 'diagnostics']
    });
    // Three distinct constant-value groups: their fitted means reproduce every value,
    // hence SSE=0. Farthest-first seed order is 10, low, high; buffers have radius zero.
    for (const [port, expected] of Object.entries({
      folds: [0, 1, 2, 1, 2, 0, 1, 2],
      training: [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
      testing: [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
      counts: [6, 2, 0, 5, 3, 0, 5, 3, 0],
      status: [0, 0, 0],
      diagnostics: [0, 2, 0, NaN]
    }))
      expect(
        Array.from(actual[port]),
        `${port}; low ${low}; high ${high}; diagnostics ${Array.from(actual['diagnostics'])}`
      ).toEqual(expected);
  }
});

it.each([
  'cpu',
  'gpu'
] as const)('R8 %s environmental folds preserve a representable translation without standardization', async form => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const shift of [0, 500000, -500000]) {
    const props: SpatialCrossValidationOperationProps = {
      rowCount: 8,
      positions: 'positions',
      blocks: 'blocks',
      foldCount: 2,
      seed: 0,
      design: {
        mode: 'environmental',
        features: 'features',
        featureCount: 1,
        standardize: false,
        diagnostics: 'diagnostics'
      },
      output: {
        folds: 'folds',
        training: 'training',
        testing: 'testing',
        counts: 'counts',
        status: 'status'
      }
    };
    const actual = await runOperationGraph(device, {
      inputs: {
        positions: {
          format: 'float32x2',
          values: Float32Array.from({length: 16}, (_, index) => (index % 2 === 0 ? index / 2 : 0))
        },
        blocks: {format: 'uint32', values: new Uint32Array(8)},
        features: {
          format: 'float32',
          values: Float32Array.from([0, 1, 2, 5, 6, 7, 9, 11], value => shift + value / 32)
        }
      },
      operations: [
        form === 'cpu'
          ? new CPUSpatialCrossValidationOperation(props)
          : new GPUSpatialCrossValidationOperation(props)
      ],
      outputs: ['folds', 'training', 'testing', 'counts', 'status', 'diagnostics']
    });
    // Stable centers 2/32 and 8.25/32; hand SSE=(14+14.75)/1024.
    for (const [port, expected] of Object.entries({
      folds: [0, 0, 0, 0, 1, 1, 1, 1],
      training: [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
      testing: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
      counts: [4, 4, 0, 4, 4, 0],
      status: [0, 0],
      diagnostics: [0, 2, 0.028076171875, NaN]
    }))
      expect(Array.from(actual[port]), `${port} shift ${shift}`).toEqual(expected);
  }
});

it('R8 GPU centering overflow reports arithmetic failure without dropping finite rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: SpatialCrossValidationOperationProps = {
    rowCount: 8,
    positions: 'positions',
    blocks: 'blocks',
    foldCount: 2,
    seed: 0,
    design: {
      mode: 'environmental',
      features: 'features',
      featureCount: 1,
      standardize: false,
      diagnostics: 'diagnostics'
    },
    output: {
      folds: 'folds',
      training: 'training',
      testing: 'testing',
      counts: 'counts',
      status: 'status'
    }
  };
  const actual = await runOperationGraph(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: Float32Array.from({length: 16}, (_, index) => (index % 2 === 0 ? index / 2 : 0))
      },
      blocks: {format: 'uint32', values: new Uint32Array(8)},
      features: {
        format: 'float32',
        values: Float32Array.of(-3e38, -3e38, -3e38, -3e38, 3e38, 3e38, 3e38, 3e38)
      }
    },
    operations: [new GPUSpatialCrossValidationOperation(props)],
    outputs: ['folds', 'training', 'testing', 'counts', 'status', 'diagnostics']
  });
  // All raw features are finite, but differences overflow. Omitting the overflowed
  // rows would give an infeasible constant cluster (status 2), rather than arithmetic status 1.
  for (const [port, expected] of Object.entries({
    folds: new Array(8).fill(0xffffffff),
    training: new Array(16).fill(0),
    testing: new Array(16).fill(0),
    counts: [0, 0, 0, 0, 0, 0],
    status: [3, 3],
    diagnostics: [1, 0, NaN, NaN]
  }))
    expect(Array.from(actual[port]), port).toEqual(expected);
});

it.each([
  'cpu',
  'gpu'
] as const)('R2 %s NNDM excludes the explicit buffer beyond its matched radius', async form => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: SpatialCrossValidationOperationProps = {
    rowCount: 8,
    positions: 'positions',
    blocks: 'blocks',
    foldCount: 8,
    bufferDistance: 3,
    design: {
      mode: 'nndm',
      predictionPositions: 'predictionPositions',
      predictionRowCount: 1,
      maximumDistance: 0,
      diagnostics: 'diagnostics'
    },
    output: {
      folds: 'folds',
      training: 'training',
      testing: 'testing',
      counts: 'counts',
      status: 'status'
    }
  };
  const actual = await runOperationGraph(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: Float32Array.from({length: 16}, (_, index) => (index % 2 === 0 ? index / 2 : 0))
      },
      blocks: {format: 'uint32', values: new Uint32Array(8)},
      predictionPositions: {format: 'float32x2', values: Float32Array.of(0, 0)}
    },
    operations: [
      form === 'cpu'
        ? new CPUSpatialCrossValidationOperation(props)
        : new GPUSpatialCrossValidationOperation(props)
    ],
    outputs: ['folds', 'training', 'testing', 'counts', 'status', 'diagnostics']
  });
  // Hand oracle: each fold tests its row; matched radius=1, explicit buffer=3,
  // so training is exactly |row-fold|>3. All eight diagnostic radii remain 1.
  const expected = {
    folds: [0, 1, 2, 3, 4, 5, 6, 7],
    training: [
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
      0, 0
    ],
    testing: [
      1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 1
    ],
    counts: [4, 1, 3, 3, 1, 4, 2, 1, 5, 1, 1, 6, 1, 1, 6, 2, 1, 5, 3, 1, 4, 4, 1, 3],
    status: [0, 0, 0, 0, 0, 0, 0, 0],
    diagnostics: [0, 0, NaN, NaN, 1, 1, 1, 1, 1, 1, 1, 1]
  };
  for (const [port, values] of Object.entries(expected))
    expect(Array.from(actual[port]), port).toEqual(values);
});

it.each([
  'cpu',
  'gpu'
] as const)('captures the %s fold operation before caller mutation and keeps every SciPy oracle port', async form => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: SpatialCrossValidationOperationProps = {
    id: 'captured-fold-operation',
    rowCount: 8,
    positions: 'positions',
    blocks: 'blocks',
    mask: 'mask',
    foldCount: 2,
    seed: 0,
    bufferDistance: 0,
    design: {
      mode: 'environmental',
      features: 'features',
      featureCount: 1,
      standardize: false,
      maximumIterations: 2,
      diagnostics: 'diagnostics'
    },
    output: {
      folds: 'folds',
      training: 'training',
      testing: 'testing',
      counts: 'counts',
      status: 'status'
    }
  };
  const operation =
    form === 'gpu'
      ? new GPUSpatialCrossValidationOperation(props)
      : new CPUSpatialCrossValidationOperation(props);
  props.rowCount = 1;
  props.bufferDistance = -1;
  props.mask = undefined;
  if (props.design?.mode === 'environmental') {
    props.design.featureCount = 2;
    props.design.maximumIterations = 1;
  }
  props.output.folds = 'changed';
  const actual = await runOperationGraph(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: Float32Array.from([0, 0, 1, 0, 2, 0, 3, 0, 10, 0, 11, 0, 12, 0, 13, 0])
      },
      blocks: {format: 'uint32', values: new Uint32Array(8)},
      mask: {format: 'uint32', values: new Uint32Array(8).fill(1)},
      features: {format: 'float32', values: Float32Array.from([0, 1, 1, 2, 10, 11, 11, 12])}
    },
    operations: [operation],
    outputs: ['folds', 'training', 'testing', 'counts', 'status', 'diagnostics']
  });
  // spatial-design-audit.oracle.py, SciPy vq at the two stable means: squared error 4.
  for (const [port, expected] of Object.entries({
    folds: [0, 0, 0, 0, 1, 1, 1, 1],
    training: [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    testing: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    counts: [4, 4, 0, 4, 4, 0],
    status: [0, 0],
    diagnostics: [0, 2, 4, NaN]
  }))
    expect(Array.from(actual[port]), port).toEqual(expected);
});

for (const mode of [
  'blocks',
  'groups',
  'environmental',
  'nndm',
  'knndm',
  'knndm-generated'
] as const)
  for (const empty of [false, true]) {
    it(`compares every cross-validation output, design=${mode}, empty=${empty}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = fixtures[mode === 'environmental' ? 'environmental' : 'knndm'];
      const positions =
        mode === 'nndm'
          ? new Float32Array([0, 0, 1, 0, 3, 0, 7, 0])
          : mode === 'knndm-generated'
            ? new Float32Array([0, 0, 1, 1, 2, 0, 8, 0, 9, 1, 10, 0])
            : new Float32Array(fixture.positions);
      const rowCount = positions.length / 2;
      const design: SpatialCrossValidationOperationProps['design'] =
        mode === 'blocks'
          ? undefined
          : mode === 'groups'
            ? {mode, diagnostics: 'diagnostics'}
            : mode === 'environmental'
              ? {mode, features: 'features', featureCount: 3, diagnostics: 'diagnostics'}
              : mode === 'nndm'
                ? {
                    mode,
                    maximumDistance: 4,
                    predictionPositions: 'predictionPositions',
                    predictionRowCount: 2,
                    diagnostics: 'diagnostics'
                  }
                : mode === 'knndm'
                  ? {
                      mode,
                      predictionPositions: 'predictionPositions',
                      predictionRowCount:
                        fixture === fixtures.knndm
                          ? fixtures.knndm.predictionPositions.length / 2
                          : 0,
                      candidates: 'candidates',
                      candidateCount: 4,
                      diagnostics: 'diagnostics'
                    }
                  : {
                      mode: 'knndm',
                      predictionPositions: 'predictionPositions',
                      predictionRowCount: 3,
                      candidateCount: 3,
                      diagnostics: 'diagnostics'
                    };
      const props: SpatialCrossValidationOperationProps = {
        id: 'cross-validation-comparison',
        rowCount,
        positions: 'positions',
        blocks: 'blocks',
        mask: 'mask',
        seed: 1,
        bufferDistance: 0.6,
        foldCount: mode === 'nndm' ? rowCount : mode === 'blocks' || mode === 'groups' ? 3 : 2,
        design,
        output: {
          folds: 'folds',
          training: 'training',
          testing: 'testing',
          counts: 'counts',
          status: 'status'
        }
      };
      const inputs: Record<string, OperationGraphInput> = {
        positions: {format: 'float32x2', values: positions},
        blocks: {
          format: 'uint32',
          values: Uint32Array.from(
            {length: rowCount},
            (_, row) => [100, 7, 100, 999, 7, 42, 9, 11][row]
          )
        },
        mask: {format: 'uint32', values: new Uint32Array(rowCount).fill(empty ? 0 : 1)}
      };
      if (mode === 'environmental')
        inputs['features'] = {
          format: 'float32',
          values: new Float32Array(fixtures.environmental.features)
        };
      if (mode === 'nndm' || mode === 'knndm' || mode === 'knndm-generated')
        inputs['predictionPositions'] = {
          format: 'float32x2',
          values: new Float32Array(
            mode === 'nndm'
              ? [20, 0, 25, 0]
              : mode === 'knndm'
                ? fixtures.knndm.predictionPositions
                : [0, 2, 4, 2, 8, 2]
          )
        };
      if (mode === 'knndm')
        inputs['candidates'] = {
          format: 'uint32',
          values: new Uint32Array(fixtures.knndm.candidates)
        };
      const outputs = [...Object.values(props.output), ...(design ? ['diagnostics'] : [])];
      const gpu = await runOperationGraph(device, {
        inputs,
        operations: [new GPUSpatialCrossValidationOperation(props)],
        outputs
      });
      const cpu = await runOperationGraph(device, {
        inputs,
        operations: [new CPUSpatialCrossValidationOperation(props)],
        outputs
      });
      for (const output of Object.values(props.output))
        expect(Array.from(gpu[output]), output).toEqual(Array.from(cpu[output]));
      if (design)
        for (const [index, expected] of cpu['diagnostics'].entries()) {
          const actual = gpu['diagnostics'][index];
          if (Number.isNaN(expected)) expect(Number.isNaN(actual)).toBe(true);
          // Status, iterations, selected candidate, and the NNDM finite sentinel compare exactly.
          else if (
            index === 0 ||
            index === 1 ||
            index === 3 ||
            expected === Math.fround(3.402823e38)
          )
            expect(actual).toBe(expected);
          else
            expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
              2e-6 + 2e-5 * Math.abs(expected)
            );
        }
      if (empty) expect(cpu['status'].every(status => status === 3)).toBe(true);
      else if (mode === 'environmental' || mode === 'knndm')
        expect(Array.from(cpu['folds'])).toEqual(fixture.expectedFolds);
      else if (mode === 'nndm')
        expect(Array.from(cpu['counts'])).toEqual([0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 1, 2]);
      else expect(new Set(cpu['folds']).size).toBe(props.foldCount);
      if (!empty && (mode === 'environmental' || mode === 'knndm')) {
        // Compare the GPU directly with the independent SciPy/NumPy fixture on every port.
        for (const [port, expected] of Object.entries({
          folds: fixture.expectedFolds,
          training: fixture.expectedTraining,
          testing: fixture.expectedTesting,
          counts: fixture.expectedCounts,
          status: fixture.expectedStatus
        }))
          expect(Array.from(gpu[port]), port).toEqual(expected);
        const expectedObjective =
          mode === 'environmental'
            ? fixtures.environmental.objective
            : fixtures.knndm.objectives[fixtures.knndm.selected]!;
        expect(Number.isFinite(gpu['diagnostics'][2])).toBe(true);
        expect(Math.abs(gpu['diagnostics'][2] - expectedObjective)).toBeLessThanOrEqual(
          2e-6 + 2e-5 * Math.abs(expectedObjective)
        );
      }
    });
  }

it('audit-validation environmental iteration budgets match the independent two-step fixture', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = {
    positions: {
      format: 'float32x2' as const,
      values: new Float32Array([0, 0, 1, 0, 2, 0, 3, 0, 10, 0, 11, 0, 12, 0, 13, 0, 99, 0])
    },
    blocks: {format: 'uint32' as const, values: new Uint32Array(9)},
    mask: {format: 'uint32' as const, values: new Uint32Array([1, 1, 1, 1, 1, 1, 1, 1, 0])},
    features: {
      format: 'float32' as const,
      values: new Float32Array([0, 1, 1, 2, 10, 11, 11, 12, 99])
    }
  };
  for (const maximumIterations of [1, 2]) {
    const props: SpatialCrossValidationOperationProps = {
      rowCount: 9,
      positions: 'positions',
      blocks: 'blocks',
      mask: 'mask',
      foldCount: 2,
      design: {
        mode: 'environmental',
        features: 'features',
        featureCount: 1,
        standardize: false,
        maximumIterations,
        diagnostics: 'diagnostics'
      },
      output: {
        folds: 'folds',
        training: 'training',
        testing: 'testing',
        counts: 'counts',
        status: 'status'
      }
    };
    const actual = await runOperationGraph(device, {
      inputs,
      operations: [new GPUSpatialCrossValidationOperation(props)],
      outputs: [...Object.values(props.output), 'diagnostics']
    });
    const converged = maximumIterations === 2;
    // spatial-design-audit.oracle.py: all arithmetic is exact on these dyadic inputs.
    expect(Array.from(actual['diagnostics'])).toEqual([
      converged ? 0 : 3,
      maximumIterations,
      converged ? 4 : 12,
      NaN
    ]);
    expect(Array.from(actual['folds'])).toEqual(
      converged ? [0, 0, 0, 0, 1, 1, 1, 1, 0xffffffff] : new Array(9).fill(0xffffffff)
    );
    expect(Array.from(actual['training'])).toEqual(
      converged ? [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0] : new Array(18).fill(0)
    );
    expect(Array.from(actual['testing'])).toEqual(
      converged ? [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0] : new Array(18).fill(0)
    );
    expect(Array.from(actual['counts'])).toEqual(
      converged ? [4, 4, 0, 4, 4, 0] : [0, 0, 0, 0, 0, 0]
    );
    expect(Array.from(actual['status'])).toEqual(converged ? [0, 0] : [3, 3]);
  }
});

it('audit-validation GPU block folds match exact Philox and inclusive SciPy buffer masks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: SpatialCrossValidationOperationProps = {
    rowCount: 9,
    positions: 'positions',
    blocks: 'blocks',
    mask: 'mask',
    foldCount: 4,
    seed: 1234,
    bufferDistance: 1,
    output: {
      folds: 'folds',
      training: 'training',
      testing: 'testing',
      counts: 'counts',
      status: 'status'
    }
  };
  for (const shift of [0, 32]) {
    const actual = await runOperationGraph(device, {
      inputs: {
        positions: {
          format: 'float32x2',
          values: Float32Array.from(BLOCK_AUDIT_POSITIONS, value => value + shift)
        },
        blocks: {format: 'uint32', values: Uint32Array.from(BLOCK_AUDIT_BLOCKS)},
        mask: {format: 'uint32', values: Uint32Array.from(BLOCK_AUDIT_MASK)}
      },
      operations: [new GPUSpatialCrossValidationOperation(props)],
      outputs: Object.values(props.output)
    });
    // Distances 0 and exactly 1 are excluded, on either translated copy.
    for (const [port, expected] of Object.entries(BLOCK_AUDIT_EXPECTED))
      expect(Array.from(actual[port]), port).toEqual(expected);
  }
});

it('keeps the last GPU-supported candidate index in range and rejects the reported excessive count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: SpatialCrossValidationOperationProps = {
    id: 'last-candidate-comparison',
    rowCount: 2,
    positions: 'positions',
    blocks: 'blocks',
    foldCount: 2,
    design: {
      mode: 'knndm',
      candidateCount: 64,
      candidates: 'candidates',
      predictionPositions: 'predictionPositions',
      predictionRowCount: 1,
      diagnostics: 'diagnostics'
    },
    output: {
      folds: 'folds',
      training: 'training',
      testing: 'testing',
      counts: 'counts',
      status: 'status'
    }
  };
  const candidates = new Uint32Array(128).fill(0xffffffff);
  candidates.set([0, 1], 126);
  const inputs = {
    positions: {format: 'float32x2' as const, values: new Float32Array([0, 0, 1, 0])},
    blocks: {format: 'uint32' as const, values: new Uint32Array(2)},
    predictionPositions: {format: 'float32x2' as const, values: new Float32Array([0, 0])},
    candidates: {format: 'uint32' as const, values: candidates}
  };
  const outputs = [...Object.values(props.output), 'diagnostics'];
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUSpatialCrossValidationOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUSpatialCrossValidationOperation(props)],
    outputs
  });
  for (const output of outputs) expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
  expect(Array.from(cpu['diagnostics'])).toEqual([0, 64, 1, 63]);
  // The reported 16,777,220-candidate input is outside the GPU constructor's 64-candidate
  // domain. Exercise that rejection without uploading a vector larger than a core binding.
  const rejectedProps: SpatialCrossValidationOperationProps = {
    ...props,
    design: {
      mode: 'knndm',
      candidateCount: 16777220,
      candidates: 'candidates',
      predictionPositions: 'predictionPositions',
      predictionRowCount: 1,
      diagnostics: 'diagnostics'
    }
  };
  expect(() => new CPUSpatialCrossValidationOperation(rejectedProps)).toThrow(/candidateCount/);
  expect(() =>
    constructSamplingContributor(new GPUSpatialCrossValidationOperation(rejectedProps), {
      positions: {format: 'float32x2', length: 2},
      blocks: {format: 'uint32', length: 2},
      predictionPositions: {format: 'float32x2', length: 1},
      candidates: {format: 'uint32', length: 33554440}
    })
  ).toThrow(/candidateCount/);
});

it('compares generated kNNDM restarts when candidate one beats candidate zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = fixtures.generated;
  const inputs = {
    positions: {format: 'float32x2' as const, values: new Float32Array(fixture.positions)},
    blocks: {format: 'uint32' as const, values: new Uint32Array(6)},
    predictionPositions: {
      format: 'float32x2' as const,
      values: new Float32Array(fixture.predictionPositions)
    }
  };
  for (const candidateCount of [1, 3]) {
    const props: SpatialCrossValidationOperationProps = {
      id: 'competing-restarts',
      rowCount: 6,
      positions: 'positions',
      blocks: 'blocks',
      foldCount: 2,
      seed: 1,
      bufferDistance: 0.6,
      design: {
        mode: 'knndm',
        candidateCount,
        predictionPositions: 'predictionPositions',
        predictionRowCount: 2,
        diagnostics: 'diagnostics'
      },
      output: {
        folds: 'folds',
        training: 'training',
        testing: 'testing',
        counts: 'counts',
        status: 'status'
      }
    };
    const outputs = [...Object.values(props.output), 'diagnostics'];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUSpatialCrossValidationOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUSpatialCrossValidationOperation(props)],
      outputs
    });
    for (const output of Object.values(props.output))
      expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
    const selected = candidateCount === 1 ? 0 : 1;
    expect(Array.from(cpu['folds'])).toEqual(fixture.partitions[selected]);
    for (const rows of [gpu, cpu]) {
      expect(rows['diagnostics'][0]).toBe(0);
      expect(rows['diagnostics'][1]).toBe(candidateCount);
      expect(rows['diagnostics'][3]).toBe(selected);
      // Positive Wasserstein objectives: 2e-5 relative, with no absolute floor.
      expect(Math.abs(rows['diagnostics'][2] - fixture.objectives[selected])).toBeLessThanOrEqual(
        2e-5 * fixture.objectives[selected]
      );
    }
  }
});

it('compares two NNDM submissions sharing a diagnostics buffer, including unwritten radii on failure', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: SpatialCrossValidationOperationProps = {
    id: 'reused-nndm-diagnostics',
    rowCount: 4,
    positions: 'positions',
    blocks: 'blocks',
    mask: 'mask',
    foldCount: 4,
    design: {
      mode: 'nndm',
      maximumDistance: 4,
      predictionPositions: 'predictionPositions',
      predictionRowCount: 2,
      diagnostics: 'diagnostics'
    },
    output: {
      folds: 'folds',
      training: 'training',
      testing: 'testing',
      counts: 'counts',
      status: 'status'
    }
  };
  const inputs = {
    positions: {format: 'float32x2' as const, values: new Float32Array([0, 0, 1, 0, 3, 0, 7, 0])},
    blocks: {format: 'uint32' as const, values: new Uint32Array(4)},
    mask: {format: 'uint32' as const, values: new Uint32Array(4).fill(1)},
    predictionPositions: {format: 'float32x2' as const, values: new Float32Array([20, 0, 25, 0])}
  };
  // runOperationGraph destroys its graph after one run. Bind the GPU operation once to
  // caller-owned buffers here so both submissions really reuse the same diagnostics view.
  const graph = new GPUCommandGraph(device);
  const operation = new GPUSpatialCrossValidationOperation(props);
  const inputBuffers = Object.fromEntries(
    Object.entries(inputs).map(([port, input]) => [port, createInputBuffer(device, input.values)])
  );
  const outputBuffers = Object.fromEntries(
    Object.entries(operation.outputs).map(([port, output]) => [
      port,
      createOutputBuffer(device, output.length)
    ])
  );
  try {
    const views = Object.fromEntries([
      ...Object.entries(inputs).map(([port, input]) => [
        port,
        importGraphBuffer(
          graph,
          port,
          inputBuffers[port],
          input.format,
          input.values.length / (input.format === 'float32x2' ? 2 : 1)
        )
      ]),
      ...Object.entries(operation.outputs).map(([port, output]) => [
        port,
        importGraphBuffer(graph, port, outputBuffers[port], output.format, output.length)
      ])
    ]);
    graph.add(operation.createContributor(port => views[port] as never));
    const compiled = graph.compile();
    try {
      let precedingRadii: number[] = [];
      for (const empty of [false, true]) {
        inputs.mask.values.fill(empty ? 0 : 1);
        inputBuffers['mask'].write(inputs.mask.values);
        submitGraph(device, compiled, undefined);
        const gpu: Record<string, number[]> = {};
        for (const [port, buffer] of Object.entries(outputBuffers))
          gpu[port] = port === 'diagnostics' ? await readFloat32(buffer) : await readUint32(buffer);
        const cpu = await runOperationGraph(device, {
          inputs,
          operations: [new CPUSpatialCrossValidationOperation(props)],
          outputs: [...Object.values(props.output), 'diagnostics']
        });
        for (const output of Object.values(props.output))
          expect(gpu[output]).toEqual(Array.from(cpu[output]));
        for (let index = 0; index < 4; index++) {
          const expected = cpu['diagnostics'][index];
          if (Number.isNaN(expected)) expect(Number.isNaN(gpu['diagnostics'][index])).toBe(true);
          else expect(gpu['diagnostics'][index]).toBe(expected);
        }
        if (empty) {
          expect(Array.from(cpu['diagnostics'])).toEqual([1, 0, NaN, NaN, 0, 0, 0, 0]);
          // Kernel hypothesis to confirm on-device: an early return leaves these four words
          // unchanged. This is a documented difference from the stateless CPU result.
          expect(gpu['diagnostics'].slice(4)).toEqual(precedingRadii);
        } else {
          precedingRadii = gpu['diagnostics'].slice(4);
          const expectedRadii = [7, 6, Math.fround(3.402823e38), 6];
          expect(Array.from(cpu['diagnostics'].subarray(4))).toEqual(expectedRadii);
          for (const [index, expected] of expectedRadii.entries()) {
            const actual = precedingRadii[index];
            // The sentinel is a copied literal. Finite radii come from WGSL distance(),
            // which need not return an exact integer even on this axis-aligned frame.
            // Apply the documented 2e-5 relative diagnostic tolerance to those positive
            // distances only; the next submission must retain the actual GPU words exactly.
            if (expected === Math.fround(3.402823e38)) expect(actual).toBe(expected);
            else expect(Math.abs(actual - expected)).toBeLessThanOrEqual(2e-5 * expected);
          }
        }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    operation.destroy();
    for (const buffer of [...Object.values(inputBuffers), ...Object.values(outputBuffers)])
      buffer.destroy();
  }
});
