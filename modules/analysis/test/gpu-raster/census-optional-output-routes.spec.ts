// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPURasterFillNoData} from '../../src/raster-analysis/gpu-raster-fill-nodata';
import {GPURasterFocalPairs} from '../../src/raster-analysis/focal-pairs/gpu-raster-focal-pairs';
import {GPURasterWalkingCostOperation} from '../../src/raster-analysis/cost-distance/gpu-raster-walking-cost-operation';
import {CPURasterWalkingCostOperation} from '../../src/raster-analysis/cost-distance/cpu-raster-walking-cost-operation';
import {getGPURasterWalkingCostParameterValues} from '../../src/raster-analysis/gpu-raster-walking-cost';
import {
  GPUScalarFieldRidges,
  getGPUScalarFieldRidgesParameterValues
} from '../../src/raster-analysis/lagrangian-diagnostics/gpu-scalar-field-ridges';
import {
  GPURasterCellStatistics,
  GPURasterReclassify,
  getGPURasterCellStatisticsParameterValues,
  getGPURasterReclassifyParameterValues
} from '../../src/raster-analysis/raster-algebra/index';
import {GPURasterJoin} from '../../src/raster-analysis/polygon-rasterization/gpu-raster-join';
import {getGPUPolygonRasterizationParameterValues} from '../../src/raster-analysis/polygon-rasterization/polygon-rasterization-parameters';
import {GPURasterEmbeddingChange} from '../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-change';
import {GPURasterEmbeddingLinearProbe} from '../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-linear-probe';
import {GPURasterEmbeddingProjection} from '../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-projection';
import {GPURasterEmbeddingPyramidLevel} from '../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-pyramid-level';
import {GPURasterEmbeddingZoneMean} from '../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-zone-mean';
import {makeGPURasterMetadata} from '../../src/raster-analysis/raster-grid/index';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';
import {
  createCensusRasterHarness,
  createCensusRasterJoinFixture,
  expectCensusValues
} from './census-raster-oracle-utils';

it('census fill omits invalidCount without suppressing values or provenance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const result = await harness.run(
    new GPURasterFillNoData({
      grid: makeGPURasterMetadata({width: 3, height: 1}),
      input: {
        id: 'gap',
        format: 'float32',
        storage: {kind: 'buffer', values: harness.input('float32', [1, NaN, 3])}
      },
      maximumDistance: 1,
      method: 'inverse-distance',
      output: {
        values: harness.output('values', 'float32', 3),
        validity: harness.output('validity', 'uint32', 3),
        filled: harness.output('filled', 'uint32', 3),
        status: {}
      }
    })
  );
  // Two equidistant original donors give (1+3)/2=2; only the middle cell was filled.
  expectCensusValues(result['values'], [1, 2, 3]);
  expectCensusValues(result['validity'], [1, 1, 1]);
  expectCensusValues(result['filled'], [0, 1, 0]);
});

const FOCAL_EXPECTATIONS = {
  covariance: [1, 2, 1],
  correlation: [1, 1, 1],
  slope: [2, 2, 2],
  intercept: [3, 3, 3],
  rSquared: [1, 1, 1],
  residualSumSquares: [0, 0, 0]
};
for (const metric of Object.keys(FOCAL_EXPECTATIONS) as (keyof typeof FOCAL_EXPECTATIONS)[]) {
  it(`census focal ${metric} runs without sibling metric outputs`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createCensusRasterHarness(device);
    const result = await harness.run(
      new GPURasterFocalPairs({
        grid: makeGPURasterMetadata({width: 3, height: 1}),
        predictor: {
          id: 'predictor',
          format: 'float32',
          storage: {kind: 'buffer', values: harness.input('float32', [1, 2, 3])}
        },
        response: {
          id: 'response',
          format: 'float32',
          storage: {kind: 'buffer', values: harness.input('float32', [5, 7, 9])}
        },
        radius: 1,
        output: {
          pairedCount: harness.output('count', 'uint32', 3),
          status: harness.output('status', 'uint32', 3),
          [metric]: harness.output('metric', 'float32', 3)
        }
      })
    );
    // Clipped windows have x=[1,2], [1,2,3], [2,3] and y=2x+3.
    // Sample covariances are 1,2,1; the exact fit has slope 2, intercept 3, r^2=1 and RSS=0.
    expectCensusValues(result['count'], [2, 3, 2]);
    expectCensusValues(result['status'], [0, 0, 0]);
    expectCensusValues(result['metric'], FOCAL_EXPECTATIONS[metric], 2e-6);
  });
}

it('census cell statistics writes minimum without requesting mean', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const result = await harness.run(
    new GPURasterCellStatistics({
      cellCount: 2,
      layerCount: 2,
      stack: harness.input('float32', [1, 5, 3, 7]),
      parameters: harness.input('float32', Array.from(getGPURasterCellStatisticsParameterValues())),
      output: {minimum: harness.output('minimum', 'float32', 2)}
    })
  );
  // Independent per-column minima of the two layers [1,5] and [3,7].
  expectCensusValues(result['minimum'], [1, 5]);
});

for (const port of ['classes', 'reclassified', 'classCounts'] as const) {
  it(`census reclassify supports ${port} as its only output`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createCensusRasterHarness(device);
    const result = await harness.run(
      new GPURasterReclassify({
        values: harness.input('float32', [4, 1, 4, 3]),
        breaks: harness.input('float32', [1, 3]),
        classValues: harness.input('float32', [10, 20, 30]),
        parameters: harness.input(
          'float32',
          Array.from(getGPURasterReclassifyParameterValues({breakCount: 2}))
        ),
        output:
          port === 'classes'
            ? {classes: harness.output('output', 'uint32', 4)}
            : port === 'reclassified'
              ? {reclassified: harness.output('output', 'float32', 4)}
              : {classCounts: harness.output('output', 'uint32', 3)}
      })
    );
    // Bins are (-infinity,1), [1,3), [3,infinity): 4,1,4,3 map to 2,1,2,2.
    // No value occupies bin zero; one occupies bin one and three occupy bin two.
    expectCensusValues(
      result['output'],
      port === 'classes' ? [2, 1, 2, 2] : port === 'reclassified' ? [30, 20, 30, 30] : [0, 1, 3]
    );
  });
}

it('census raster join writes point zones with counts absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const {grid, points, zones} = createCensusRasterJoinFixture();
  const result = await harness.run(
    new GPURasterJoin({
      grid,
      parameters: harness.input(
        'float32',
        Array.from(getGPUPolygonRasterizationParameterValues(grid))
      ),
      points: harness.input('float32x2', points),
      zones: harness.input('uint32', zones),
      zoneCount: 2,
      output: {pointZones: harness.output('zones', 'uint32', 4)}
    })
  );
  // Top-left origin (0,1) and southward unit rows cover [0,2) x [0,1).
  // Raster zones [1,0] and queries right, left, right, outside distinguish point rows,
  // cell indices and zone IDs. The outside point must receive the unsigned NONE sentinel.
  expectCensusValues(result['zones'], [0, 1, 0, 0xffffffff]);
});

it('census embedding change writes similarity with angle absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const result = await harness.run(
    new GPURasterEmbeddingChange({
      pixelCount: 1,
      dimensions: 2,
      before: harness.input('float32', [1, 0]),
      after: harness.input('float32', [0, 1]),
      output: {similarity: harness.output('similarity', 'float32', 1)}
    })
  );
  expectCensusValues(result['similarity'], [0]); // Orthogonal unit vectors.
});

it('census linear probe writes confidence with probabilities absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const result = await harness.run(
    new GPURasterEmbeddingLinearProbe({
      pixelCount: 1,
      dimensions: 1,
      classCount: 2,
      vectors: harness.input('float32', [2]),
      weights: harness.input('float32', [1, -1]),
      bias: harness.input('float32', [0, 0]),
      output: {
        labels: harness.output('labels', 'uint32', 1),
        confidence: harness.output('confidence', 'float32', 1)
      }
    })
  );
  // Logits [2,-2] have winner 0 and softmax probability 1/(1+exp(-4)).
  expectCensusValues(result['labels'], [0]);
  expectCensusValues(result['confidence'], [1 / (1 + Math.exp(-4))], 2e-6);
});

it('census embedding projection writes RGB without projected scores', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const result = await harness.run(
    new GPURasterEmbeddingProjection({
      pixelCount: 1,
      dimensions: 3,
      componentCount: 3,
      vectors: harness.input('float32', [1, 2, 3]),
      mean: harness.input('float32', [0, 0, 0]),
      components: harness.input('float32', [1, 0, 0, 0, 1, 0, 0, 0, 1]),
      parameters: harness.input('float32', [0, 4, 0, 4, 0, 4]),
      output: {rgb: harness.output('rgb', 'float32', 3)}
    })
  );
  // Identity projection and [0,4] channel ranges divide [1,2,3] by four, exactly.
  expectCensusValues(result['rgb'], [0.25, 0.5, 0.75]);
});

it('census embedding pyramid writes vectors without validity or support outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const result = await harness.run(
    new GPURasterEmbeddingPyramidLevel({
      sourceWidth: 2,
      sourceHeight: 1,
      dimensions: 2,
      scale: 2,
      vectors: harness.input('float32', [1, 0, 0, 1]),
      output: {vectors: harness.output('vectors', 'float32', 2)}
    })
  );
  // Normalize the arithmetic mean (.5,.5), with float32 square-root/division tolerance.
  expectCensusValues(result['vectors'], [Math.SQRT1_2, Math.SQRT1_2], 2e-6);
});

it('census embedding zone mean writes means without weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const result = await harness.run(
    new GPURasterEmbeddingZoneMean({
      pixelCount: 2,
      dimensions: 2,
      zoneCount: 1,
      vectors: harness.input('float32', [1, 0, 0, 1]),
      zoneRaster: harness.input('uint32', [0, 0]),
      output: {means: harness.output('means', 'float32', 2)}
    })
  );
  expectCensusValues(result['means'], [Math.SQRT1_2, Math.SQRT1_2], 2e-6);
});

it('census scalar ridges publishes the quadratic ridge with strength absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const grid = makeGPURasterMetadata({width: 3, height: 3});
  const result = await harness.run(
    new GPUScalarFieldRidges({
      grid,
      input: {
        id: 'quadratic',
        format: 'float32',
        storage: {kind: 'buffer', values: harness.input('float32', [1, 2, 1, 1, 2, 1, 1, 2, 1])}
      },
      parameters: harness.input(
        'float32',
        Array.from(getGPUScalarFieldRidgesParameterValues({}, grid))
      ),
      output: {
        ridges: harness.output('ridges', 'uint32', 9),
        validity: harness.output('validity', 'uint32', 9)
      }
    })
  );
  // f(x,y)=2-(x-1)^2: the only complete 3x3 stencil is the center, whose normal
  // gradient is zero and smaller Hessian eigenvalue is -2; borders have no full stencil.
  expectCensusValues(result['ridges'], [0, 0, 0, 0, 1, 0, 0, 0, 0]);
  expectCensusValues(result['validity'], [0, 0, 0, 0, 1, 0, 0, 0, 0]);
});

for (const Operation of [GPURasterWalkingCostOperation, CPURasterWalkingCostOperation]) {
  it(`census ${Operation.name} runs with only required convergence status`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const grid = makeGPURasterMetadata({width: 1, height: 1});
    const result = await runOperationGraph(device, {
      inputs: {
        elevation: {format: 'float32', values: Float32Array.of(0)},
        friction: {format: 'float32', values: Float32Array.of(1)},
        sources: {format: 'uint32', values: Uint32Array.of(1)},
        parameters: {format: 'float32', values: getGPURasterWalkingCostParameterValues({}, grid)}
      },
      operations: [
        new Operation({
          grid,
          elevation: {
            id: 'elevation',
            format: 'float32',
            storage: {kind: 'buffer', values: 'elevation'}
          },
          friction: {
            id: 'friction',
            format: 'float32',
            storage: {kind: 'buffer', values: 'friction'}
          },
          sources: 'sources',
          parameters: 'parameters',
          output: {costs: 'costs', validity: 'validity', status: {converged: 'converged'}}
        })
      ],
      outputs: ['costs', 'validity', 'converged']
    });
    // A sole valid source has zero cost and no outgoing edges; it is already a fixed point.
    expectCensusValues(result['costs'], [0]);
    expectCensusValues(result['validity'], [1]);
    expectCensusValues(result['converged'], [1]);
  });
}
