// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/raster-analysis';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  GPUGeomorphons,
  type GPUGeomorphonsOutput
} from '../../../src/gpu-terrain/geomorphons/gpu-geomorphons';
import {
  GPUTerrainFlow,
  type GPUTerrainFlowOutput
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {
  GPUTerrainHeightAboveDrainage,
  type GPUTerrainHeightAboveDrainageOutput
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage';
import {
  GPUTerrainHydrologicIndices,
  type GPUTerrainHydrologicIndicesOutput
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-hydrologic-indices';
import {
  GPUSimpleLocalRelief,
  type GPUSimpleLocalReliefOutput
} from '../../../src/gpu-terrain/relief-visualization/gpu-simple-local-relief';
import {
  GPUMultiScaleRelief,
  type GPUMultiScaleReliefOutput
} from '../../../src/gpu-terrain/relief-visualization/gpu-multi-scale-relief';
import {
  GPULocalDominance,
  type GPULocalDominanceOutput
} from '../../../src/gpu-terrain/relief-visualization/gpu-local-dominance';
import {
  GPUReliefBlend,
  type GPUReliefBlendOutput
} from '../../../src/gpu-terrain/relief-visualization/gpu-relief-blend';
import {
  GPUTerrainCurvature,
  type GPUTerrainCurvatureOutput
} from '../../../src/gpu-terrain/terrain-curvature/gpu-terrain-curvature';
import {
  GPUTerrainSummits,
  type GPUTerrainSummitsOutput
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-summits';
import {
  GPUProfilePeaks,
  type GPUProfilePeaksOutput
} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks';
import {
  GPUTerrainRuggedness,
  type GPUTerrainRuggednessOutput
} from '../../../src/gpu-terrain/topographic-position/gpu-terrain-ruggedness';
import {
  GPUTerrainVectorRuggedness,
  type GPUTerrainVectorRuggednessOutput
} from '../../../src/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness';
import {
  GPUTerrainTopographicPosition,
  type GPUTerrainTopographicPositionOutput
} from '../../../src/gpu-terrain/topographic-position/gpu-terrain-topographic-position';
import {
  GPUTerrainWeissLandforms,
  type GPUTerrainWeissLandformsOutput
} from '../../../src/gpu-terrain/topographic-position/gpu-terrain-weiss-landforms';

it('DEF-09 accepts each minimal terminal and rejects empty JavaScript selections for all fifteen classes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width: 3, height: 3});
  const floating = createTransientView(graph, 'elevation', 'float32', 9);
  const unsigned = createTransientView(graph, 'directions', 'uint32', 9);
  const offsets = createTransientView(graph, 'offsets', 'uint32', 2);
  const parameters = createTransientView(graph, 'parameters', 'float32', 32);
  const base = {grid, elevation: makeGPURasterBand(floating, grid), parameters};
  try {
    const output0: GPUGeomorphonsOutput = {
      forms: createTransientView(graph, 'output-0', 'uint32', 9)
    };
    const create0 = (output: GPUGeomorphonsOutput) =>
      new GPUGeomorphons({...base, searchRadius: 2, output});
    expect(() => create0(output0)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty0: GPUGeomorphonsOutput = {};
    expect(() => create0(empty0)).toThrow(/at least one output/);
    const output1: GPUTerrainFlowOutput = {
      flowDirections: createTransientView(graph, 'output-1', 'uint32', 9)
    };
    const create1 = (output: GPUTerrainFlowOutput) => new GPUTerrainFlow({...base, output});
    expect(() => create1(output1)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty1: GPUTerrainFlowOutput = {};
    expect(() => create1(empty1)).toThrow(/at least one output/);
    const output2: GPUTerrainHeightAboveDrainageOutput = {
      heightAboveDrainage: createTransientView(graph, 'output-2', 'float32', 9)
    };
    const create2 = (output: GPUTerrainHeightAboveDrainageOutput) =>
      new GPUTerrainHeightAboveDrainage({
        ...base,
        flowDirections: unsigned,
        streams: unsigned,
        output
      });
    expect(() => create2(output2)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty2: GPUTerrainHeightAboveDrainageOutput = {};
    expect(() => create2(empty2)).toThrow(/at least one output/);
    const output3: GPUTerrainHydrologicIndicesOutput = {
      descentTangent: createTransientView(graph, 'output-3', 'float32', 9)
    };
    const create3 = (output: GPUTerrainHydrologicIndicesOutput) =>
      new GPUTerrainHydrologicIndices({...base, accumulation: floating, output});
    expect(() => create3(output3)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty3: GPUTerrainHydrologicIndicesOutput = {};
    expect(() => create3(empty3)).toThrow(/at least one output/);
    const output4: GPUSimpleLocalReliefOutput = {
      relief: createTransientView(graph, 'output-4', 'float32', 9)
    };
    const create4 = (output: GPUSimpleLocalReliefOutput) =>
      new GPUSimpleLocalRelief({...base, maximumRadius: 1, output});
    expect(() => create4(output4)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty4: GPUSimpleLocalReliefOutput = {};
    expect(() => create4(empty4)).toThrow(/at least one output/);
    const output5: GPUMultiScaleReliefOutput = {
      relief: createTransientView(graph, 'output-5', 'float32', 9)
    };
    const create5 = (output: GPUMultiScaleReliefOutput) =>
      new GPUMultiScaleRelief({
        ...base,
        resolution: 1,
        featureMinimum: 1,
        featureMaximum: 5,
        scalingFactor: 1,
        output
      });
    expect(() => create5(output5)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty5: GPUMultiScaleReliefOutput = {};
    expect(() => create5(empty5)).toThrow(/at least one output/);
    const output6: GPULocalDominanceOutput = {
      dominance: createTransientView(graph, 'output-6', 'float32', 9)
    };
    const create6 = (output: GPULocalDominanceOutput) =>
      new GPULocalDominance({...base, minimumRadius: 1, maximumRadius: 1, output});
    expect(() => create6(output6)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty6: GPULocalDominanceOutput = {};
    expect(() => create6(empty6)).toThrow(/at least one output/);
    const output7: GPUReliefBlendOutput = {
      blend: createTransientView(graph, 'output-7', 'float32', 9)
    };
    const create7 = (output: GPUReliefBlendOutput) =>
      new GPUReliefBlend({...base, layers: [floating], output});
    expect(() => create7(output7)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty7: GPUReliefBlendOutput = {};
    expect(() => create7(empty7)).toThrow(/at least one output/);
    const output8: GPUTerrainCurvatureOutput = {
      ringCurvature: createTransientView(graph, 'output-8', 'float32', 9)
    };
    const create8 = (output: GPUTerrainCurvatureOutput) =>
      new GPUTerrainCurvature({...base, output});
    expect(() => create8(output8)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty8: GPUTerrainCurvatureOutput = {};
    expect(() => create8(empty8)).toThrow(/at least one output/);
    const output9: GPUTerrainSummitsOutput = {
      summitMask: createTransientView(graph, 'output-9', 'uint32', 9)
    };
    const create9 = (output: GPUTerrainSummitsOutput) => new GPUTerrainSummits({...base, output});
    expect(() => create9(output9)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty9: GPUTerrainSummitsOutput = {};
    expect(() => create9(empty9)).toThrow(/at least one output/);
    const output10: GPUProfilePeaksOutput = {
      peakMask: createTransientView(graph, 'output-10', 'uint32', 9)
    };
    const create10 = (output: GPUProfilePeaksOutput) =>
      new GPUProfilePeaks({...base, values: floating, offsets, window: 1, output});
    expect(() => create10(output10)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty10: GPUProfilePeaksOutput = {};
    expect(() => create10(empty10)).toThrow(/at least one output/);
    const output11: GPUTerrainRuggednessOutput = {
      roughness: createTransientView(graph, 'output-11', 'float32', 9)
    };
    const create11 = (output: GPUTerrainRuggednessOutput) =>
      new GPUTerrainRuggedness({...base, output});
    expect(() => create11(output11)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty11: GPUTerrainRuggednessOutput = {};
    expect(() => create11(empty11)).toThrow(/at least one output/);
    const output12: GPUTerrainVectorRuggednessOutput = {
      vectorRuggedness: createTransientView(graph, 'output-12', 'float32', 9)
    };
    const create12 = (output: GPUTerrainVectorRuggednessOutput) =>
      new GPUTerrainVectorRuggedness({...base, output});
    expect(() => create12(output12)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty12: GPUTerrainVectorRuggednessOutput = {};
    expect(() => create12(empty12)).toThrow(/at least one output/);
    const output13: GPUTerrainTopographicPositionOutput = {
      topographicPositionIndex: createTransientView(graph, 'output-13', 'float32', 9)
    };
    const create13 = (output: GPUTerrainTopographicPositionOutput) =>
      new GPUTerrainTopographicPosition({...base, scales: [{radius: 1}], output});
    expect(() => create13(output13)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty13: GPUTerrainTopographicPositionOutput = {};
    expect(() => create13(empty13)).toThrow(/at least one output/);
    const output14: GPUTerrainWeissLandformsOutput = {
      landforms: createTransientView(graph, 'output-14', 'uint32', 9)
    };
    const create14 = (output: GPUTerrainWeissLandformsOutput) =>
      new GPUTerrainWeissLandforms({...base, output});
    expect(() => create14(output14)).not.toThrow();
    // @ts-expect-error DEF-09 requires an accepted terminal output.
    const empty14: GPUTerrainWeissLandformsOutput = {};
    expect(() => create14(empty14)).toThrow(/at least one output/);
    // @ts-expect-error DEF-09 flow needs a data output even when a status leaf is present.
    const statusOnlyFlow: GPUTerrainFlowOutput = {status: {flatsConverged: unsigned}};
    void statusOnlyFlow;
    // @ts-expect-error DEF-09 empty nested curvatures do not request a terminal.
    const emptyCurvatures: GPUTerrainCurvatureOutput = {curvatures: {}};
    expect(() => new GPUTerrainCurvature({...base, output: emptyCurvatures})).toThrow(
      /at least one output/
    );
    // @ts-expect-error DEF-09 an empty nested profile status does not request a terminal.
    const emptyProfileStatus: GPUProfilePeaksOutput = {status: {}};
    void emptyProfileStatus;
    const curvature: GPUTerrainCurvatureOutput = {
      curvatures: {mean: createTransientView(graph, 'mean', 'float32', 9)}
    };
    expect(() => new GPUTerrainCurvature({...base, output: curvature})).not.toThrow();
    const profileStatus: GPUProfilePeaksOutput = {
      status: {converged: createTransientView(graph, 'converged', 'uint32', 1)}
    };
    expect(
      () =>
        new GPUProfilePeaks({...base, values: floating, offsets, window: 1, output: profileStatus})
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});
