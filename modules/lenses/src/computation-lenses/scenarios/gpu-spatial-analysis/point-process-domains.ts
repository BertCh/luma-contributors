// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationKeySpace} from '@luma-contributors/analysis/computation-record';

const DOMAINS: Record<
  string,
  {
    spaces: readonly (readonly [string, number])[];
    captures: Readonly<Record<string, string>>;
  }
> = {
  GPUInhomogeneousPointPattern: {
    spaces: [
      ['event', 3],
      ['radius-bin', 2],
      ['window-ring-offset', 2],
      ['window-polygon-offset', 2],
      ['window-feature-offset', 2],
      ['status-word', 2]
    ],
    captures: {
      positions: 'event',
      suppliedIntensity: 'event',
      intensity: 'event',
      radii: 'radius-bin',
      k: 'radius-bin',
      pairCorrelation: 'radius-bin',
      ringOffsets: 'window-ring-offset',
      polygonOffsets: 'window-polygon-offset',
      featureOffsets: 'window-feature-offset',
      status: 'status-word'
    }
  },
  GPUPointPatternEnvelope: {
    spaces: [
      ['curve-bin', 2],
      ['simulation', 3],
      ['curve-statistic', 4],
      ['status-word', 1]
    ],
    captures: {
      observed: 'curve-bin',
      reference: 'curve-bin',
      integrationWeights: 'curve-bin',
      lower: 'curve-bin',
      upper: 'curve-bin',
      simulations: 'simulation',
      statistics: 'curve-statistic',
      status: 'status-word'
    }
  },
  GPUSpatialBalancedSample: {
    spaces: [
      ['frame-row', 4],
      ['sample-row', 2],
      ['status-word', 2]
    ],
    captures: {
      positions: 'frame-row',
      frameIds: 'frame-row',
      selectedFrameIds: 'sample-row',
      inclusionProbabilities: 'sample-row',
      designWeights: 'sample-row',
      status: 'status-word'
    }
  },
  GPUOPTICS: {
    spaces: [
      ['event', 4],
      ['ordering-rank', 4],
      ['status-word', 1]
    ],
    captures: {
      positions: 'event',
      ordering: 'ordering-rank',
      coreDistances: 'event',
      reachability: 'event',
      predecessors: 'event',
      labels: 'event',
      iterationCount: 'status-word',
      converged: 'status-word',
      invalidCount: 'status-word'
    }
  }
};

/** Row domains for the fixed illustrative point-process scenarios. */
export function getPointProcessKeySpaces(element: string): GPUComputationKeySpace[] {
  return DOMAINS[element].spaces.map(([id, size]) => ({id, label: id, size, indexSpace: '1d'}));
}

/** Presentation metadata distinguishes entity rows, curve bins, topology boundaries and status words. */
export function getPointProcessCaptureMetadata(element: string, name: string) {
  const keySpace = DOMAINS[element].captures[name];
  if (!keySpace) throw new Error(`${element} capture ${name} requires an explicit row domain`);
  return {keySpace, ...(name === 'simulations' && {components: 2})};
}
