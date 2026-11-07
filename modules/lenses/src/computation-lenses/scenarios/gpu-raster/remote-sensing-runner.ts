// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';

import type {Device} from '@luma.gl/core';
import '../../../gpu-raster/computation-pack/index';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {runRasterLens, type RasterLensSession} from './raster-lens-harness';
import {runExample as runRemapping} from './remote-sensing-conservative-remapping.example';
import {runExample as runTexture} from './remote-sensing-texture.example';
import {runExample as runTemporal} from './remote-sensing-temporal.example';
import {runExample as runRegistration} from './remote-sensing-registration.example';
import {runExample as runRegionGraph} from './remote-sensing-region-graph.example';
import {runExample as runUnmixing} from './remote-sensing-spectral-unmixing.example';
import {runExample as runSeasonalBreak} from './remote-sensing-seasonal-break.example';

export const REMOTE_SENSING_GRID: GPURasterMetadata = {
  width: 4,
  height: 4,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};

/** Executes public contributors and captures actual GPU outputs and graph diagnostics. */
export function runRemoteSensingLens(device: Device, element: string, scenarioId: string) {
  return runRasterLens({
    device,
    scenarioId,
    domain: 'gpu-raster',
    label: element,
    grid: REMOTE_SENSING_GRID,
    coordinateModel: 'projected',
    unit: 'm',
    dataset: {id: 'remote-sensing-synthetic', version: '1'},
    keySpaces: [
      {id: 'regions', label: 'original source regions', size: 4},
      {id: 'tileCells', label: 'cells in each explicit region tile core', size: 8},
      {id: 'regionPairs', label: 'source-region boundary matrix entries', size: 16},
      {id: 'alignment', label: 'whole-image alignment', size: 1},
      {id: 'endmembers', label: 'supplied endmember spectra', size: 2},
      {id: 'observations', label: 'cell-major regular-time observations', size: 16 * 36}
    ],
    build: session => buildRemoteSensingScenario(session, element)
  });
}

/** Constructs the same executable examples used in the viewer, with semantic capture shapes. */
export async function buildRemoteSensingScenario(
  session: RasterLensSession,
  element: string
): Promise<void> {
  const count = 16;
  const rasterInput = (name: string, values: number[], meaning: string) =>
    session.input(name, 'float32', values, {role: 'input', geometry: 'raster', meaning});
  const rasterFloat = (name: string, meaning: string) =>
    session.output(name, 'float32', {role: 'output', geometry: 'raster', meaning}, count);
  const rasterStatus = (name: string, meaning: string) =>
    session.output(name, 'uint32', {role: 'context', geometry: 'raster', meaning}, count);
  if (element === 'GPURasterConservativeRemapping') {
    await runRemapping(session.graph, {
      source: {width: 4, height: 4, origin: [0, 0], spacing: [1, 1]},
      destination: {width: 4, height: 4, origin: [0.5, 0.5], spacing: [1, 1]},
      values: rasterInput(
        'sourceValues',
        Array.from({length: count}, (_, index) => index),
        'Intensive source observations on the unshifted grid'
      ),
      normalization: 'covered-area',
      output: {
        values: rasterFloat(
          'remappedValues',
          'Valid-overlap intensive average on the half-cell-shifted destination'
        ),
        coveredArea: rasterFloat(
          'coveredArea',
          'Valid source overlap area for each destination cell'
        ),
        status: rasterStatus('status', 'Zero identifies a defined conservative average')
      }
    });
    return;
  }
  if (element === 'GPURasterTexture') {
    await runTexture(session.graph, {
      grid: makeGPURasterMetadata({width: 4, height: 4}),
      values: rasterInput(
        'checkerboard',
        Array.from({length: count}, (_, index) => ((index % 4) + Math.floor(index / 4)) % 2),
        'Binary checkerboard observations before quantization'
      ),
      levels: 2,
      range: [0, 1],
      radius: 1,
      offset: [1, 0],
      symmetric: true,
      features: session.output(
        'features',
        'float32',
        {
          role: 'output',
          keySpace: 'cells',
          components: 7,
          meaning: 'Contrast, dissimilarity, homogeneity, ASM, energy, correlation and entropy'
        },
        count * 7
      ),
      pairCount: session.output(
        'pairCount',
        'uint32',
        {
          role: 'output',
          geometry: 'raster',
          meaning: 'Directed paired-valid observations, including symmetric reverse pairs'
        },
        count
      )
    });
    return;
  }
  if (element === 'GPURasterTemporal') {
    const slices = 9;
    const length = count * slices;
    await runTemporal(session.graph, {
      cellCount: count,
      sliceCount: slices,
      values: session.input(
        'observations',
        'float32',
        Array.from({length}, (_, index) => {
          const time = index % slices;
          return Math.floor(index / slices) + 2 + 3 * time + time * time;
        }),
        {
          role: 'input',
          keySpace: 'cells',
          components: slices,
          meaning: 'Cell-major quadratic observations at nine regular acquisition dates'
        }
      ),
      validity: session.input(
        'observationQA',
        'uint32',
        Array.from({length}, (_, index) => (index % slices === 2 ? 0 : 1)),
        {
          role: 'context',
          keySpace: 'cells',
          components: slices,
          meaning: 'Acquisition two is rejected independently for every cell'
        }
      ),
      operation: 'savitzky-golay',
      windowLength: 7,
      polynomialOrder: 2,
      edgeMode: 'interp',
      output: {
        values: session.output(
          'smoothedValues',
          'float32',
          {
            role: 'output',
            keySpace: 'cells',
            components: slices,
            meaning: 'QR polynomial predictions at all original acquisition dates'
          },
          length
        ),
        counts: session.output(
          'supportCount',
          'uint32',
          {
            role: 'context',
            keySpace: 'cells',
            components: slices,
            meaning: 'QA-approved support retained in each local polynomial fit'
          },
          length
        ),
        chosenSlice: session.output(
          'chosenSlice',
          'uint32',
          {
            role: 'context',
            keySpace: 'cells',
            components: slices,
            meaning: 'No single source date exists for a fitted polynomial: 0xffffffff'
          },
          length
        ),
        status: session.output(
          'status',
          'uint32',
          {
            role: 'context',
            keySpace: 'cells',
            components: slices,
            meaning: 'Zero records a full-rank local polynomial fit'
          },
          length
        )
      }
    });
    return;
  }
  if (element === 'GPURasterRegistration') {
    const reference = [0, 1, 0, 2, 3, 0, 4, 0, 0, 5, 1, 0, 2, 0, 0, 3];
    const shiftScalars = session.output(
      'shift',
      'float32',
      {
        role: 'output',
        keySpace: 'alignment',
        components: 2,
        meaning: 'Column/row translation to apply to moving: expected (-1, 0)'
      },
      2
    );
    await runRegistration(session.graph, {
      mode: 'masked-linear',
      grid: makeGPURasterMetadata({width: 4, height: 4}),
      reference: rasterInput(
        'referenceImage',
        reference,
        'Finite reference image with a unique nonperiodic pattern'
      ),
      moving: rasterInput(
        'movingImage',
        reference.map((_, index) => (index % 4 === 0 ? 0 : reference[index - 1])),
        'Reference image shifted one column right with an invalid first column'
      ),
      movingValidity: session.input(
        'movingQA',
        'uint32',
        reference.map((_, index) => (index % 4 === 0 ? 0 : 1)),
        {
          role: 'context',
          geometry: 'raster',
          meaning: 'The newly exposed first column has no observation'
        }
      ),
      maximumShift: [1, 1],
      upsampleFactor: 1,
      output: {
        shift: session.graph.createDataView(shiftScalars.buffer, {
          format: 'float32x2',
          byteOffset: shiftScalars.byteOffset,
          length: 1
        }),
        peak: session.output(
          'correlationPeak',
          'float32',
          {
            role: 'output',
            keySpace: 'alignment',
            meaning: 'Normalized correlation on the accepted paired overlap'
          },
          1
        ),
        status: session.output(
          'status',
          'uint32',
          {
            role: 'context',
            keySpace: 'alignment',
            meaning: 'Zero accepts the bounded masked alignment'
          },
          1
        ),
        overlap: session.output(
          'overlapCount',
          'uint32',
          {
            role: 'output',
            keySpace: 'alignment',
            meaning: 'Accepted paired-valid pixels in the linear overlap'
          },
          1
        )
      }
    });
    return;
  }
  if (element === 'GPURasterRegionGraph') {
    await runRegionGraph(session.graph, {
      grid: makeGPURasterMetadata({width: 4, height: 4}),
      tiles: [0, 2].map((row, tile) => ({
        grid: makeGPURasterMetadata({width: 4, height: 2}),
        origin: [0, row] as const,
        labels: session.input(
          `tile${tile}Labels`,
          'uint32',
          Array.from({length: 8}, (_, index) => (tile === 0 ? index % 4 : 3 - (index % 4))),
          {
            role: 'input',
            keySpace: 'tileCells',
            meaning: 'Tile-local region labels; the lower tile reverses its local numbering'
          }
        ),
        values: session.input(
          `tile${tile}Values`,
          'float32',
          Array.from({length: 8}, (_, index) => [0, 1, 2, 6][index % 4]),
          {
            role: 'input',
            keySpace: 'tileCells',
            meaning: 'Finite tile observations for shared source-region means'
          }
        ),
        sourceRegionIds: session.input(
          `tile${tile}SourceIds`,
          'uint32',
          tile === 0 ? [0, 1, 2, 3] : [3, 2, 1, 0],
          {
            role: 'context',
            keySpace: 'regions',
            meaning: 'Explicit local-label to stable source-region equivalence map'
          }
        )
      })),
      regionCount: 4,
      mergeMode: 'adaptive',
      mergeThreshold: 1.1,
      maximumIterations: 3,
      output: {
        adjacency: session.output(
          'adjacency',
          'uint32',
          {
            role: 'context',
            keySpace: 'regionPairs',
            meaning: 'Symmetric original-region shared-boundary counts'
          },
          16
        ),
        weights: session.output(
          'updatedWeights',
          'float32',
          {
            role: 'output',
            keySpace: 'regionPairs',
            meaning: 'Final component-mean distances at original region boundaries'
          },
          16
        ),
        mapping: session.output(
          'sourceToRoot',
          'uint32',
          {
            role: 'output',
            keySpace: 'regions',
            meaning: 'Original region to surviving source root after adaptive merges'
          },
          4
        ),
        status: session.output(
          'status',
          'uint32',
          {
            role: 'context',
            keySpace: 'alignment',
            meaning: 'One means another eligible adaptive merge remains'
          },
          1
        )
      }
    });
    return;
  }
  if (element === 'GPURasterSpectralUnmixing') {
    await runUnmixing(session.graph, {
      cellCount: count,
      bandCount: 3,
      endmemberCount: 2,
      values: session.input(
        'spectra',
        'float32',
        Array.from({length: count * 3}, (_, index) => {
          const abundance = Math.floor(index / 3) / 15;
          return index % 3 === 0 ? abundance : index % 3 === 1 ? 1 - abundance : 1;
        }),
        {
          role: 'input',
          keySpace: 'cells',
          components: 3,
          meaning: 'Three-band exact mixtures with a spatially varying first abundance'
        }
      ),
      endmembers: session.input('endmembers', 'float32', [1, 0, 1, 0, 1, 1], {
        role: 'context',
        keySpace: 'endmembers',
        components: 3,
        meaning: 'Two supplied three-band endmember spectra'
      }),
      constraint: 'simplex',
      maximumIterations: 256,
      tolerance: 1e-6,
      output: {
        abundances: session.output(
          'abundances',
          'float32',
          {
            role: 'output',
            keySpace: 'cells',
            components: 2,
            meaning: 'Nonnegative first and second abundances, constrained to sum one'
          },
          count * 2
        ),
        residual: rasterFloat(
          'squaredResidual',
          'Squared spectral reconstruction error over the three valid bands'
        ),
        status: rasterStatus('status', 'Zero records convergence of the constrained spectral fit'),
        iterationCounts: rasterStatus(
          'iterations',
          'Projected-gradient iterations needed for the stopping criterion'
        )
      }
    });
    return;
  }
  if (element === 'GPURasterSeasonalBreak') {
    const slices = 36;
    const length = count * slices;
    await runSeasonalBreak(session.graph, {
      cellCount: count,
      sliceCount: slices,
      values: session.input(
        'seasonalObservations',
        'float32',
        Array.from({length}, (_, index) => {
          const time = index % slices;
          const cell = Math.floor(index / slices);
          return (
            5 +
            cell * 0.1 +
            time * 0.2 +
            [0, 1, -1, 0.5][time % 4] +
            (cell >= 8 && time >= 28 ? 4 : 0)
          );
        }),
        {
          role: 'input',
          keySpace: 'cells',
          components: slices,
          meaning:
            'Dummy seasons plus trend; lower half receives a monitoring shift at acquisition twenty-eight'
        }
      ),
      period: 4,
      seasonality: 'dummy',
      monitoring: {historyLength: 24, threshold: 4, minimumConsecutive: 2},
      minimumSegmentLength: 8,
      output: {
        coefficients: session.output(
          'coefficients',
          'float32',
          {
            role: 'output',
            keySpace: 'cells',
            components: 6,
            meaning: 'Intercept, normalized slope, three dummy seasons and historical jump'
          },
          count * 6
        ),
        components: session.output(
          'components',
          'float32',
          {
            role: 'output',
            keySpace: 'observations',
            components: 3,
            meaning: 'Trend including jump, seasonality and residual at each acquisition'
          },
          length * 3
        ),
        breakIndex: session.output(
          'breakIndex',
          'uint32',
          {
            role: 'output',
            geometry: 'raster',
            meaning: 'Selected original break date; 0xffffffff means no break'
          },
          count
        ),
        improvement: rasterFloat(
          'bicImprovement',
          'Selected BIC gain over the no-break harmonic trend model'
        ),
        status: rasterStatus('status', 'Zero records an identifiable historical model'),
        diagnostics: session.output(
          'diagnostics',
          'float32',
          {
            role: 'output',
            keySpace: 'cells',
            components: 6,
            meaning:
              'Iterations, convergence, training support, residual scale, first monitoring alarm and maximum score'
          },
          count * 6
        )
      }
    });
    return;
  }
  throw new Error(`Unknown remote-sensing scenario: ${element}`);
}
