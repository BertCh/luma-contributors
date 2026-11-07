// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {runExample as runGPULOSHResidualPermutationTest} from './losh-residual-permutation-test.example';
import {runExample as runGPULocalSpatialHeteroscedasticity} from './local-spatial-heteroscedasticity.example';
import {runExample as runGPUMultivariateLocalGeary} from './multivariate-local-geary.example';
import {runExample as runGPUGeneralizedGeographicallyWeightedRegression} from './generalized-geographically-weighted-regression.example';
import {runExample as runGPUMultiscaleGeographicallyWeightedRegression} from './multiscale-geographically-weighted-regression.example';
import {runExample as runGPUSpatialPanelRegression} from './spatial-panel-regression.example';
import {runExample as runGPUSpatialRegimeRegression} from './spatial-regime-regression.example';
import {runExample as runGPUConcaveHull} from './concave-hull.example';
import {runExample as runGPUGroupConcaveHull} from './group-concave-hull.example';
import {runExample as runGPUConstrainedDelaunayTriangulation} from './constrained-delaunay-triangulation.example';
import {runExample as runGPUH3CellIdentities} from './h3-cell-identities.example';
import {runExample as runGPUBoundedFacilityAllocation} from './bounded-facility-allocation.example';
import {runExample as runAddZonalDistributionRecipe} from './zonal-distribution-recipe.example';
import {runExample as runAddRasterZonalDistributionRecipe} from './raster-zonal-distribution-recipe.example';
import {runExample as runAddExposureByZoneRecipe} from './exposure-by-zone-recipe.example';
import {runExample as runAddRegionalRasterChangeRecipe} from './regional-raster-change-recipe.example';
import {runExample as runAddSurfacePreparationRecipe} from './surface-preparation-recipe.example';
import {runExample as runAddTrackQualityRecipe} from '../gpu-network/track-quality-recipe.example';
import {runExample as runAddInterpolationDiagnosticsRecipe} from './interpolation-diagnostics-recipe.example';
import {runExample as runAddDrainageNetworkRecipe} from './drainage-network-recipe.example';
import {runExample as runAddLocalVariabilityRecipe} from './local-variability-recipe.example';
import {runExample as runAddCellFlowTopologyRecipe} from './cell-flow-topology-recipe.example';
import {type GPUCommandGraph, type GraphDataView} from '@luma-contributors/graph/gpu-core';
import type {GPUVectorFormat} from '@luma-contributors/graph/gpu-data';
import type {GPURasterMetadata} from '@luma-contributors/graph/gpu-raster';
import {getGPUChangeDetectionParameterValues} from '@luma-contributors/graph/gpu-raster';
import {getGPUTerrainFlowParameterValues} from '@luma-contributors/graph/gpu-terrain';
import {
  getGPUKrigingParameterValues,
  getGPUPermutationParameterValues,
  getGPUSpatialAutocorrelationParameterValues
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {createWGSLKernelNode} from '@luma-contributors/graph';
import {type S2Uploads, type S2CaptureSpec} from './s2-lens-run';
const reference = {
  zones: [0, 0, 0, 1, 1, 1, 2, 2],
  values: [1, 3, 3, 4, null, 8, null, null],
  probabilities: [0, 0.25, 0.5, 0.75, 1]
};

/** Explicit illustrative inputs, in arbitrary planar units; never represented as observed data. */
export async function buildExpansionScenario(
  element: string,
  graph: GPUCommandGraph<undefined>,
  uploads: S2Uploads
): Promise<Record<string, S2CaptureSpec>> {
  let serial = 0;
  const captures: Record<string, S2CaptureSpec> = {};
  const input = <Format extends GPUVectorFormat>(format: Format, values: number[]) =>
    uploads.view(
      `input-${serial++}`,
      format.startsWith('uint') ? Uint32Array.from(values) : Float32Array.from(values),
      format
    );
  const output = <Format extends GPUVectorFormat>(name: string, format: Format, length: number) => {
    const components = Number(format.match(/x([234])$/)?.[1] ?? 1);
    const view = uploads.view(
      `output-${serial++}`,
      format.startsWith('uint')
        ? new Uint32Array(length * components)
        : new Float32Array(length * components),
      format,
      length
    );
    captures[name] = {view, role: 'output'};
    return {view};
  };
  const captureOutputs = (value: unknown, prefix = 'result'): Record<string, S2CaptureSpec> => {
    if (
      value &&
      typeof value === 'object' &&
      'buffer' in value &&
      'format' in value &&
      'length' in value
    ) {
      const source = value as GraphDataView;
      const components = Number(source.format.match(/x([234])$/)?.[1] ?? 1);
      const destination = output(prefix, source.format, source.length).view;
      const type = source.format.startsWith('uint') ? 'u32' : 'f32';
      graph.add(
        createWGSLKernelNode(graph, {
          id: `capture-${serial++}`,
          operation: 'lens-output-copy',
          invocationCount: source.length * components,
          bindings: [
            {name: 'source', view: source, type, access: 'read'},
            {name: 'destination', view: destination, type, access: 'read_write'}
          ],
          body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
        })
      );
      captures[prefix] = {view: destination, role: 'output'};
    } else if (value && typeof value === 'object')
      for (const [name, child] of Object.entries(value)) captureOutputs(child, `${prefix}-${name}`);
    return captures;
  };
  const fixture = {
    graph,
    input: <Format extends GPUVectorFormat>(
      name: string,
      data: Float32Array | Uint32Array | Int32Array,
      format: Format,
      length: number
    ) => uploads.view(name, data, format, length),
    parameters: <Format extends GPUVectorFormat>(
      name: string,
      format: Format,
      data: Float32Array
    ) => uploads.view(name, data, format)
  };
  const grid: GPURasterMetadata = {
    width: 4,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0],
    pixelInterpretation: 'area'
  };
  const weights = {
    offsets: input('uint32', [0, 1, 3, 5, 6]),
    neighbors: input('uint32', [1, 0, 2, 1, 3, 2]),
    weights: input('float32', [1, 1, 1, 1, 1, 1])
  };
  switch (element) {
    case 'addZonalDistributionRecipe': {
      const result = await runAddZonalDistributionRecipe(fixture.graph, {
        source: {
          kind: 'zones',
          zones: fixture.input('zones', Uint32Array.from(reference.zones), 'uint32', 8),
          zoneCount: 4
        },
        values: fixture.input(
          'values',
          Float32Array.from(reference.values.map(value => value ?? NaN)),
          'float32',
          8
        ),
        probabilities: fixture.input(
          'probabilities',
          Float32Array.from(reference.probabilities),
          'float32',
          5
        )
      });

      return captureOutputs(result.outputs);
    }
    case 'addRasterZonalDistributionRecipe': {
      const result = await runAddRasterZonalDistributionRecipe(fixture.graph, {
        grid,
        input: {
          id: 'raw',
          format: 'sint32',
          storage: {
            kind: 'buffer',
            values: fixture.input('raw', new Int32Array([1, 3, -9, 8]), 'sint32', 4)
          },
          noDataValue: -9,
          scale: 2,
          offset: 10
        },
        zones: fixture.input('zones', new Uint32Array([0, 0, 1, 1]), 'uint32', 4),
        zoneCount: 3
      });

      return captureOutputs(result.outputs);
    }
    case 'addExposureByZoneRecipe': {
      const result = await runAddExposureByZoneRecipe(fixture.graph, {
        grid,
        zones: fixture.input('zones', new Uint32Array([0, 0, 0, 1]), 'uint32', 4),
        zoneCount: 2,
        hazard: {
          id: 'hazard',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: fixture.input('hazard', new Float32Array([5, NaN, 1, 8]), 'float32', 4)
          }
        },
        population: {
          id: 'population',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: fixture.input('population', new Float32Array([10, 20, 30, 0]), 'float32', 4)
          }
        },
        weightMeaning: 'count',
        threshold: fixture.input('threshold', new Float32Array([4]), 'float32', 1)
      });

      return captureOutputs(result.outputs);
    }
    case 'addRegionalRasterChangeRecipe': {
      const result = await runAddRegionalRasterChangeRecipe(fixture.graph, {
        grid,
        change: {
          slices: fixture.input(
            'slices',
            new Float32Array([1, 3, NaN, 4, 5, 4, 2, 2]),
            'float32',
            8
          ),
          sliceCount: 2,
          parameters: fixture.parameters(
            'parameters',
            'float32',
            getGPUChangeDetectionParameterValues({beforeSlice: 0, afterSlice: 1})
          )
        },
        zones: fixture.input('zones', new Uint32Array([0, 0, 1, 1]), 'uint32', 4),
        zoneCount: 2,
        cellAreas: fixture.input('areas', new Float32Array([2, 2, 3, 3]), 'float32', 4),
        threshold: fixture.input('threshold', new Float32Array([1]), 'float32', 1)
      });

      return captureOutputs(result.outputs);
    }
    case 'addSurfacePreparationRecipe': {
      const result = await runAddSurfacePreparationRecipe(fixture.graph, {
        grid,
        input: {
          id: 'source',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: fixture.input('values', new Float32Array([2, NaN, NaN, NaN]), 'float32', 4)
          }
        },
        maximumDistance: 1,
        method: 'nearest'
      });

      return captureOutputs(result.outputs);
    }
    case 'addTrackQualityRecipe': {
      const result = await runAddTrackQualityRecipe(fixture.graph, {
        trajectory: {
          positions: fixture.input(
            'positions',
            new Float32Array([0, 0, 1, 0, 2, 0, 20, 0]),
            'float32x2',
            4
          ),
          timestamps: fixture.input('times', new Float32Array([0, 0, 1, 10]), 'float32', 4),
          trackOffsets: fixture.input('offsets', new Uint32Array([0, 4]), 'uint32', 2)
        },
        thresholds: fixture.input('thresholds', new Float32Array([1.5, 5]), 'float32', 2)
      });

      return captureOutputs(result.outputs);
    }
    case 'addInterpolationDiagnosticsRecipe': {
      const result = await runAddInterpolationDiagnosticsRecipe(fixture.graph, {
        kriging: {
          positions: fixture.input(
            'positions',
            new Float32Array([0, 0, 0, 0, 1, 0, 1, 0]),
            'float32x2',
            4
          ),
          values: fixture.input('values', new Float32Array([2, 2, 2, 2]), 'float32', 4),
          grid,
          indexGridSize: [2, 1],
          indexBounds: [-1, -1, 4, 1],
          maximumNeighborCount: 4,
          parameters: fixture.parameters(
            'parameters',
            'float32',
            getGPUKrigingParameterValues(
              {
                searchRadius: Infinity,
                neighborCount: 4,
                minimumNeighborCount: 1,
                variogram: {model: 'spherical', range: 10, sill: 1, nugget: 0}
              },
              grid
            )
          )
        }
      });

      return captureOutputs(result.outputs);
    }
    case 'addDrainageNetworkRecipe': {
      const drainageGrid: GPURasterMetadata = {...grid, width: 3, height: 3};
      const result = await runAddDrainageNetworkRecipe(fixture.graph, {
        flow: {
          grid: drainageGrid,
          elevation: {
            id: 'dem',
            format: 'float32',
            storage: {
              kind: 'buffer',
              values: fixture.input(
                'dem',
                new Float32Array([9, 8, 7, 6, 5, 4, 3, 2, 1]),
                'float32',
                9
              )
            }
          },
          parameters: fixture.parameters(
            'parameters',
            'float32',
            getGPUTerrainFlowParameterValues({streamThreshold: 1}, drainageGrid)
          ),
          maximumAccumulationIterations: 32
        },
        maximumSteps: 9
      });

      return captureOutputs(result.outputs);
    }
    case 'GPULOSHResidualPermutationTest':
      await runGPULOSHResidualPermutationTest(graph, {
        weights,
        values: input('float32', [1, 2, 4, 8]),
        parameters: uploads.view(
          'permutation-parameters',
          getGPUPermutationParameterValues({seed: 17, permutationCount: 31}),
          'uint32'
        ),
        maximumPermutations: 31,
        maximumNeighbors: 2,
        alternative: 'greater',
        output: {
          statistic: output('statistic', 'float32', 4).view,
          status: output('status', 'uint32', 4).view,
          normalizedResiduals: output('normalizedResiduals', 'float32', 4).view,
          inference: {
            exceedances: output('exceedances', 'uint32', 4).view,
            pseudoPValues: output('pseudoPValues', 'float32', 4).view,
            status: {overflow: output('overflow', 'uint32', 1).view}
          }
        }
      });
      break;
    case 'GPULocalSpatialHeteroscedasticity':
      await runGPULocalSpatialHeteroscedasticity(graph, {
        weights,
        values: input('float32', [1, 2, 4, 8]),
        output: {
          statistic: output('statistic', 'float32', 4).view,
          status: output('status', 'uint32', 4).view
        }
      });
      break;
    case 'GPUMultivariateLocalGeary':
      await runGPUMultivariateLocalGeary(graph, {
        weights,
        values: input('float32', [1, 1, 2, 4, 4, 2, 8, 7]),
        variableCount: 2,
        output: {
          statistic: output('statistic', 'float32', 4).view,
          status: output('status', 'uint32', 4).view
        }
      });
      break;
    case 'addLocalVariabilityRecipe':
      return captureOutputs(
        (
          await runAddLocalVariabilityRecipe(graph, {
            weights,
            values: input('float32', [1, 2, 4, 8]),
            parameters: uploads.view(
              'parameters',
              getGPUSpatialAutocorrelationParameterValues({}),
              'float32'
            )
          })
        ).outputs
      );
    case 'GPUGeneralizedGeographicallyWeightedRegression':
      await runGPUGeneralizedGeographicallyWeightedRegression(graph, {
        family: 'poisson',
        bandwidthMode: 'adaptive',
        maximumNeighborCount: 6,
        positions: input('float32x2', [0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1]),
        predictors: input('float32', [-1, 0, 1, -1, 0, 1]),
        response: input('float32', [1, 2, 4, 1, 2, 4]),
        predictorCount: 1,
        parameters: input('float32', [6, 1e-5]),
        maximumIterations: 32,
        bandwidthSelection: {
          bandwidths: [4, 5],
          output: {
            scores: output('bandwidthScores', 'float32', 2).view,
            bandwidth: output('selectedBandwidth', 'float32', 1).view,
            status: output('bandwidthStatus', 'uint32', 1).view
          }
        },
        inference: {
          standardErrors: output('standardErrors', 'float32', 12).view,
          influence: output('influence', 'float32', 6).view
        },
        output: {
          coefficients: output('coefficients', 'float32', 12).view,
          status: {
            codes: output('codes', 'uint32', 6).view,
            iterationCount: output('iterationCount', 'uint32', 6).view
          },
          fitted: output('fitted', 'float32', 6).view
        }
      });
      break;
    case 'GPUMultiscaleGeographicallyWeightedRegression':
      await runGPUMultiscaleGeographicallyWeightedRegression(graph, {
        positions: input('float32x2', [0, 0, 0, 0, 0, 0]),
        predictors: input('float32', [-1, 0, 1]),
        response: input('float32', [1, 3, 5]),
        predictorCount: 1,
        parameters: input('float32', [1e-5, 1, 2]),
        maximumIterations: 4,
        inference: {
          output: {
            standardErrors: output('fullStandardErrors', 'float32', 6).view,
            effectiveParameters: output('effectiveParameters', 'float32', 2).view,
            summary: output('inferenceSummary', 'float32', 4).view,
            status: output('inferenceStatus', 'uint32', 1).view
          }
        },
        output: {
          coefficients: output('coefficients', 'float32', 6).view,
          status: {
            codes: output('codes', 'uint32', 3).view,
            iterationCount: output('iterationCount', 'uint32', 1).view
          },
          fitted: output('fitted', 'float32', 3).view
        }
      });
      break;
    case 'GPUSpatialRegimeRegression':
      await runGPUSpatialRegimeRegression(graph, {
        predictors: input('float32', [0, 1, 2, 0, 1, 2]),
        response: input('float32', [1, 3, 5, 4, 3, 2]),
        regimes: input('uint32', [0, 0, 0, 1, 1, 1]),
        predictorCount: 1,
        regimeCount: 3,
        output: {
          coefficients: output('coefficients', 'float32', 6).view,
          fitted: output('fitted', 'float32', 6).view,
          status: output('status', 'uint32', 3).view,
          counts: output('counts', 'uint32', 3).view
        }
      });
      break;
    case 'GPUSpatialPanelRegression':
      await runGPUSpatialPanelRegression(graph, {
        effects: 'unit',
        weights: {
          offsets: input('uint32', [0, 2, 4, 6, 8]),
          neighbors: input('uint32', [1, 3, 0, 2, 1, 3, 0, 2]),
          weights: input('float32', new Array(8).fill(0.5))
        },
        periodCount: 2,
        predictorCount: 1,
        predictors: input('float32', [1, 4, 2, 8, 3, 9, 4, 1]),
        response: input('float32', [2, 7, 3, 5, 6, 2, 9, 8]),
        output: {
          table: output('table', 'float32', 12).view,
          summary: output('summary', 'float32', 32).view,
          status: {code: output('code', 'uint32', 1).view}
        }
      });
      break;
    case 'GPUConcaveHull':
      await runGPUConcaveHull(graph, {
        positions: input('float32x2', [0, 0, 2, 0, 2, 2, 0, 2]),
        maximumCircumradius: 2,
        output: {
          triangles: output('triangles', 'uint32x3', 3).view,
          triangleValidity: output('triangleValidity', 'uint32', 3).view,
          boundaryEdges: output('boundaryEdges', 'uint32x2', 9).view,
          edgeValidity: output('edgeValidity', 'uint32', 9).view,
          status: {converged: output('converged', 'uint32', 1).view}
        }
      });
      break;
    case 'GPUGroupConcaveHull':
      await runGPUGroupConcaveHull(graph, {
        positions: input('float32x2', [0, 0, 1, 0, 0, 1, 10, 0, 11, 0, 10, 1]),
        groupIds: input('uint32', [0, 0, 0, 1, 1, 1]),
        groupCount: 3,
        maximumCircumradius: 1,
        output: {
          triangles: output('triangles', 'uint32x3', 21).view,
          triangleValidity: output('triangleValidity', 'uint32', 21).view,
          boundaryEdges: output('boundaryEdges', 'uint32x2', 63).view,
          edgeValidity: output('edgeValidity', 'uint32', 63).view,
          status: {converged: output('converged', 'uint32', 3).view}
        }
      });
      break;
    case 'GPUConstrainedDelaunayTriangulation':
      await runGPUConstrainedDelaunayTriangulation(graph, {
        positions: input('float32x2', [0, 0, 2, 0, 2, 2, 0, 2]),
        featureOffsets: input('uint32', [0, 1]),
        polygonOffsets: input('uint32', [0, 1]),
        ringOffsets: input('uint32', [0, 4]),
        output: {
          indices: output('indices', 'uint32', 6).view,
          valid: output('valid', 'uint32', 1).view,
          status: {converged: output('converged', 'uint32', 1).view}
        }
      });
      break;
    case 'GPUBoundedFacilityAllocation':
      await runGPUBoundedFacilityAllocation(graph, {
        operation: 'p-center',
        facilityCount: 3,
        selectCount: 2,
        costs: input('float32', [1, 4, 5, 2, 1, 4, 5, 3, 1]),
        demand: input('float32', [1, 1, 1]),
        capacities: input('float32', [1, 1, 1]),
        output: {
          selectedFacilities: output('selectedFacilities', 'uint32', 3).view,
          assignments: output('assignments', 'uint32', 3).view,
          objective: output('objective', 'float32', 1).view,
          status: output('status', 'uint32', 3).view
        }
      });
      break;
    case 'GPUH3CellIdentities':
      await runGPUH3CellIdentities(graph, {
        cells: input('uint32x2', [0x59ffffff, 0x8928308]),
        operation: 'directed-edges',
        output: output('directedEdges', 'uint32x2', 6).view
      });
      break;
    case 'addCellFlowTopologyRecipe':
      return captureOutputs(
        (
          await runAddCellFlowTopologyRecipe(graph, {
            cells: input('uint32x2', [0x59ffffff, 0x8928308]),
            origins: input('uint32', [0]),
            destinations: input('uint32', [0]),
            values: input('float32', [5])
          })
        ).outputs
      );
    default:
      throw new Error(`Unknown expansion lens ${element}`);
  }
  return captures;
}
