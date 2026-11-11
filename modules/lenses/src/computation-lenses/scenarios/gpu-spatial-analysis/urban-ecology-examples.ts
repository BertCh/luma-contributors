// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {getRasterResistanceConnectivityParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {S2Uploads, S2CaptureSpec} from './s2-lens-run';
import {runExample as runStreetProfiles} from './street-profiles-recipe.example';
import {runExample as runBuildingAdjacency} from './building-adjacency-recipe.example';
import {runExample as runIsovists} from './isovists-recipe.example';
import {runExample as runVisibilityGraph} from './visibility-graph-recipe.example';
import {runExample as runBuildingTessellation} from './building-tessellation-recipe.example';
import {runExample as runLandscapeMetrics} from './landscape-metrics-recipe.example';
import {runExample as runPatchProximity} from './patch-proximity-recipe.example';
import {runExample as runResistanceConnectivity} from './resistance-connectivity-recipe.example';
import {runExample as runSparseResistanceConnectivity} from './sparse-resistance-connectivity-recipe.example';
import {runExample as runRasterResistanceConnectivity} from './raster-resistance-connectivity-recipe.example';
import {runExample as runSegregationDecomposition} from './segregation-decomposition-recipe.example';

import {runExample as runaddSegregationCounterfactualRecipe} from './segregation-counterfactual-recipe.example';
import {runExample as runaddPopulationSupportAlignmentRecipe} from './population-support-alignment-recipe.example';

/** Small explicit fixtures run the same published examples that appear in the lens source rung. */
export async function buildUrbanEcologyScenario(
  element: string,
  graph: GPUCommandGraph<undefined>,
  uploads: S2Uploads
): Promise<Record<string, S2CaptureSpec>> {
  const captures: Record<string, S2CaptureSpec> = {};
  let serial = 0;
  const input = <Format extends GPUVectorFormat>(
    name: string,
    format: Format,
    values: number[]
  ) => {
    const view = uploads.view(
      `urban-ecology-${name}-${serial++}`,
      format.startsWith('uint') ? Uint32Array.from(values) : Float32Array.from(values),
      format
    );
    captures[`input-${name}`] = {view, role: 'input'};
    return view;
  };
  const capture = (value: unknown, name: string) => {
    if (!value || typeof value !== 'object') return;
    if ('buffer' in value && 'format' in value && 'length' in value) {
      // Recipe output trees are heterogeneous; narrow view metadata at this capture boundary.
      const source = value as GraphDataView;
      const components = Number(source.format.match(/x([234])$/)?.[1] ?? 1);
      const destination = uploads.view(
        `urban-ecology-capture-${serial++}`,
        source.format.startsWith('uint')
          ? new Uint32Array(source.length * components)
          : new Float32Array(source.length * components),
        source.format,
        source.length
      );
      const type = source.format.startsWith('uint') ? 'u32' : 'f32';
      graph.add(
        createWGSLKernelNode(graph, {
          id: `urban-ecology-copy-${serial++}`,
          operation: 'lens-output-copy',
          invocationCount: source.length * components,
          bindings: [
            {name: 'source', view: source, type, access: 'read'},
            {name: 'destination', view: destination, type, access: 'read_write'}
          ],
          body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
        })
      );
      captures[name] = {view: destination, role: 'output'};
    } else {
      for (const [childName, child] of Object.entries(value))
        capture(child, `${name}-${childName}`);
    }
  };
  const finish = (result: {outputs: unknown; status?: unknown}) => {
    capture(result.outputs, 'output');
    if (result.status) capture(result.status, 'status');
    return captures;
  };
  switch (element) {
    case 'addStreetProfilesRecipe':
      return finish(
        await runStreetProfiles(graph, {
          positions: input('positions', 'float32x2', [0, 0]),
          normals: input('normals', 'float32x2', [0, 1]),
          segments: input('segments', 'float32x4', [-10, 2, 10, 2, -10, -3, 10, -3]),
          segmentBuildingIds: input('building-ids', 'uint32', [0, 1]),
          buildingHeights: input('building-heights', 'float32', [6, 9]),
          maximumDistance: 10,
          sampleSpacing: 1
        })
      );
    case 'addBuildingAdjacencyRecipe':
      return finish(
        await runBuildingAdjacency(graph, {
          id: 'buildings',
          sharedPaths: {
            runs: {
              positions: input('wall-positions', 'float32x2', [0, 0, 0, 2]),
              offsets: input('wall-offsets', 'uint32', [0, 2]),
              leftSourceIds: input('left-lines', 'uint32', [0]),
              rightSourceIds: input('right-lines', 'uint32', [1]),
              forward: input('forward', 'uint32', [0])
            },
            status: {count: input('wall-count', 'uint32', [1])}
          },
          lineBuildingIds: input('line-building-ids', 'uint32', [0, 1, 2]),
          orientations: input('orientations', 'float32', [Math.PI * 0.49, -Math.PI * 0.49, 0]),
          structures: {
            geometry: {
              kind: 'polygons',
              positions: input(
                'building-positions',
                'float32x2',
                [-1, -1, 0, -1, 0, 1, -1, 1, 0, -1, 1, -1, 1, 1, 0, 1, 3, 0, 4, 0, 4, 1, 3, 1]
              ),
              ringOffsets: input('building-rings', 'uint32', [0, 4, 8, 12]),
              polygonOffsets: input('building-polygons', 'uint32', [0, 1, 2, 3]),
              featureOffsets: input('building-features', 'uint32', [0, 1, 2, 3])
            },
            vertexTolerance: 1 / 1024,
            // Three rectangles need at most twelve edges; keep this captured graph compact.
            capacity: {intersections: 16, nodedSegments: 32, vertices: 24, rings: 8}
          }
        })
      );
    case 'addIsovistsRecipe':
      return finish(
        await runIsovists(graph, {
          positions: input('observers', 'float32x2', [0, 0, 2, 0]),
          segments: input(
            'walls',
            'float32x4',
            [-2, -2, 2, -2, 2, -2, 2, 2, 2, 2, -2, 2, -2, 2, -2, -2]
          ),
          rayCount: 8,
          region: {fillRule: 'even-odd', freeSpace: 'inside'},
          maximumDistance: 10
        })
      );
    case 'addVisibilityGraphRecipe':
      return finish(
        await runVisibilityGraph(graph, {
          positions: input('observers', 'float32x2', [0, 0, 1, 0, 2, 0]),
          segments: input('walls', 'float32x4', [0, 10, 2, 10]),
          maximumDistance: 1.1,
          edgeCapacity: 1
        })
      );
    case 'addBuildingTessellationRecipe':
      return finish(
        await runBuildingTessellation(graph, {
          segments: input(
            'building-boundaries',
            'float32x4',
            [
              -2, -0.5, -1, -0.5, -1, -0.5, -1, 0.5, -1, 0.5, -2, 0.5, -2, 0.5, -2, -0.5, 1, -0.5,
              2, -0.5, 2, -0.5, 2, 0.5, 2, 0.5, 1, 0.5, 1, 0.5, 1, -0.5
            ]
          ),
          segmentBuildingIds: input('building-ids', 'uint32', [0, 0, 0, 0, 1, 1, 1, 1]),
          buildingCount: 2,
          enclosure: input('enclosure', 'float32x2', [-3, -2, 3, -2, 3, 2, -3, 2]),
          sampleSpacing: 1,
          sampleCapacity: 8,
          vertexTolerance: 1 / 1024,
          capacity: {intersections: 256, nodedSegments: 256, vertices: 128, rings: 32}
        })
      );
    case 'addLandscapeMetricsRecipe':
      return finish(
        await runLandscapeMetrics(graph, {
          patches: input('patches', 'uint32', [0, 0, 1, 1, 0, 0, 1, 1]),
          patchClasses: input('patch-classes', 'uint32', [0, 1]),
          width: 4,
          height: 2,
          classCount: 2,
          cellSize: [1, 1],
          connectivity: 4
        })
      );
    case 'addPatchProximityRecipe':
      return finish(
        await runPatchProximity(graph, {
          patches: input('patches', 'uint32', [0, 0, 0xffffffff, 1, 1]),
          patchClasses: input('patch-classes', 'uint32', [0, 0]),
          width: 5,
          height: 1,
          cellSize: [1, 1],
          edgeDepth: 0.4,
          searchRadius: 2
        })
      );
    case 'addResistanceConnectivityRecipe':
      return finish(
        await runResistanceConnectivity(graph, {
          edges: input('edges', 'uint32x2', [0, 1, 1, 2, 0, 2]),
          conductances: input('conductances', 'float32', [1, 1, 0.5]),
          vertexCount: 3,
          source: 0,
          sink: 2
        })
      );
    case 'addSparseResistanceConnectivityRecipe':
      // A 3 x 3 lattice with unit conductances and one weak diagonal chord, corner to corner.
      return finish(
        await runSparseResistanceConnectivity(graph, {
          edges: input(
            'edges',
            'uint32x2',
            [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 0, 3, 3, 6, 1, 4, 4, 7, 2, 5, 5, 8, 0, 4]
          ),
          conductances: input(
            'conductances',
            'float32',
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.25]
          ),
          vertexCount: 9,
          source: 0,
          sink: 8,
          tolerance: 1e-5,
          maximumIterations: 64
        })
      );
    case 'addRasterResistanceConnectivityRecipe': {
      // A 6 x 5 resistance raster: a no-data wall in column 3 with a gap in row 3, and a
      // high-resistance cell beside the gap. Unit current flows from the west to the east edge.
      const width = 6;
      const height = 5;
      const cellCount = width * height;
      const grid = {
        width,
        height,
        affine: [1, 0, 0, 0, -1, 0] as const,
        pixelInterpretation: 'area' as const
      };
      const resistance = Array.from({length: cellCount}, (_, cell) => (cell === 14 ? 8 : 1));
      const validity = resistance.map((_, cell) => (cell % width === 3 && cell !== 21 ? 0 : 1));
      const output = <Format extends GPUVectorFormat>(
        name: string,
        format: Format,
        length: number
      ) =>
        uploads.view(
          `urban-ecology-output-${name}-${serial++}`,
          format.startsWith('uint')
            ? new Uint32Array(length * Number(format.match(/x([234])$/)?.[1] ?? 1))
            : new Float32Array(length * Number(format.match(/x([234])$/)?.[1] ?? 1)),
          format,
          length
        );
      const result = await runRasterResistanceConnectivity(graph, {
        grid,
        resistance: {
          id: 'resistance',
          format: 'float32',
          storage: {kind: 'buffer', values: input('resistance', 'float32', resistance)},
          validity: input('validity', 'uint32', validity)
        },
        parameters: input(
          'parameters',
          'float32',
          Array.from(getRasterResistanceConnectivityParameterValues(grid))
        ),
        neighborhood: 8,
        source: 2 * width,
        sink: 2 * width + width - 1,
        tolerance: 1e-5,
        // The solve converges in 17 iterations; 64 keeps the unrolled solver (and its record) docs-sized.
        maximumIterations: 64,
        output: {
          potentials: output('potentials', 'float32', cellCount),
          vertexCurrents: output('vertex-currents', 'float32', cellCount),
          diagnostics: output('diagnostics', 'float32x4', 1),
          solverStatus: output('solver-status', 'uint32x4', 1)
        }
      });
      capture(result.intermediates.conductances, 'intermediate-conductances');
      return finish(result);
    }
    case 'addSegregationDecompositionRecipe':
      return finish(
        await runSegregationDecomposition(graph, {
          populations: [
            input('baseline-populations', 'float32x2', [9, 1, 1, 9]),
            input('composition-counterfactual', 'float32x2', [8, 2, 2, 8]),
            input('allocation-counterfactual', 'float32x2', [7, 3, 3, 7]),
            input('observed-populations', 'float32x2', [5, 5, 5, 5])
          ],
          supportId: 'two-aligned-synthetic-zones',
          counterfactualMethod: 'explicit-supplied-counterfactuals'
        })
      );
    case 'addSegregationCounterfactualRecipe':
      return finish(
        await runaddSegregationCounterfactualRecipe(graph, {
          baseline: input('baseline', 'float32x2', [9, 1, 1, 9]),
          final: input('final', 'float32x2', [5, 5, 5, 5]),
          supportId: 'two-common-units',
          method: 'composition'
        })
      );
    case 'addPopulationSupportAlignmentRecipe':
      return finish(
        await runaddPopulationSupportAlignmentRecipe(graph, {
          populations: input('populations', 'float32x2', [2, 8, 6, 4]),
          sourceIds: input('source-ids', 'uint32', [20, 10]),
          targetIds: input('target-ids', 'uint32', [10, 20])
        })
      );
    default:
      throw new Error(`Unknown urban/ecology scenario: ${element}`);
  }
}
