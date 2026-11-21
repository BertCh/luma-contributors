// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  addLandscapeMetricsRecipe,
  addPatchProximityRecipe
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import type {RecipeTestFixture} from '../recipes/recipe-harness';

export function declareLandscapeDenseLabelsFixture(
  fixture: RecipeTestFixture,
  route: 'metrics' | 'reference' | 'boundary',
  reconcilePatches: boolean
) {
  const results = [];
  for (const [name, converged, overflow, componentCount, expected] of [
    ['complete', 1, 0, 1, 2],
    ['unfinished', 0, 0, 1, NaN],
    ['overflow', 1, 1, 1, NaN],
    ['too-many-components', 1, 0, 2, NaN],
    ['no-components', 1, 0, 0, 0]
  ] as const) {
    const reconciliation = reconcilePatches
      ? {
          patches: fixture.output(`caller-${name}-reconciled`, 'uint32', 3),
          parents: fixture.output(`caller-${name}-parents`, 'uint32', 1)
        }
      : undefined;
    const reconciliationOutput = reconciliation
      ? {reconciledPatches: reconciliation.patches.view, patchParents: reconciliation.parents.view}
      : {};
    const result = fixture.output(name, 'float32x4', 1);
    const boundary =
      route === 'boundary'
        ? {
            validity: fixture.output(`${name}-validity`, 'uint32', 1),
            count: fixture.output(`${name}-valid-count`, 'uint32', 1),
            candidateOverflow: fixture.output(`${name}-candidate-loss`, 'uint32', 1),
            overflow: fixture.output(`${name}-final-loss`, 'uint32', 1)
          }
        : undefined;
    const props = {
      id: name,
      denseLabels: {
        grid: {
          width: 3,
          height: 1,
          affine: [1, 0, 0, 0, 1, 0] as const,
          pixelInterpretation: 'area' as const
        },
        labels: fixture.input(
          `${name}-labels`,
          new Uint32Array(name === 'too-many-components' ? [1, 2, 0] : [0, 1, 1]),
          'uint32',
          3
        ),
        status: {
          converged: fixture.input(`${name}-converged`, new Uint32Array([converged]), 'uint32', 1),
          overflow: fixture.input(`${name}-overflow`, new Uint32Array([overflow]), 'uint32', 1),
          componentCount: fixture.input(
            `${name}-count`,
            new Uint32Array([componentCount]),
            'uint32',
            1
          )
        }
      },
      patchClasses: fixture.input(`input-${name}-classes`, new Uint32Array([0]), 'uint32', 1),
      width: 3,
      height: 1,
      cellSize: [1, 1] as const,
      connectivity: 4 as const,
      reconcilePatches
    };
    if (route === 'metrics')
      addLandscapeMetricsRecipe(fixture.graph, {
        ...props,
        classCount: 1,
        output: {...reconciliationOutput, patchMetrics: result.view}
      });
    else if (route === 'reference')
      addPatchProximityRecipe(fixture.graph, {
        ...props,
        mode: 'reference',
        edgeDepth: 0,
        searchRadius: 1,
        output: {...reconciliationOutput, metrics: result.view}
      });
    else {
      addPatchProximityRecipe(fixture.graph, {
        ...props,
        mode: 'boundary',
        edgeDepth: 0,
        searchRadius: 1,
        output: {
          ...reconciliationOutput,
          metrics: result.view,
          validity: boundary!.validity.view,
          status: {
            count: boundary!.count.view,
            candidateOverflow: boundary!.candidateOverflow.view,
            overflow: boundary!.overflow.view
          }
        }
      });
    }
    // One unit square per foreground cell: area = 2; background contributes zero.
    results.push({result, expected, boundary, reconciliation});
  }

  return results;
}
