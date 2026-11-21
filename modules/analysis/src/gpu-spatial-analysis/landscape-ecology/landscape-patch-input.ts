// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphViewsBelongToGraph} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateLandscapeViews} from './landscape-validation';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  createPatchKeysNode,
  createPatchUnresolvedNodes,
  getPatchLabelViews,
  validatePatchLabels,
  type GPURasterPatchLabels
} from '../../raster-analysis/raster-patches/patch-labels';
import {createRecipeKernelNode} from '../recipes/expansion-recipe-utils';

/** Explicit alternatives: zero-based patch rows, or raster labels with zero background. */
export type LandscapePatchInput = {
  /** Zero-based patch rows; rows at least patchClasses.length are background. Supply only this or denseLabels. */
  patches?: GraphDataView<'uint32'>;
  /** One-based raster labels with zero background. Guards suppress unfinished results. Supply only this or patches. */
  denseLabels?: GPURasterPatchLabels;
};

/** Validate without allocating; the returned view is only used for host layout checks. @internal */
export function prepareLandscapePatchInput<
  Parameters,
  Props extends LandscapePatchInput & {
    width: number;
    height: number;
    cellSize: readonly [number, number];
  }
>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: Props
): Props & {patches: GraphDataView<'uint32'>} {
  props = snapshotConstructionFields(props, [
    'patches',
    'denseLabels',
    'patchClasses',
    'width',
    'height',
    'cellSize',
    'classCount',
    'connectivity',
    'reconcilePatches',
    'countBoundary',
    'countBackground',
    'backgroundClass',
    'contagionWeighting',
    'output',
    'mode',
    'edgeDepth',
    'searchRadius',
    'distanceModel',
    'coreModel',
    'coreBoundary',
    'candidateRadius',
    'candidateCapacity'
  ] as readonly (keyof Props)[]);
  const source =
    props.denseLabels &&
    snapshotConstructionFields(props.denseLabels, ['grid', 'labels', 'mask', 'status']);
  const denseLabels = source && {
    ...source,
    grid: snapshotConstructionFields(source.grid, [
      'width',
      'height',
      'affine',
      'pixelInterpretation',
      'coordinateReferenceSystem',
      'level',
      'levelZeroOrigin'
    ] as readonly (keyof GPURasterPatchLabels['grid'])[]),
    status:
      source.status &&
      snapshotConstructionFields(source.status, ['converged', 'overflow', 'componentCount'])
  };
  validateRequirement(
    id,
    props.cellSize.length === 2,
    `cellSize must have exactly 2 entries; received ${props.cellSize.length}`
  );
  validateRequirement(
    id,
    Boolean(props.patches) !== Boolean(props.denseLabels),
    'supply exactly one of patches (zero-based rows) or denseLabels (one-based raster labels)'
  );
  if (denseLabels) {
    validatePatchLabels(id, denseLabels);
    validateRequirement(
      id,
      denseLabels.grid.width === props.width && denseLabels.grid.height === props.height,
      'denseLabels.grid width and height must match width and height'
    );
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        getPatchLabelViews(denseLabels).map((view, index) => [`denseLabels.${index}`, view])
      )
    );
  }
  if (denseLabels)
    denseLabels.grid = {
      ...denseLabels.grid,
      affine: [...denseLabels.grid.affine] as GPURasterPatchLabels['grid']['affine'],
      ...(denseLabels.grid.levelZeroOrigin
        ? {levelZeroOrigin: [...denseLabels.grid.levelZeroOrigin] as readonly [number, number]}
        : {}),
      ...(denseLabels.grid.coordinateReferenceSystem
        ? {
            coordinateReferenceSystem: snapshotConstructionFields(
              denseLabels.grid.coordinateReferenceSystem,
              ['authority', 'wellKnownText']
            )
          }
        : {})
    };
  return {
    ...props,
    id,
    denseLabels,
    cellSize: [props.cellSize[0], props.cellSize[1]] as const,
    patches: props.patches ?? denseLabels!.labels
  };
}

/** Preserve the legacy reconciliation path by adapting dense labels exactly once. @internal */
export function addLandscapePatchInput<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: LandscapePatchInput & {patchClasses: GraphDataView<'uint32'>}
): GraphDataView<'uint32'> {
  if (!props.denseLabels) return props.patches!;
  const {node, keys} = createPatchKeysNode(
    graph,
    `${id}-dense-input`,
    id,
    props.denseLabels,
    props.patchClasses.length
  );
  graph.add(node);
  const patches = createTransientView(graph, `${id}-dense-input-owners`, 'uint32', keys.length);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-dense-input-owners`,
      operation: id,
      invocationCount: keys.length,
      bindings: [
        {name: 'keys', view: keys, type: 'u32', access: 'read'},
        {name: 'owners', view: patches, type: 'u32', access: 'read_write'}
      ],
      body: 'let label=keys[keysOffset+index];owners[ownersOffset+index]=select(0xffffffffu,label-1u,label!=0u);'
    })
  );
  return patches;
}

/** Include every upstream guard in preflight, including input/output alias checks. @internal */
export function validateLandscapePatchDestinations<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: LandscapePatchInput,
  outputs: readonly (GraphDataView | undefined)[]
): void {
  if (input.denseLabels) {
    const inputs = getPatchLabelViews(input.denseLabels);
    validateGraphViewsBelongToGraph(id, graph, [...inputs, ...outputs]);
    validateLandscapeViews(
      id,
      Object.fromEntries(inputs.map((view, index) => [`denseLabels.${index}`, view])),
      Object.fromEntries(outputs.map((view, index) => [`output.${index}`, view])),
      []
    );
  }
}

/** Suppress results from unfinished labels with the raster family's unresolved sentinel. @internal */
export function suppressUnresolvedLandscapeResults<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: LandscapePatchInput & {patchClasses: GraphDataView<'uint32'>},
  outputs: readonly GraphDataView[],
  lossFlags: readonly {view: GraphDataView<'uint32'>; value: 0 | 1}[] = []
): void {
  if (!input.denseLabels) return;
  let labels = input.denseLabels;
  if (labels.status?.componentCount) {
    const overflow = createTransientView(graph, `${id}-dense-input-capacity-loss`, 'uint32', 1);
    const upstreamOverflow = labels.status.overflow;
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-dense-input-capacity-loss`,
        operation: id,
        invocationCount: 1,
        bindings: [
          {name: 'componentCount', view: labels.status.componentCount, type: 'u32', access: 'read'},
          ...(upstreamOverflow
            ? [
                {
                  name: 'upstreamOverflow',
                  view: upstreamOverflow,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'overflow', view: overflow, type: 'u32', access: 'read_write'}
        ],
        body: `overflow[overflowOffset]=u32(componentCount[componentCountOffset]>${input.patchClasses.length}u${upstreamOverflow ? '||upstreamOverflow[upstreamOverflowOffset]!=0u' : ''});`
      })
    );
    labels = {...labels, status: {...labels.status, overflow}};
  }
  const scalarOutputs = outputs.map(view => {
    const layout = {byteOffset: view.byteOffset, length: (view.length * view.rowByteLength) / 4};
    return view.format.startsWith('float32')
      ? {
          type: 'f32' as const,
          view: graph.createDataView(view.buffer, {...layout, format: 'float32'})
        }
      : {
          type: 'u32' as const,
          view: graph.createDataView(view.buffer, {...layout, format: 'uint32'})
        };
  });
  graph.add(createPatchUnresolvedNodes(graph, `${id}-dense-input`, id, labels, scalarOutputs));
  // Finished masks/flags and bounded counts cannot contain the unresolved word. Normalize it
  // after suppression so validity and readable counts remain zero, and loss remains a flag.
  for (const [outputIndex, output] of lossFlags.entries()) {
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-dense-input-loss-flag-${outputIndex}`,
        operation: id,
        invocationCount: output.view.length,
        bindings: [{name: 'lossValues', view: output.view, type: 'u32', access: 'read_write'}],
        body: `let value=lossValues[lossValuesOffset+index];lossValues[lossValuesOffset+index]=select(value,${output.value}u,value==0xffffffffu);`
      })
    );
  }
}
