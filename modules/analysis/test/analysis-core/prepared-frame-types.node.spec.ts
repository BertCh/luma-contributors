// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {GPUSpatialJoinPrepared} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUStudyWindowPreparation} from '../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {
  GPUSpatialJoinCandidates,
  type GPUSpatialJoinCandidatesProps
} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates';
import type {GPUSpatialPredicateJoinProps} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import type {GPUPointInPolygonJoinProps} from '../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import type {GPUNearestFeatureJoinProps} from '../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';

function registerVoidCandidates(
  graph: GPUCommandGraph<void>,
  props: GPUSpatialJoinCandidatesProps
) {
  if (props.prepared && !props.prepared.isDeclaredIn(graph)) graph.add(props.prepared);
  graph.add(new GPUSpatialJoinCandidates(props));
}

function registerTypedConsumers<Frame>(
  graph: GPUCommandGraph<Frame>,
  props:
    | GPUSpatialJoinCandidatesProps<Frame>
    | GPUSpatialPredicateJoinProps<Frame>
    | GPUPointInPolygonJoinProps<GraphDataView<'uint32'>, Frame>
    | GPUNearestFeatureJoinProps<
        GraphDataView<'uint32'>,
        GraphDataView<'float32'>,
        GPUNearestFeatureJoinProps['queries'],
        Frame
      >
) {
  if (props.prepared && !props.prepared.isDeclaredIn(graph)) graph.add(props.prepared);
}

it('uses the declared frame type for prepared handles and their rebuild predicates', () => {
  type Frame = {geometryChanged: boolean};
  expectTypeOf<GPUSpatialJoinCandidatesProps<Frame>['prepared']>().toEqualTypeOf<
    GPUSpatialJoinPrepared<Frame> | undefined
  >();
  expectTypeOf<GPUSpatialPredicateJoinProps<Frame>['prepared']>().toEqualTypeOf<
    GPUSpatialJoinPrepared<Frame> | undefined
  >();
  expectTypeOf<
    GPUPointInPolygonJoinProps<GraphDataView<'uint32'>, Frame>['prepared']
  >().toEqualTypeOf<GPUSpatialJoinPrepared<Frame> | undefined>();
  expectTypeOf<
    GPUNearestFeatureJoinProps<
      GraphDataView<'uint32'>,
      GraphDataView<'float32'>,
      GPUNearestFeatureJoinProps['queries'],
      Frame
    >['prepared']
  >().toEqualTypeOf<GPUSpatialJoinPrepared<Frame> | undefined>();
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph<Frame>(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 3);
  const join = new GPUSpatialJoinPrepared({
    geometry: {kind: 'points', positions},
    rebuildWhen: (frame: Frame) => frame.geometryChanged
  });
  const window = new GPUStudyWindowPreparation({
    positions,
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    samplingGrid: [2, 2],
    rebuildWhen: (frame: Frame) => frame.geometryChanged
  });
  try {
    expect(join.props.rebuildWhen?.({geometryChanged: true})).toBe(true);
    expect(window.props.rebuildWhen?.({geometryChanged: false})).toBe(false);
    const joinNodes = join.getCommandNodes(graph);
    const windowNodes = window.getCommandNodes(graph);
    expect(joinNodes.length).toBeGreaterThan(0);
    expect(windowNodes.length).toBeGreaterThan(0);
    // Typed handles remain usable as a consumer's prepared-index reference.
    const sharedIndex: GPUSpatialJoinPrepared<never> = join;
    expect(sharedIndex).toBe(join);
    if (false) {
      registerVoidCandidates(
        new GPUCommandGraph<void>(device),
        {} as GPUSpatialJoinCandidatesProps
      );
      registerTypedConsumers(graph, {} as GPUSpatialJoinCandidatesProps<Frame>);
      const typedProps = {} as GPUSpatialJoinCandidatesProps<Frame>;
      if (typedProps.prepared) {
        // @ts-expect-error Reading the handle out of consumer props must retain its frame type.
        new GPUCommandGraph<void>(device).add(typedProps.prepared);
      }
      // @ts-expect-error A frame-aware predicate cannot be added to a void graph.
      new GPUCommandGraph<void>(device).add(join);
      // @ts-expect-error Both the window and its graph use the same frame type.
      window.getCommandNodes(new GPUCommandGraph<void>(device));
      // @ts-expect-error A frame predicate needs its declared geometryChanged field.
      join.props.rebuildWhen?.({unrelated: true});
    }
  } finally {
    join.destroy();
    window.destroy();
    device.destroy();
  }
});
