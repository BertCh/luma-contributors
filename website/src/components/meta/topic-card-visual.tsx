// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The picture on each how-it-works index card: a small live luma.gl scene when the topic has one,
// over the static SVG thumbnail that prerendering and GPU-less browsers keep.

import {useMemo} from 'react';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import type {GpuSceneLoader} from '../../gpu/types';
import {layoutLensGraph} from '../../lens/layout-graph';
import type {LensViewModel} from '../../lens/types';
import {COLUMNS_LAYOUT} from './columns-demo';
import {makeFlowGraph} from './lens-graph-panel';
import {makePackageFlowGraph} from './package-layers';
import {createColumnsSceneLoader} from './scenes/columns-scene';
import {createFlowSceneLoader, type FlowSceneOptions} from './scenes/flow-scene';
import {createMapSceneLoader} from './scenes/map-scene';
import {createTilingSceneLoader, createTilingStore} from './scenes/tiling-scene';
import {createWorkgroupSceneLoader, DEFAULT_WORKGROUP_GRID} from './scenes/workgroup-scene';
import {TopicThumb} from './topic-thumbs';
import type {TopicSlug} from './topics';
import './meta.css';

const CARD_ASPECT_RATIO = 16 / 10;
const CARD_GRAPH_LAYOUT = {
  nodeWidth: 188,
  nodeHeight: 42,
  layerGap: 64,
  rowGap: 14,
  padding: 22,
  targetWidth: 1040
};

function makeGraphLoader(
  model: LensViewModel | undefined,
  options: FlowSceneOptions
): GpuSceneLoader | undefined {
  return model
    ? createFlowSceneLoader(
        makeFlowGraph(model, layoutLensGraph(model, CARD_GRAPH_LAYOUT)),
        options
      )
    : undefined;
}

export type TopicCardModels = {
  density?: LensViewModel;
  moran?: LensViewModel;
};

export function TopicCardVisual({slug, models}: {slug: TopicSlug; models: TopicCardModels}) {
  const loader = useMemo<GpuSceneLoader | undefined>(() => {
    switch (slug) {
      case 'anatomy':
        return createWorkgroupSceneLoader(
          {
            ...DEFAULT_WORKGROUP_GRID,
            columns: 8,
            rows: 5,
            laneSide: 5,
            invocationCount: 8 * 5 * 25 - 33
          },
          {interaction: 'dispatch'}
        );
      case 'command-graph':
        return makeGraphLoader(models.moran, {interaction: 'dependencies'});
      case 'scheduling':
        return makeGraphLoader(models.density, {interaction: 'scrub'});
      case 'kernels':
        return createTilingSceneLoader(createTilingStore(32));
      case 'data-on-the-gpu':
        return createColumnsSceneLoader(COLUMNS_LAYOUT);
      case 'lenses':
        return models.density?.map ? createMapSceneLoader(models.density.map) : undefined;
      case 'this-page':
        return createWorkgroupSceneLoader(
          {
            ...DEFAULT_WORKGROUP_GRID,
            columns: 12,
            rows: 6,
            laneSide: 4,
            invocationCount: 12 * 6 * 16 - 21
          },
          {interaction: 'trail'}
        );
      case 'upstream-first':
        return createFlowSceneLoader(makePackageFlowGraph(), {
          interaction: 'dependencies',
          dependencyDepth: 'transitive'
        });
      default:
        return undefined;
    }
  }, [slug, models]);

  const thumb = (
    <div className="topic-card__thumb-svg">
      <TopicThumb slug={slug} />
    </div>
  );
  if (!loader) return <div className="topic-card__live">{thumb}</div>;
  return (
    <div className="topic-card__live">
      <GpuCanvas
        scene={loader}
        fallback={thumb}
        seed={1}
        label={`Live picture for ${slug}`}
        aspectRatio={CARD_ASPECT_RATIO}
        interactive
        pointerSurface="card"
        hint={false}
      />
    </div>
  );
}
