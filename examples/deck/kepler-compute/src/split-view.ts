// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  WebMercatorViewport,
  Deck,
  type DeckProps,
  MapView,
  type MapViewState,
  Layer
} from '@deck.gl/core';
import {appState, type SecondaryViewCapture} from './app-state';
import {
  withExactGlobeControllers,
  withTouchRotate,
  syncTopLevelController,
  applyExactGlobeController
} from './globe-wheel-controller';
import React from 'react';
import {MapContainerFactory} from '@kepler.gl/components';
import {
  getResidentSubViewProjection,
  planResidentViews,
  createResidentViewLayerFilter
} from './resident-view-plan';
import {
  flattenDeckLayers,
  isGlslGlobeLayer,
  toWebGPUGlobeLayer,
  orderGlobeDepthDisk,
  getRootLayerId
} from './main-globe-layers';

/** Deck's GlobeViewport carries `resolution` (degrees; Kepler uses 5), which selects H3's polygon branch and the globe geometry variant. */
export function makeNativeMapViewport(
  mapState: ConstructorParameters<typeof WebMercatorViewport>[0] & {globe?: {enabled?: boolean}}
): WebMercatorViewport {
  const viewport = new WebMercatorViewport(mapState);
  // Deck freezes viewports, so layer the resolution over a derived object.
  return mapState.globe?.enabled
    ? Object.create(viewport, {resolution: {value: 5, enumerable: true}})
    : viewport;
}

/**
 * Dual and Swipe render in ONE Deck with two viewports on the shared device (a second Deck would
 * share the device's single default canvas context). `?views=1` forces the original single-view
 * host, which rejects split and swipe.
 */
export const VIEWS_ENABLED =
  typeof location === 'undefined' || new URLSearchParams(location.search).get('views') !== '1';

export const SUPPORTED_VIEW_COUNT = VIEWS_ENABLED ? 2 : 1;

type DeckEventHandler = (...parameters: never[]) => unknown;

/**
 * Kepler assembles the shared Deck's `viewState`, `onViewStateChange`, `onHover` and `onClick` for
 * map 0 after the host can intercept them. While split, give each view its own view-state and send
 * events from view 1 to map 1's native handlers (so hover, click and pan/zoom update map 1).
 */
class ResidentSplitDeck extends Deck {
  override setProps(props: DeckProps): void {
    const viewStates = appState.splitViewStates;
    if (!viewStates) {
      super.setProps(withExactGlobeControllers(withTouchRotate(props)));
      return;
    }
    props = withTouchRotate(props);
    const primary = props as unknown as Record<string, DeckEventHandler | undefined>;
    const route =
      (name: string, viewIdOf: (parameters: never) => unknown) =>
      (...parameters: never[]) => {
        const secondaryHandler = (
          appState.secondaryDeck?.props as unknown as Record<string, DeckEventHandler> | null
        )?.[name];
        const handler = viewIdOf(parameters[0]) === 'view-1' ? secondaryHandler : primary[name];
        return handler?.(...parameters);
      };
    super.setProps(
      syncTopLevelController({
        ...props,
        viewState: viewStates,
        // Deck applies a top-level controller to one view only; every split view needs its own.
        views: ([props.views].flat() as (MapView | undefined)[])
          .map(view =>
            view && !view.props.controller ? view.clone({controller: props.controller}) : view
          )

          .map(view =>
            view ? (applyExactGlobeController(view, props.controller) as MapView) : view
          ),
        onViewStateChange: route(
          'onViewStateChange',
          (parameters: {viewId?: string}) => parameters.viewId
        ),
        onHover: route('onHover', (info: {viewport?: {id: string}}) => info?.viewport?.id),
        onClick: route('onClick', (info: {viewport?: {id: string}}) => info?.viewport?.id)
      } as unknown as DeckProps)
    );
  }
}

export function setSwipePercentage(next: number | null): void {
  if (next === appState.swipePercentage) return;
  appState.swipePercentage = next;
  queueMicrotask(() => secondaryViewListeners.forEach(listener => listener()));
}

export const secondaryViewListeners = new Set<() => void>();

export function setSecondaryView(next: SecondaryViewCapture | null): void {
  const changed =
    next?.layers !== appState.secondaryView?.layers ||
    next?.viewState !== appState.secondaryView?.viewState;
  appState.secondaryView = next;
  if (changed) queueMicrotask(() => secondaryViewListeners.forEach(listener => listener()));
}

function getSwipeViews(
  props: React.ComponentProps<ReturnType<typeof MapContainerFactory>>,
  viewStates: Record<string, unknown>,
  baseView: SplitBaseView
): {views: SplitBaseView[]} {
  const {width, height} = props.mapState;
  const divider = Math.round(
    (width * (appState.swipePercentage ?? props.mapState.swipeComparePercentage ?? 50)) / 100
  );
  const views: SplitBaseView[] = [];
  for (const [id, x, viewWidth] of [
    ['view-0', 0, divider],
    ['view-1', divider, width - divider]
  ] as const) {
    if (viewWidth <= 0) continue;
    // The primary view class (MapView or GlobeView) makes the full-size viewport, so pitch,
    // bearing and globe projections are sliced the same way.
    const fullViewport = baseView.clone({id}).makeViewport({
      width,
      height,
      viewState: viewStates[id] as MapViewState
    });
    if (!fullViewport) continue;
    views.push(
      baseView.clone({
        id,
        x,
        width: viewWidth,
        projectionMatrix: getResidentSubViewProjection(
          fullViewport.projectionMatrix,
          width,
          x,
          viewWidth
        )
      })
    );
  }
  return {views};
}

/** The primary map's view (MapView or GlobeView); split views are clones of it. */
type SplitBaseView = MapView;

/** Merge the captured second map into the primary Deck properties: layers once, filtered per view. */
export function applySecondaryView<Properties extends Record<string, unknown>>(
  props: React.ComponentProps<ReturnType<typeof MapContainerFactory>>,
  result: Properties,
  internalViewStates: readonly [unknown, unknown]
): Properties {
  const secondary = appState.secondaryView;
  const splitMode = props.mapState.mapSplitMode;
  appState.splitViewStates = null;
  if (!VIEWS_ENABLED) return result;
  // The host always creates its Deck as the routing subclass; it is inert outside Dual and Swipe.
  // Deck merges props, so clear a split layer filter explicitly when leaving Dual or Swipe.
  const withDeckClass = {
    ...result,
    Deck: ResidentSplitDeck,
    layerFilter: result['layerFilter'] ?? null
  };
  if (!secondary || (splitMode !== 'DUAL_MAP' && splitMode !== 'SWIPE_COMPARE'))
    return withDeckClass;
  const plan = planResidentViews({
    layers: props.visState.layers.map(layer => ({
      id: layer.id,
      isVisible: Boolean(layer.config.isVisible)
    })),
    splitMaps: props.visState.splitMaps,
    mapSplitMode: props.mapState.mapSplitMode,
    supportedViewCount: SUPPORTED_VIEW_COUNT
  });
  if (!plan.supported) return withDeckClass;
  const viewStates = {'view-0': internalViewStates[0], 'view-1': internalViewStates[1]};
  appState.splitViewStates = viewStates;
  const secondaryLayers = flattenDeckLayers(secondary.layers)
    .filter(layer => !isGlslGlobeLayer(layer))
    .map(toWebGPUGlobeLayer);
  const baseView = ([result['views']].flat()[0] as SplitBaseView | undefined) ?? new MapView({});
  // Native hides a layer in the map that excludes it; the layer filter now owns per-view
  // visibility, so a layer hidden only in the primary map is drawn from the secondary instance.
  const primaryLayers = flattenDeckLayers(result['layers']).map(layer =>
    layer.props.visible === false
      ? (secondaryLayers.find(
          candidate => candidate.id === layer.id && candidate.props.visible !== false
        ) ?? layer)
      : layer
  );
  const known = new Set(primaryLayers.map(layer => layer.id));
  // Kepler derives Deck layer ids from its layer id (`<layerId>-<suffix>`); match the longest owner.
  const keplerLayerIds = props.visState.layers.map(layer => layer.id);
  const filter = createResidentViewLayerFilter(
    plan.views,
    deckLayerId =>
      keplerLayerIds
        .filter(id => deckLayerId === id || deckLayerId.startsWith(`${id}-`))
        .sort((left, right) => right.length - left.length)[0] ?? deckLayerId
  );
  const previousFilter = result['layerFilter'] as
    | ((options: {layer: Layer; viewport: {id: string}}) => boolean)
    | undefined;
  return {
    ...withDeckClass,
    // Each resident layer is rendered once; layerFilter admits it in every view where it is visible.
    layers: orderGlobeDepthDisk(
      [...primaryLayers, ...secondaryLayers.filter(l => !known.has(l.id))],
      result['views']
    ),
    ...(splitMode === 'SWIPE_COMPARE'
      ? getSwipeViews(props, viewStates, baseView)
      : {
          views: [
            baseView.clone({id: 'view-0', x: 0, width: '50%'}),
            baseView.clone({id: 'view-1', x: '50%', width: '50%'})
          ]
        }),
    layerFilter: (options: {layer: Layer; viewport: {id: string}}) =>
      (previousFilter?.(options) ?? true) &&
      filter({layer: {id: getRootLayerId(options.layer)}, viewport: options.viewport}),
    viewState: viewStates
  };
}
