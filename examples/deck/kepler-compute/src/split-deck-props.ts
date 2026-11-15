// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DeckProps, View} from '@deck.gl/core';
import {
  applyExactGlobeController,
  syncTopLevelController,
  withExactGlobeControllers,
  withTouchRotate
} from './globe-wheel-controller';

type DeckEventHandler = (...parameters: never[]) => unknown;
type DeckEventName = 'onViewStateChange' | 'onHover' | 'onClick';
type DeckEventProps = Partial<Record<DeckEventName, DeckEventHandler>>;
export type ResidentSplitDeckProps = DeckProps<View | View[] | null>;

/** Adapt full renders and incremental Deck updates without replacing omitted properties. */
export function buildResidentSplitDeckProps(
  props: ResidentSplitDeckProps,
  viewStates: Record<string, unknown> | null,
  getSecondaryProps: () => ResidentSplitDeckProps | null | undefined
): ResidentSplitDeckProps {
  const result = {...withTouchRotate(props)};
  // Controller normalization may add `views: undefined`. Deck merges partial props, so even
  // that would discard its current views. A layers-only retirement must retain both views.
  if (!('views' in props)) delete result.views;
  if (!viewStates) return withExactGlobeControllers(result);

  result.viewState = viewStates;
  if (result.views) {
    // Deck applies a top-level controller to one view only; every split view needs its own.
    const views = Array.isArray(result.views) ? result.views : [result.views];
    result.views = views.map((view: View) => {
      const controlledView =
        view.props.controller === undefined && result.controller !== undefined
          ? view.clone({controller: result.controller})
          : view;
      return applyExactGlobeController(controlledView, result.controller);
    });
  }

  const route = (name: DeckEventName, viewIdOf: (parameters: never) => unknown) => {
    // Do not replace handlers with closures over missing callbacks on incremental updates.
    if (!(name in props)) return;
    const primaryHandler = (props as DeckEventProps)[name];
    (result as DeckEventProps)[name] = (...parameters: never[]) => {
      const secondaryHandler = (getSecondaryProps() as DeckEventProps | null)?.[name];
      const handler = viewIdOf(parameters[0]) === 'view-1' ? secondaryHandler : primaryHandler;
      return handler?.(...parameters);
    };
  };
  route('onViewStateChange', (parameters: {viewId?: string}) => parameters?.viewId);
  route('onHover', (info: {viewport?: {id: string}}) => info?.viewport?.id);
  route('onClick', (info: {viewport?: {id: string}}) => info?.viewport?.id);
  return syncTopLevelController(result);
}
