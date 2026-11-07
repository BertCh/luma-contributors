// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {_GlobeView as GlobeView, _GlobeController, _GlobeViewport} from '@deck.gl/core';
import type {View} from '@deck.gl/core';
import {
  getExactGlobeWheelPivot,
  normalizeLongitude,
  type GlobeViewState
} from './globe-wheel-pivot';

type ControllerStateLike = {getViewportProps: () => Record<string, unknown>};

type GlobeControllerClass = typeof _GlobeController;

/**
 * Subclasses a GlobeController so its wheel zoom keeps the geography under the cursor fixed to
 * floating point precision. Deck's native anchoring is damped and ignores the latitude dependent
 * globe scale. The base class keeps its own drag and state behavior (Kepler's controller freezes
 * the view at the limb). When the cursor misses the globe, the view is above Deck's Mercator
 * switch, or the solve fails, the base class result is used unchanged.
 */
function makeExactGlobeController(BaseController: GlobeControllerClass): GlobeControllerClass {
  class ExactGlobeController extends BaseController {
    static readonly isExactGlobeController = true;
    private wheelPixel: [number, number] | null = null;

    protected override _onWheel(
      event: Parameters<GlobeControllerClass['prototype']['handleEvent']>[0]
    ): boolean {
      this.wheelPixel = this.getZoomPosition(this.getCenter(event as never));
      try {
        return super._onWheel(event as never);
      } finally {
        this.wheelPixel = null;
      }
    }

    protected override updateViewport(
      newControllerState: never,
      extraProps: Record<string, unknown> | null = null,
      interactionState: Record<string, unknown> = {}
    ): void {
      const pixel = this.wheelPixel;
      let exactProps = extraProps;
      if (pixel && interactionState['isZooming']) {
        const oldViewState = this.controllerState.getViewportProps() as unknown as GlobeViewState;
        const nativeProps = (newControllerState as ControllerStateLike).getViewportProps();
        const pivot = getExactGlobeWheelPivot({
          makeViewport: viewState => {
            const viewport = this.makeViewport(viewState as never);
            return viewport instanceof _GlobeViewport ? viewport : null;
          },
          oldViewState,
          newViewState: {...oldViewState, ...nativeProps} as unknown as GlobeViewState,
          pixel
        });
        if (pivot) {
          // `around` would re-anchor a smooth transition with Deck's native (inexact) rule.
          const {around: _around, ...transitionProps} = extraProps ?? {};
          // Bearing and pitch are pinned to the old values the solve used; a base controller may
          // otherwise rotate them while it anchors the zoom, which would move the pivot.
          exactProps = {
            ...transitionProps,
            longitude: normalizeLongitude(pivot.longitude),
            latitude: pivot.latitude,
            zoom: pivot.zoom,
            bearing: pivot.bearing,
            pitch: pivot.pitch
          };
        }
      }
      super.updateViewport(newControllerState, exactProps, interactionState);
    }
  }
  return ExactGlobeController;
}

const exactControllers = new Map<GlobeControllerClass, GlobeControllerClass>();

/** Returns the exact-pivot subclass of a globe controller class (cached; idempotent). */
export function getExactGlobeController(
  BaseController: GlobeControllerClass
): GlobeControllerClass {
  if ((BaseController as {isExactGlobeController?: boolean}).isExactGlobeController) {
    return BaseController;
  }
  let controller = exactControllers.get(BaseController);
  if (!controller) {
    controller = makeExactGlobeController(BaseController);
    exactControllers.set(BaseController, controller);
  }
  return controller;
}

const isExactController = (type: unknown): boolean =>
  Boolean((type as {isExactGlobeController?: boolean} | undefined)?.isExactGlobeController);

type ViewWithController = View & {props: {controller?: unknown}};
const exactViews = new Map<string, {source: View; topLevel: unknown; result: View}>();

const isGlobeControllerClass = (type: unknown): type is GlobeControllerClass =>
  type === _GlobeController ||
  (typeof type === 'function' && type.prototype instanceof _GlobeController);

/**
 * Returns the view with the exact-pivot version of its globe controller (its own `type` option, else
 * the view's default controller, such as Kepler's), or the same view when it is not a GlobeView
 * with an interactive globe controller.
 */
export function applyExactGlobeController(view: View, topLevelController?: unknown): View {
  if (!(view instanceof GlobeView)) return view;
  const own = (view as ViewWithController).props.controller;
  const controller = own ?? topLevelController;
  if (controller === false || controller === null) return view;
  let options: Record<string, unknown>;
  let BaseController: unknown = (view as unknown as {ControllerType: unknown}).ControllerType;
  if (controller === undefined || controller === true) options = {};
  else if (isGlobeControllerClass(controller)) {
    options = {};
    BaseController = controller;
  } else if (typeof controller === 'object') {
    const type = (controller as {type?: unknown}).type;
    if (type) BaseController = type;
    options = controller as Record<string, unknown>;
  } else return view;
  if (!isGlobeControllerClass(BaseController)) return view;
  const ExactController = getExactGlobeController(BaseController);
  if (ExactController === BaseController && isExactController((own as {type?: unknown})?.type)) {
    return view;
  }
  const cached = exactViews.get(view.id);
  if (cached && cached.source === view && cached.topLevel === topLevelController) {
    return cached.result;
  }
  const result = view.clone({controller: {...options, type: ExactController}} as never);
  exactViews.set(view.id, {source: view, topLevel: topLevelController, result});
  return result;
}

/**
 * Deck re-clones the first view with the top-level `controller` prop, which would replace the exact
 * controller. When the first view uses it, make the top-level prop agree.
 */
export function syncTopLevelController<Props extends {views?: unknown; controller?: unknown}>(
  props: Props
): Props {
  if (!props.controller) return props;
  const firstView = [props.views].flat()[0] as ViewWithController | undefined;
  const firstController = firstView?.props?.controller as {type?: unknown} | undefined;
  return isExactController(firstController?.type) ? {...props, controller: firstController} : props;
}

/** Gives every GlobeView in Deck props the exact wheel pivot controller. */
export function withExactGlobeControllers<Props extends {views?: unknown; controller?: unknown}>(
  props: Props
): Props {
  if (!props.views) return props;
  const views = Array.isArray(props.views) ? props.views : [props.views];
  const exactViews = views.map(view =>
    view && typeof view === 'object' && 'clone' in view
      ? applyExactGlobeController(view as View, props.controller)
      : view
  );
  if (exactViews.every((view, index) => view === views[index]))
    return syncTopLevelController(props);
  return syncTopLevelController({
    ...props,
    views: Array.isArray(props.views) ? exactViews : exactViews[0]
  });
}

/**
 * Enables Deck 9.4's native two-finger rotate (`touchRotate`, default false) on a view. MapController
 * and GlobeController both rotate the bearing from the pinch angle, so no custom gesture code is
 * needed; `touch-gestures.ts` keeps only the divider math. An explicit `touchRotate` is respected,
 * and views without an interactive controller are returned unchanged.
 */
export function enableTouchRotate(view: View, topLevelController?: unknown): View {
  const own = (view as ViewWithController).props.controller;
  const controller = own ?? topLevelController;
  if (controller === undefined || controller === false || controller === null) return view;
  let options: Record<string, unknown>;
  if (controller === true) options = {};
  else if (typeof controller === 'object') options = controller as Record<string, unknown>;
  else return view;
  if ('touchRotate' in options || 'multiTouchDrag' in options) {
    return own === undefined ? view.clone({controller: options} as never) : view;
  }
  return view.clone({controller: {...options, touchRotate: true}} as never);
}

/** Applies `enableTouchRotate` to the top-level Deck controller and to every view's own controller. */
export function withTouchRotate<Props extends {views?: unknown; controller?: unknown}>(
  props: Props
): Props {
  const controller = props.controller;
  const isOptions = controller === true || (typeof controller === 'object' && controller !== null);
  const options = controller === true ? {} : (controller as Record<string, unknown> | undefined);
  const nextController =
    isOptions && !('touchRotate' in (options ?? {})) && !('multiTouchDrag' in (options ?? {}))
      ? {...options, touchRotate: true}
      : controller;
  const views = props.views;
  const nextViews = !views
    ? views
    : Array.isArray(views)
      ? views.map(view =>
          view && typeof view === 'object' && 'clone' in view
            ? enableTouchRotate(view as View, nextController)
            : view
        )
      : typeof views === 'object' && 'clone' in views
        ? enableTouchRotate(views as View, nextController)
        : views;
  const viewsUnchanged = Array.isArray(views)
    ? (nextViews as unknown[]).every((view, index) => view === views[index])
    : nextViews === views;
  if (nextController === controller && viewsUnchanged) return props;
  return {...props, controller: nextController, views: nextViews};
}
