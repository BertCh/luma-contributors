// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useSyncExternalStore,
  type ComponentProps
} from 'react';
import {
  MapContainerFactory,
  MapViewStateContext,
  PlotContainerFactory
} from '@kepler.gl/components';
import {areAnyDeckLayersLoading, findMapBounds} from '@kepler.gl/reducers';
import {
  getCenterAndZoomFromBounds,
  getScaleFromImageSize,
  scaleMapStyleByResolution
} from '@kepler.gl/utils';
import {EMPTY_MAPBOX_STYLE, GEOCODER_LAYER_ID} from '@kepler.gl/constants';
import {MapStateActions} from '@kepler.gl/actions';
import {captureCanvas, composeMapImage, validateImageDimensions} from './native-image-capture';

import {clamp} from '@math.gl/core';

type ImageProps = ComponentProps<ReturnType<typeof PlotContainerFactory>>;
type MapProps = ComponentProps<ReturnType<typeof MapContainerFactory>>;
type CaptureRequest = {props: ImageProps};

/** Reuse the host's device and resident buffers. Native PlotContainer would create a second Deck. */
export function createNativeImageExportFactories(HostMapFactory: typeof MapContainerFactory) {
  let request: CaptureRequest | null = null;
  const listeners = new Set<() => void>();
  const subscribe = (listener: () => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  };
  const getRequest = () => request;
  const publish = (next: CaptureRequest | null) => {
    request = next;
    for (const listener of listeners) listener();
  };

  function NativePlotContainerFactory() {
    return function NativePlotContainer(props: ImageProps) {
      const {
        ratio,
        resolution,
        legend,
        charts,
        transparentBackground,
        center,
        imageSize,
        mapFields
      } = props;
      useLayoutEffect(() => {
        const current = {props};
        props.setExportImageSetting({processing: true, error: false, imageDataUri: ''});
        publish(current);
        return () => {
          if (request === current) publish(null);
        };
      }, [
        ratio,
        resolution,
        legend,
        charts,
        transparentBackground,
        center,
        imageSize.imageW,
        imageSize.imageH,
        imageSize.scale,
        mapFields.mapState,
        mapFields.mapStyle,
        mapFields.visState.layers,
        mapFields.visState.layerData,
        mapFields.visState.charts
      ]);
      return null;
    };
  }
  const plotDependencies: typeof PlotContainerFactory.deps = [];
  NativePlotContainerFactory.deps = plotDependencies;

  function NativeImageMapContainerFactory(...dependencies: Parameters<typeof MapContainerFactory>) {
    const HostMap = HostMapFactory(...dependencies);
    return function NativeImageMapContainer(props: MapProps) {
      const current = useSyncExternalStore(subscribe, getRequest, getRequest);
      const outerViewState = useContext(MapViewStateContext);
      const element = useRef<HTMLDivElement>(null);
      const capture = useRef<{
        request: CaptureRequest;
        canvases: Map<HTMLCanvasElement, string>;
        ready: boolean;
        started: boolean;
        after: number;
      } | null>(null);
      const plan = useMemo(() => {
        if (!current) return null;
        try {
          return makeExportMapProps(props, current.props);
        } catch (error) {
          return {error: error instanceof Error ? error : new Error(String(error))};
        }
      }, [current]);
      const exporting = plan && !('error' in plan) ? plan : null;
      const fail = (error: unknown) => {
        if (request !== current || !current) return;
        const message = error instanceof Error ? error.message : String(error);
        current.props.setExportImageError(
          new Error(
            `Map image export failed: ${message}${error instanceof DOMException && error.name === 'SecurityError' ? ' Remote basemap images must allow canvas export through CORS.' : ''}`
          )
        );
        publish(null);
      };
      useLayoutEffect(() => {
        if (!current || !exporting) return;
        capture.current = {
          request: current,
          canvases: new Map(),
          ready: false,
          started: false,
          after: performance.now() + 250
        };
        return () => {
          capture.current = null;
        };
      }, [current, exporting]);
      useEffect(() => {
        if (plan && 'error' in plan) fail(plan.error);
      }, [plan]);
      useEffect(() => {
        if (!current || !exporting) return;
        const timer = window.setTimeout(
          () => fail(new Error('The map did not finish rendering within 30 seconds.')),
          30000
        );
        let frame = 0;
        const check = () => {
          const state = capture.current;
          const source = element.current;
          if (state?.request !== current || !source || state.started) return;
          const canvases = Array.from(source.querySelectorAll('canvas'));
          if (
            state.ready &&
            performance.now() >= state.after &&
            canvases.length &&
            canvases.every(canvas => state.canvases.has(canvas))
          ) {
            state.started = true;
            void composeMapImage(
              source,
              state.canvases,
              exporting.mapState.width,
              exporting.mapState.height,
              () => request === current,
              current.props.escapeXhtmlForWebpack
            )
              .then(image => {
                if (request !== current) return;
                current.props.setExportImageDataUri(image);
                publish(null);
              })
              .catch(fail)
              .finally(() => window.clearTimeout(timer));
            return;
          }
          frame = requestAnimationFrame(check);
        };
        frame = requestAnimationFrame(check);
        return () => {
          window.clearTimeout(timer);
          cancelAnimationFrame(frame);
        };
      }, [current, exporting]);
      const rememberCanvas = (canvas: HTMLCanvasElement) => {
        const state = capture.current;
        if (!state || state.request !== current || state.started || performance.now() < state.after)
          return;
        if (state.canvases.has(canvas)) return;
        try {
          state.canvases.set(canvas, captureCanvas(canvas));
        } catch (error) {
          fail(error);
        }
      };
      const effective = exporting ?? props;
      const viewState = exporting
        ? {getInternalViewState: () => exporting.mapState, setInternalViewState: () => {}}
        : outerViewState;
      return (
        <div
          ref={element}
          className={exporting ? 'native-image-export-map export-map-instance' : undefined}
          style={
            exporting
              ? {
                  position: 'absolute',
                  left: -100000,
                  top: 0,
                  width: exporting.mapState.width,
                  height: exporting.mapState.height,
                  pointerEvents: 'none',
                  overflow: 'hidden'
                }
              : {position: 'relative', width: '100%', height: '100%'}
          }
        >
          {exporting && (
            <style>{`.native-image-export-map .map-control-panel, .native-image-export-map .chart-manager {zoom: ${clamp(exporting.mapState.height / 1080, 0.5, 3)};}`}</style>
          )}
          <MapViewStateContext.Provider value={viewState}>
            <HostMap
              {...effective}
              onMapRender={map => {
                effective.onMapRender?.(map);
                if (map?.isStyleLoaded()) rememberCanvas(map.getCanvas());
                if (exporting && map && !capture.current?.canvases.has(map.getCanvas()))
                  map.triggerRepaint();
              }}
              topMapContainerProps={{
                ...effective.topMapContainerProps,
                onRender: (event: {
                  target: {
                    isStyleLoaded(): boolean;
                    getCanvas(): HTMLCanvasElement;
                    triggerRepaint(): void;
                  };
                }) => {
                  effective.topMapContainerProps?.onRender?.(event);
                  if (event.target.isStyleLoaded()) rememberCanvas(event.target.getCanvas());
                  if (exporting && !capture.current?.canvases.has(event.target.getCanvas()))
                    event.target.triggerRepaint();
                }
              }}
              deckRenderCallbacks={{
                ...effective.deckRenderCallbacks,
                onDeckAfterRender: properties => {
                  effective.deckRenderCallbacks?.onDeckAfterRender?.(properties);
                  const layers = properties['layers'];
                  if (
                    !exporting ||
                    !capture.current ||
                    !Array.isArray(layers) ||
                    areAnyDeckLayersLoading(layers)
                  )
                    return;
                  // Deck calls onAfterRender before AnimationLoop submits WebGPU commands.
                  // A microtask runs after submission, before this presentation task ends.
                  queueMicrotask(() => {
                    if (capture.current?.request !== current) return;
                    capture.current.ready = true;
                    for (const canvas of element.current?.querySelectorAll<HTMLCanvasElement>(
                      'canvas:not(.maplibregl-canvas):not(.mapboxgl-canvas)'
                    ) ?? [])
                      rememberCanvas(canvas);
                  });
                }
              }}
            />
          </MapViewStateContext.Provider>
        </div>
      );
    };
  }
  NativeImageMapContainerFactory.deps = MapContainerFactory.deps;
  return {NativePlotContainerFactory, NativeImageMapContainerFactory};
}

function makeExportMapProps(props: MapProps, image: ImageProps): MapProps {
  const width = image.imageSize.imageW;
  const height = image.imageSize.imageH;
  validateImageDimensions(width, height);
  if (props.mapState.isSplit || (image.splitMaps?.length ?? 0) > 1)
    throw new Error('Image export requires the single active map.');
  const scale =
    image.imageSize.scale ||
    getScaleFromImageSize(width, height, props.mapState.width, props.mapState.height);
  const zoomOffset = Math.log2(scale);
  let mapState = {
    ...props.mapState,
    width,
    height,
    zoom: props.mapState.zoom + zoomOffset,
    zoomOffset
  };
  if (image.center) {
    const layers = props.visState.layers.filter(
      (layer, index) =>
        layer.id !== GEOCODER_LAYER_ID && layer.shouldRenderLayer(props.visState.layerData[index])
    );
    const center = getCenterAndZoomFromBounds(findMapBounds(layers), {width, height});
    if (center)
      mapState = {
        ...mapState,
        longitude: center.center[0],
        latitude: center.center[1],
        zoom: center.zoom + zoomOffset
      };
  }
  return {
    ...props,
    mapState,
    // Native ResizeObserver dispatches even for isExport maps. Return unbound actions
    // during capture so export dimensions cannot change the saved/live viewport.
    mapStateActions: MapStateActions,
    isExport: true,
    activeSidePanel: null,
    transparentBackground: Boolean(image.transparentBackground),
    mapStyle: {
      ...props.mapStyle,
      bottomMapStyle: image.transparentBackground
        ? EMPTY_MAPBOX_STYLE
        : scaleMapStyleByResolution(props.mapStyle.bottomMapStyle, scale),
      topMapStyle: image.transparentBackground
        ? undefined
        : scaleMapStyleByResolution(props.mapStyle.topMapStyle, scale)
    },
    mapControls: {
      mapLegend: {
        ...props.mapControls.mapLegend,
        show: Boolean(image.legend),
        active: Boolean(image.legend)
      },
      chart: {...props.mapControls.chart, show: false, active: Boolean(image.charts)}
    },
    deckGlProps: {...props.deckGlProps, _isExport: true, useDevicePixels: false, _animate: true},
    logoComponent: image.logoComponent
  };
}
