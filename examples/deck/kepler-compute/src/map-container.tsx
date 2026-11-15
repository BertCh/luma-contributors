// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getResidentMapSplitMode} from './resident-view-plan';
import {SAMPLE_MAP_BACKGROUND_STYLE} from './sample-map-catalog';

import React, {useState, useEffect, useContext} from 'react';
import {
  MapContainerFactory,
  MapViewStateContext,
  flattenDeps,
  appInjector,
  MapPopoverFactory,
  MapControlFactory,
  LayerManagerFactory,
  LayerConfiguratorFactory,
  LoadDataModalFactory,
  PanelHeaderFactory,
  MapPopoverContentFactory,
  PlotContainerFactory,
  ExportImageModalFactory,
  DataTableModalFactory,
  ModalContainerFactory,
  MapsLayoutFactory,
  LayerLegendContentFactory,
  LayerColorLegendFactory,
  ContainerFactory,
  CustomPanelsFactory
} from '@kepler.gl/components';
import {WorkbenchPanelsFactory} from './workflow/workbench-app';
import {appState} from './app-state';
import {SINGLE_VIEW_REASON} from './resident-action-guards';
import {
  withoutGlobeViewClear,
  orderGlobeDepthDisk,
  orderGlobeBasemapTiles,
  isGlslGlobeLayer,
  toWebGPUGlobeLayer
} from './main-globe-layers';
import {makeResidentEditorLayer} from './resident-editor-layer';
import {makeTile3DPickable} from './tile3d-picking';
import {InterpolationSurfaceTooltip} from './interpolation-surface-tooltip';
import {interpolationSurfaceHover} from './interpolation-surface-hover';
import {createNativeImageExportFactories} from './native-image-export';
import {registerMediaExportHost} from './native-video-export';
import {makeKeplerMediaExportHost} from './native-media-host';
import {
  wrapTo,
  updateMap,
  setLayerAnimationTime,
  removeFilter,
  interactionConfigChange
} from '@kepler.gl/actions';
import {ChartManagerFactory, ChartPanelContentFactory} from '@kepler.gl/components/charts';
import {createResidentChannelDomainReader} from './resident-channel-domain';
import {DATASET_ID} from './sample-data';
import {createResidentSelectGeometryFactory} from './resident-select-geometry-control';
import {ResidentMapControlFactory} from './resident-chart-control';
import {
  NativeLayerManagerFactory,
  NativeLoadDataModalFactory,
  NativePanelHeaderFactory
} from './native-workflows';
import {createFlowConfiguratorFactory} from './native-flow-config-controls';
import {
  createResidentChartPopoverFactory,
  createResidentChartPanelFactory
} from './resident-chart-panel';
import {ExportImageModalTiledFactory} from './native-export-image-tiled';
import {createResidentDataTableModalFactory} from './resident-data-table';
import {AUTO_PREPARE_LIMITS} from './native-auto-prepare';
import {createResidentModalContainerFactory} from './resident-workflow-guards';
import {createResidentMapsLayoutFactory} from './resident-maps-layout';
import {createResidentChannelLegendFactory} from './resident-channel-legend';
import {createResidentStyleLegendFactory} from './resident-style-legend';
import {
  statusListeners,
  vectorTileLive,
  vectorTileResident,
  effect,
  filterWebGPUEffects,
  reportError
} from './app-services';
import {
  VIEWS_ENABLED,
  secondaryViewListeners,
  setSecondaryView,
  makeNativeMapViewport,
  applySecondaryView,
  setSwipePercentage
} from './split-view';
import {initializeDevice, recordSubmittedPerformanceFrame} from './device-init';
import {rasterLiveHost} from './main-raster-live';
import {getResidentAnalysisLayers} from './layers';
import {store} from './app-store';
import {isResidentDataset} from './publication';
import {readSelectedGeometry, readNativeTooltip} from './selection';
import {nativeAutoPrepare} from './native-hosts';

/**
 * A swipe view is narrower than the map, and Deck reports its viewport size in view-state changes;
 * keep those from replacing the native map dimensions (only resize may set them).
 */
function getViewportSizeFilteredActions(
  props: React.ComponentProps<ReturnType<typeof MapContainerFactory>>,
  isSwipe: boolean
) {
  if (!isSwipe) return props.mapStateActions;
  return {
    ...props.mapStateActions,
    updateMap: (
      viewport: Parameters<typeof props.mapStateActions.updateMap>[0],
      mapIndex?: number
    ) => {
      const {width, height, ...viewState} = viewport;
      return props.mapStateActions.updateMap(
        Object.keys(viewState).length > 0 ? viewState : viewport,
        mapIndex
      );
    }
  };
}

/** These callbacks must be injected after Kepler assembles its final Deck properties. */
function ComputeMapContainerFactory(...dependencies: Parameters<typeof MapContainerFactory>) {
  const OriginalMapContainer = MapContainerFactory(...dependencies);
  return function ComputeMapContainer(props: React.ComponentProps<typeof OriginalMapContainer>) {
    const [generation, setGeneration] = useState(appState.deviceGeneration);
    useEffect(() => {
      const updateGeneration = () => setGeneration(appState.deviceGeneration);
      statusListeners.add(updateGeneration);
      return () => {
        statusListeners.delete(updateGeneration);
      };
    }, []);
    const [, setViewRevision] = useState(0);
    const viewStateContext = useContext(MapViewStateContext);
    const viewIndex = props.index ?? 0;
    useEffect(() => {
      if (!VIEWS_ENABLED) return undefined;
      // Only the primary view re-renders on a capture; the capturing view re-rendering itself would
      // produce fresh layers each pass and loop forever.
      const listener = () => setViewRevision(value => value + 1);
      if (viewIndex === 0) secondaryViewListeners.add(listener);
      return () => {
        secondaryViewListeners.delete(listener);
        // Closing view 1 only drops its captured layers; the shared Deck/device is untouched.
        if (viewIndex === 1) {
          appState.secondaryDeck = null;
          setSecondaryView(null);
        }
      };
    }, [viewIndex]);
    if (viewIndex > 0 && (!VIEWS_ENABLED || viewIndex > 1))
      return <p role="status">{SINGLE_VIEW_REASON}</p>;
    if (viewIndex === 1) {
      // Capture-only: the primary Deck draws this map's layers in its second viewport.
      return (
        <OriginalMapContainer
          key={generation}
          {...props}
          mapStateActions={getViewportSizeFilteredActions(
            props,
            props.mapState.mapSplitMode === 'SWIPE_COMPARE'
          )}
          // The primary Deck draws this map's layers. Kepler mounts this map's basemap and controls
          // inside its own (empty) Deck, so keep that Deck alive and sized (DeckGL mounts its basemap
          // children only for a sized, attached canvas) but never paint or hit-test its canvas.
          getMapboxRef={(mapRef, mapIndex) => {
            props.getMapboxRef?.(mapRef, mapIndex);
            // The basemap mounts while its container is still being sized; follow later size changes.
            appState.secondaryMapResizeObserver?.disconnect();
            appState.secondaryMapResizeObserver = null;
            const map = mapRef?.getMap?.();
            if (map && typeof ResizeObserver !== 'undefined') {
              appState.secondaryMapResizeObserver = new ResizeObserver(() => map.resize());
              appState.secondaryMapResizeObserver.observe(map.getContainer());
            }
          }}
          onDeckInitialized={deck => {
            appState.secondaryDeck = deck;
            const canvas = deck.getCanvas();
            if (canvas) Object.assign(canvas.style, {visibility: 'hidden', pointerEvents: 'none'});
          }}
          deckRenderCallbacks={{
            onDeckRender: properties => {
              setSecondaryView({
                layers: properties['layers'],
                viewState: properties['viewState'],
                onViewStateChange: properties['onViewStateChange'] as
                  | ((parameters: unknown) => void)
                  | undefined
              });
              // The hidden Deck must not clear the shared canvas either (globe view clear flags).
              return {
                ...properties,
                layers: [],
                effects: [],
                views: withoutGlobeViewClear(properties['views'], 'secondary')
              };
            }
          }}
        />
      );
    }
    if (props.mapState !== appState.previousNativeMapState) {
      appState.previousNativeMapState = props.mapState;
      appState.nativeMapZoom = props.mapState.zoom;
      // The host's native viewport carries `resolution` in degrees (Kepler uses 5; the Deck view gets 72 intervals through withoutGlobeViewClear), which selects H3's polygon branch.
      appState.nativeMapViewport = makeNativeMapViewport(props.mapState);
    }
    const isDual =
      getResidentMapSplitMode(props.mapState) === 'DUAL_MAP' && Boolean(appState.secondaryView);
    const isSwipe =
      props.mapState.mapSplitMode === 'SWIPE_COMPARE' && Boolean(appState.secondaryView);
    const mapStateActions = getViewportSizeFilteredActions(props, isSwipe);
    const primaryContainer = (
      <OriginalMapContainer
        key={generation}
        {...props}
        mapStateActions={mapStateActions}
        onDeckInitialized={(deck, device) => initializeDevice(deck, device, generation)}
        deckRenderCallbacks={{
          onDeckRender: properties =>
            applySecondaryView(
              props,
              {
                ...properties,
                // Preserve native editing modes with asynchronous WebGPU picks and compatible extensions.
                layers: orderGlobeDepthDisk(
                  orderGlobeBasemapTiles([
                    ...(Array.isArray(properties['layers'])
                      ? properties['layers']
                          .flat(Infinity)
                          .filter(layer => !isGlslGlobeLayer(layer))
                          .map(toWebGPUGlobeLayer)
                          .map(layer =>
                            layer &&
                            typeof layer === 'object' &&
                            layer.id === 'kepler_editor_layer' &&
                            typeof layer.clone === 'function'
                              ? makeResidentEditorLayer(layer)
                              : layer
                          )
                          .map(layer => vectorTileLive.wrapLayer(layer))
                          .map(layer => rasterLiveHost.wrapLayer(layer))
                          .map(layer => makeTile3DPickable(layer))
                      : properties['layers']
                        ? [properties['layers']]
                        : []),
                    ...vectorTileResident.getLayers(),
                    ...(appState.deviceRecovery.status.phase === 'ready'
                      ? getResidentAnalysisLayers()
                      : [])
                  ]),
                  properties['views']
                ),
                views: withoutGlobeViewClear(properties['views']),
                // Kepler's global globe `cull` would back-face cull resident triangles of either winding.
                parameters:
                  properties['parameters'] && 'cull' in (properties['parameters'] as object)
                    ? {}
                    : properties['parameters'],
                effects: [effect, ...filterWebGPUEffects(properties['effects'])]
              },
              [viewStateContext.getInternalViewState(0), viewStateContext.getInternalViewState(1)]
            ),
          onDeckAfterRender: properties => {
            const renderedRuntime = appState.runtime;
            queueMicrotask(() => {
              if (appState.runtime === renderedRuntime) {
                renderedRuntime?.labels.afterRender();
                renderedRuntime?.pointAltitudes.afterRender();
              }
            });
            if (import.meta.env.DEV && appState.runtime?.isReadyForRender)
              recordSubmittedPerformanceFrame(appState.runtime);
            if (appState.deviceRecovery.status.phase === 'ready')
              void appState.runtime?.readSummary().catch(reportError);
            props.deckRenderCallbacks?.onDeckAfterRender?.(properties);
          }
        }}
      />
    );
    return (
      <>
        {VIEWS_ENABLED ? (
          // Stable wrapper under the flag so toggling split never remounts (and finalizes) the Deck.
          // In Dual mode the left wrapper hosts a Deck that spans both halves.
          <div
            style={{
              position: 'relative',
              height: '100%',
              width: isDual ? '200%' : '100%',
              // Draw the shared Deck over the second map's basemap, below its controls.
              zIndex: isDual || isSwipe ? 1 : undefined
            }}
          >
            {primaryContainer}
          </div>
        ) : (
          primaryContainer
        )}
        {(props.mapControls.chart?.active ||
          (!props.isExport && props.visState.charts?.some(chart => chart.pinned !== false))) && (
          <div
            style={{
              position: 'absolute',
              // Sit left of the native map-control column so its buttons stay clickable.
              right: 64,
              top: 48,
              bottom: 20,
              zIndex: 11,
              width: 320,
              display: 'flex'
            }}
          >
            <NativeChartManager
              isExport={props.isExport}
              panelActive={Boolean(props.mapControls.chart?.active)}
              visState={props.visState}
              visStateActions={props.visStateActions}
            />
          </div>
        )}
        <div style={{position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 12}}>
          <InterpolationSurfaceTooltip store={interpolationSurfaceHover} />
        </div>
      </>
    );
  };
}

ComputeMapContainerFactory.deps = MapContainerFactory.deps;

const {NativePlotContainerFactory, NativeImageMapContainerFactory} =
  createNativeImageExportFactories(ComputeMapContainerFactory);

registerMediaExportHost(
  makeKeplerMediaExportHost(store, 'compute', {
    updateViewState: viewState => wrapTo('compute', updateMap(viewState, 0)),
    setAnimationTime: time => wrapTo('compute', setLayerAnimationTime(time))
  })
);

// ChartManager is shipped separately from Container's registered dependency tree.
const chartInjector = flattenDeps([], ChartManagerFactory).reduce(
  (injector, factory) => injector.provide(factory, factory),
  appInjector
);

const channelDomainReader = createResidentChannelDomainReader({
  resolveScope: datasetId =>
    datasetId === DATASET_ID
      ? appState.runtime?.tableBinding
      : appState.runtime?.publications.get(datasetId)?.scopes,
  resolveProvider: datasetId =>
    datasetId === DATASET_ID
      ? appState.runtime?.nativeTable
      : appState.runtime?.publications.get(datasetId)?.consumer.table,
  notify: () => {
    for (const listener of statusListeners) listener();
  }
});

const computeInjector = chartInjector
  .provide(
    MapPopoverFactory,
    createResidentSelectGeometryFactory({
      isResidentDataset,
      readGeometry: readSelectedGeometry,
      getMapIndex: () => {
        const visState = store.getState().keplerGl['compute']?.visState;
        return visState?.clicked?.mapIndex ?? visState?.hoverInfo?.mapIndex ?? 0;
      }
    })
  )
  .provide(MapControlFactory, ResidentMapControlFactory)
  .provide(CustomPanelsFactory, WorkbenchPanelsFactory as never)
  .provide(LayerManagerFactory, NativeLayerManagerFactory)
  .provide(LayerConfiguratorFactory, createFlowConfiguratorFactory())
  .provide(LoadDataModalFactory, NativeLoadDataModalFactory)
  .provide(PanelHeaderFactory, NativePanelHeaderFactory)
  .provide(
    MapPopoverContentFactory,
    createResidentChartPopoverFactory({
      isResidentDataset,
      readTooltip: readNativeTooltip,
      getRevisionKey: datasetId => {
        const published = appState.runtime?.publications.get(datasetId);
        return String(
          datasetId === DATASET_ID
            ? (appState.runtime?.tableBinding.currentRevision ?? 0)
            : (published?.scopes.currentRevision ?? 0)
        );
      }
    })
  )
  .provide(PlotContainerFactory, NativePlotContainerFactory)
  .provide(ExportImageModalFactory, ExportImageModalTiledFactory)
  .provide(MapContainerFactory, NativeImageMapContainerFactory)
  .provide(
    DataTableModalFactory,
    createResidentDataTableModalFactory({
      columnEditHost: {
        getVisState: () => store.getState().keplerGl['compute']?.visState as never,
        removeFilter: index => store.dispatch(wrapTo('compute', removeFilter(index))),
        setTooltipFieldsToShow(datasetId, fieldsToShow) {
          const tooltip =
            store.getState().keplerGl['compute']?.visState?.interactionConfig?.tooltip;
          if (tooltip)
            store.dispatch(
              wrapTo(
                'compute',
                interactionConfigChange({
                  ...tooltip,
                  config: {
                    ...tooltip.config,
                    fieldsToShow: {...tooltip.config.fieldsToShow, [datasetId]: fieldsToShow}
                  }
                } as never)
              )
            );
        }
      },
      nativeColumnEditHost: {
        getDataset: datasetId =>
          store.getState().keplerGl['compute']?.visState?.datasets[datasetId] as never,
        requestRepublish: datasetId => nativeAutoPrepare.requestRepublish(datasetId),
        maximumRows: AUTO_PREPARE_LIMITS.maximumRows
      },
      isResidentDataset,
      resolveProvider: datasetId =>
        datasetId === DATASET_ID
          ? appState.runtime?.nativeTable
          : appState.runtime?.publications.get(datasetId)?.consumer.table
    })
  )
  .provide(ModalContainerFactory, createResidentModalContainerFactory({isResidentDataset}))
  .provide(
    MapsLayoutFactory,
    VIEWS_ENABLED
      ? createResidentMapsLayoutFactory({onSwipePercentageChange: setSwipePercentage})
      : MapsLayoutFactory
  )
  .provide(
    LayerLegendContentFactory,
    createResidentChannelLegendFactory({
      isResidentDataset,
      getChannelDomain: (datasetId, fieldName) =>
        channelDomainReader.getDomain(datasetId, fieldName),
      getChannelDomainFailure: (datasetId, fieldName) =>
        channelDomainReader.getFailure(datasetId, fieldName),
      getAnalysis: datasetId => {
        const analysis = appState.runtime?.analysis;
        return analysis?.active && analysis.datasetId === datasetId
          ? {mode: analysis.settings.mode, summary: analysis.summary}
          : null;
      },
      subscribe: listener => {
        statusListeners.add(listener);
        return () => statusListeners.delete(listener);
      }
    })
  )
  .provide(
    LayerColorLegendFactory,
    createResidentStyleLegendFactory({
      isResidentDataset,
      getLegend: (layerId, channel) => appState.runtime?.styles.getLegend(layerId, channel) ?? null,
      subscribe: listener => {
        statusListeners.add(listener);
        return () => statusListeners.delete(listener);
      }
    })
  )
  .provide(
    ChartPanelContentFactory,
    createResidentChartPanelFactory({
      isResidentDataset,
      resolveProvider: datasetId =>
        datasetId === DATASET_ID
          ? appState.runtime?.getCharts()
          : appState.runtime?.publications.get(datasetId)?.getCharts()
    })
  );

const NativeChartManager = computeInjector.get(ChartManagerFactory);

export const KeplerGl = computeInjector.get(ContainerFactory);

export const mapStyles = [SAMPLE_MAP_BACKGROUND_STYLE];
