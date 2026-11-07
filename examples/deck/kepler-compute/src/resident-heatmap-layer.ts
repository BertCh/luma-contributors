// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {HeatmapLayer} from '@deck.gl/aggregation-layers';
import {
  COORDINATE_SYSTEM,
  type Layer,
  type LayerContext,
  type UpdateParameters,
  type Viewport
} from '@deck.gl/core';
import {assert, type Device} from '@luma.gl/core';
import {
  createGlobeHeatmapMesh,
  ResidentGlobeHeatmapTriangleLayer,
  type GlobeHeatmapMesh
} from './resident-globe-heatmap-layer';
import {HeatmapWeightRuntime, type ResidentHeatmapInputs} from './heatmap-weight-runtime';

export type {ResidentHeatmapInputs} from './heatmap-weight-runtime';

type ResidentHeatmapExtraProps = {
  residentInputs: ResidentHeatmapInputs;
  globeMode?: boolean;
  densityBounds?: readonly [number, number, number, number] | null;
};

/** Globe framing methods that only Kepler's heatmap subclass defines. */
type KeplerGlobeHeatmap = {_isGlobeHeatmap?(): boolean; _getRadiusScale?(): number};

type ResidentHeatmapFactory = (nativeLayer: HeatmapLayer, inputs: ResidentHeatmapInputs) => Layer;
const residentFactories = new Map<typeof HeatmapLayer, ResidentHeatmapFactory>();
const renderResidentDensity = Symbol('renderResidentDensity');
const readResidentReadiness = Symbol('readResidentReadiness');

export type ResidentHeatmapReadiness = {
  revision: number;
  completedRevision?: number;
  densityPending: boolean;
  pipelinesReady: boolean;
  presentationReady: boolean;
};

/** Run after resident selection and normalized-weight producers, before Deck's display pass. */
export function renderResidentHeatmapPasses(layers: readonly Layer[]): void {
  for (const layer of layers) {
    if (renderResidentDensity in layer && typeof layer[renderResidentDensity] === 'function') {
      layer[renderResidentDensity]();
    }
  }
}

/** Bounded metadata only; this never reads a source buffer or density texture. */
export function getResidentHeatmapReadiness(layer: Layer): ResidentHeatmapReadiness | undefined {
  if (readResidentReadiness in layer && typeof layer[readResidentReadiness] === 'function') {
    return layer[readResidentReadiness]();
  }
  return undefined;
}

/** Native 512² rgba16float textures plus bounded ramp, quad, and per-layer uniforms. */
export const RESIDENT_HEATMAP_WORKING_MEMORY_BYTES = 512 * 512 * 8 + 8 + 65536;

/** An explicit boundary: there is no lower-precision texture fallback. */
export function getResidentHeatmapUnsupportedReason(device: Device): string | undefined {
  if (device.type !== 'webgpu') return 'Resident heatmaps require WebGPU';
  if (
    device.limits.maxTextureDimension2D < 512 ||
    !device.getTextureFormatCapabilities('rgba16float').blend
  )
    return 'Resident heatmaps require blendable rgba16float textures of size 512';
  return undefined;
}

/**
 * Keeps the exact installed Kepler heatmap's viewport framing, maximum reduction and ramp.
 * Only the attribute/weight pass is replaced. Native renderLayer is configuration-only;
 * callers must bypass its CPU formatLayerData/filter path before creating this instance.
 * Call renderResidentHeatmapPasses from every preRender after resident producers encode;
 * updateState intentionally cannot consume inputs before that ordered producer stage.
 */
export function makeResidentHeatmapLayer(nativeLayer: Layer, inputs: ResidentHeatmapInputs): Layer {
  // Pinned Kepler's CJS dependency and the ESM Deck export can have distinct constructor
  // identities. Validate the native heatmap extension seam, then retain its actual constructor.
  assert(
    typeof Reflect.get(nativeLayer, '_createWeightsTransform') === 'function' &&
      typeof Reflect.get(nativeLayer, '_updateWeightmap') === 'function'
  );
  const heatmapLayer = nativeLayer as HeatmapLayer;
  const NativeHeatmap = heatmapLayer.constructor as typeof HeatmapLayer;
  let createResidentLayer = residentFactories.get(NativeHeatmap);
  if (!createResidentLayer) {
    class ResidentGPUHeatmapLayer extends NativeHeatmap<unknown, ResidentHeatmapExtraProps> {
      static override layerName = 'ResidentGPUHeatmapLayer';
      static override defaultProps = {
        ...NativeHeatmap.defaultProps,
        residentInputs: {type: 'object', value: null, compare: false}
      };
      declare state: HeatmapLayer['state'] & {
        residentWeightRuntime?: HeatmapWeightRuntime;
        residentPaletteKey?: string;
        residentDensityPending?: boolean;
        residentRetryScheduled?: boolean;
        residentCompletedRevision?: number;
        residentPresentationTimestamp?: number;
        residentDensityViewport?: Viewport;
        residentCommonBounds?: [number, number, number, number];
        residentRadiusScale?: number;
        residentGlobeMesh?: GlobeHeatmapMesh;
        residentGlobeMeshKey?: string;
      };

      override initializeState(): void {
        const reason = getResidentHeatmapUnsupportedReason(this.context.device);
        if (reason) throw new Error(reason);
        // Allocation and texel semantics are intentionally pinned to native Kepler's 512² grid.
        assert(this.props.weightsTextureSize === 512 && this.props.colorRange.length <= 256);
        super.initializeState();
      }

      override get isLoaded(): boolean {
        if (!super.isLoaded) return false;
        if (!this.props.visible) return true;
        const readiness = this[readResidentReadiness]();
        return !readiness.densityPending && readiness.pipelinesReady && readiness.presentationReady;
      }

      [readResidentReadiness](): ResidentHeatmapReadiness {
        const models = this.getSubLayers().flatMap(layer => layer.getModels());
        const timestamp = this.state?.residentPresentationTimestamp;
        const revision = this.props.residentInputs.revision;
        return {
          revision,
          completedRevision: this.state?.residentCompletedRevision,
          densityPending:
            Boolean(this.state?.residentDensityPending) ||
            this.state?.residentCompletedRevision !== revision,
          pipelinesReady:
            this.state?.residentWeightRuntime?.transform.model.pipeline.isPending === false &&
            this.state?.maxWeightTransform?.model.pipeline.isPending === false &&
            models.length > 0 &&
            models.every(model => !model.pipeline.isPending && !model.pipeline.isErrored),
          presentationReady:
            timestamp !== undefined &&
            models.length > 0 &&
            models.every(model => model._lastDrawTimestamp >= timestamp)
        };
      }

      override _setupAttributes(): void {}
      protected override _updateAttributes(): void {}

      override _createWeightsTransform(): void {
        const previous = this.state.residentWeightRuntime;
        const current = this.props.residentInputs;
        const texture = this.state.weightsTexture;
        if (
          previous &&
          previous.transform.getTargetTexture() === texture &&
          previous.inputs.positions === current.positions &&
          previous.inputs.weights === current.weights &&
          previous.inputs.selectionMask === current.selectionMask &&
          previous.inputs.weightValidity === current.weightValidity
        )
          return;
        previous?.destroy();
        assert(texture);
        const residentWeightRuntime = new HeatmapWeightRuntime(
          this.context.device,
          texture,
          this.props.residentInputs
        );
        this.setState({residentWeightRuntime, weightsTransform: residentWeightRuntime.transform});
      }

      override _getChangeFlags(
        options: UpdateParameters<this>
      ): ReturnType<HeatmapLayer['_getChangeFlags']> {
        const current = options.props.residentInputs;
        const previous = options.oldProps.residentInputs;
        const dataChanged =
          Boolean(this.state.residentDensityPending) ||
          !previous ||
          current.positions !== previous.positions ||
          current.selectionMask !== previous.selectionMask ||
          current.weights !== previous.weights ||
          current.weightValidity !== previous.weightValidity ||
          current.revision !== previous.revision ||
          options.props.radiusPixels !== options.oldProps.radiusPixels ||
          options.props.colorDomain !== options.oldProps.colorDomain ||
          options.props.aggregation !== options.oldProps.aggregation ||
          options.props.densityBounds !== options.oldProps.densityBounds;
        return {
          dataChanged: dataChanged ? 'resident heatmap revision' : false,
          // Native globe framing clears worldBounds when the density frame changes; reframe then.
          viewportChanged: options.changeFlags.viewportChanged || !this.state.worldBounds,
          viewportZoomChanged: this.context.viewport.zoom !== this.state.zoom
        };
      }

      override _updateColorTexture(options: UpdateParameters<this>): void {
        // Native Kepler constructs a fresh bounded ramp on each render; reuse equal palettes.
        const residentPaletteKey = JSON.stringify(options.props.colorRange);
        if (residentPaletteKey === this.state.residentPaletteKey) return;
        super._updateColorTexture(options);
        this.setState({residentPaletteKey});
      }

      override _updateWeightmap(): void {
        const {worldBounds, textureSize} = this.state;
        assert(worldBounds);
        const commonBounds = this._worldToCommonBounds(worldBounds, {
          useLayerCoordinateSystem: true
        });
        const {colorDomain, aggregation} = this.props;
        if (colorDomain && aggregation === 'SUM') {
          const metersPerPixel =
            (this.context.viewport.distanceScales.metersPerUnit[2] *
              (commonBounds[2] - commonBounds[0])) /
            textureSize;
          this.state.colorDomain = [
            colorDomain[0] * metersPerPixel,
            colorDomain[1] * metersPerPixel
          ];
        } else {
          this.state.colorDomain = colorDomain || [0, 0];
        }
        // Native updateState precedes resident producer effects. Do not submit density here:
        // newly allocated masks and normalized weights may not have been encoded yet.
        this.state.residentDensityViewport = this.context.viewport;
        // Native globe framing reuses a smaller data-bounds frame; keep its world radius constant.
        const globe = this as unknown as KeplerGlobeHeatmap;
        this.state.residentRadiusScale = globe._isGlobeHeatmap?.() ? globe._getRadiusScale!() : 1;
        this.state.residentCommonBounds = commonBounds;
        this.state.residentDensityPending = true;
        this.state.isWeightMapDirty = true;
      }

      [renderResidentDensity](): void {
        if (!this.props.visible || !this.state?.residentWeightRuntime) return;
        if (!this.state.residentDensityPending) {
          if (!this.isLoaded) this.setNeedsRedraw();
          return;
        }
        const {residentWeightRuntime, residentDensityViewport, residentCommonBounds} = this.state;
        assert(residentDensityViewport && residentCommonBounds);
        const maximumModel = this.state.maxWeightTransform?.model;
        // Failed pipelines cannot become ready through retries.
        assert(
          !residentWeightRuntime.transform.model.pipeline.isErrored &&
            !maximumModel?.pipeline.isErrored
        );
        if (
          residentWeightRuntime.transform.model.pipeline.isPending ||
          maximumModel?.pipeline.isPending
        ) {
          this.scheduleDensityRetry();
          return;
        }
        const weightsDrawn = residentWeightRuntime.render({
          viewport: residentDensityViewport,
          devicePixelRatio: this.context.device.canvasContext?.cssToDeviceRatio() ?? 1,
          coordinateSystem: this.props.coordinateSystem,
          coordinateOrigin: this.props.coordinateOrigin,
          modelMatrix: this.props.modelMatrix,
          commonBounds: residentCommonBounds,
          radiusPixels:
            this.state.residentRadiusScale && this.state.residentRadiusScale !== 1
              ? Math.max(1, this.props.radiusPixels * this.state.residentRadiusScale)
              : this.props.radiusPixels
        });
        if (!weightsDrawn) {
          this.scheduleDensityRetry();
          return;
        }
        this._updateMaxWeightValue();
        if (maximumModel?.pipeline.isPending || maximumModel?.needsRedraw()) {
          this.scheduleDensityRetry();
          return;
        }
        this.state.residentDensityPending = false;
        this.state.isWeightMapDirty = false;
        this.state.residentCompletedRevision = this.props.residentInputs.revision;
        // A successful density submission is not yet a visible triangle draw. Use the
        // model's draw timestamp to keep native image export pending until that draw occurs.
        this.context.device.incrementTimestamp();
        this.state.residentPresentationTimestamp = this.context.device.timestamp;
        this.setNeedsRedraw();
      }

      /** Native globe mode: the density frame covers the data bounds and drapes over a sphere mesh. */
      override renderLayers(): ReturnType<HeatmapLayer['renderLayers']> {
        if (!(this as unknown as KeplerGlobeHeatmap)._isGlobeHeatmap?.())
          return super.renderLayers();
        const {weightsTexture, maxWeightsTexture, colorTexture, colorDomain, worldBounds} =
          this.state;
        const {residentDensityViewport, residentCommonBounds} = this.state;
        const resolution = this.context.viewport.resolution;
        if (
          !weightsTexture ||
          !maxWeightsTexture ||
          !colorTexture ||
          !worldBounds ||
          !residentDensityViewport ||
          !residentCommonBounds ||
          !resolution
        )
          return [];
        const meshKey = `${worldBounds.join(',')}|${resolution}`;
        if (this.state.residentGlobeMeshKey !== meshKey || !this.state.residentGlobeMesh) {
          const [west, south, east, north] = residentCommonBounds;
          this.state.residentGlobeMesh = createGlobeHeatmapMesh(
            worldBounds as [number, number, number, number],
            (longitude, latitude) => {
              const common = residentDensityViewport.projectPosition([longitude, latitude, 0]);
              return [(common[0] - west) / (east - west), (common[1] - south) / (north - south)];
            },
            resolution
          );
          this.state.residentGlobeMeshKey = meshKey;
        }
        const {updateTriggers, intensity, threshold, aggregation} = this.props;
        return new ResidentGlobeHeatmapTriangleLayer(
          this.getSubLayerProps({id: 'globe-triangle-layer', updateTriggers}),
          {
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            data: {length: 1},
            mesh: this.state.residentGlobeMesh,
            maxTexture: maxWeightsTexture,
            colorTexture,
            aggregationMode: aggregation === 'MEAN' ? 1 : 0,
            weightsTexture,
            intensity,
            threshold,
            colorDomain
          }
        );
      }

      private scheduleDensityRetry(): void {
        this.state.residentDensityPending = true;
        this.state.isWeightMapDirty = true;
        if (this.state.residentRetryScheduled) return;
        this.state.residentRetryScheduled = true;
        // Schedule after the current Deck update clears its flags, using the transferred layer.
        queueMicrotask(() => {
          if (!this.state.residentWeightRuntime) return;
          this.state.residentRetryScheduled = false;
          this.getCurrentLayer()?.setNeedsRedraw();
        });
      }

      override finalizeState(context: LayerContext): void {
        this.state.residentWeightRuntime?.destroy();
        this.state.residentWeightRuntime = undefined;
        this.state.weightsTransform = undefined;
        this.state.maxWeightTransform?.sampler.destroy();
        super.finalizeState(context);
      }
    }
    createResidentLayer = (nativeLayer, inputs) =>
      new ResidentGPUHeatmapLayer(nativeLayer.props, {
        residentInputs: inputs,
        data: [],
        // All source filtering is expressed by the resident selection mask.
        extensions: [],
        pickable: false,
        autoHighlight: false,
        weightsTextureSize: 512,
        debounceTimeout: 0
      });
    residentFactories.set(NativeHeatmap, createResidentLayer);
  }
  return createResidentLayer(heatmapLayer, inputs);
}
