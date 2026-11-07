// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {isPolygonFootprintLayerType} from './cell-layer-types';
import {assert, type CommandEncoder} from '@luma.gl/core';
import type {GeoJSONTripGPUVectors, PreparedGeometryGPUVectors} from '@luma.gl/engine/arrow';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {KeplerArrowDataset, KeplerComputeSession} from '@luma-contributors/kepler';
import {KeplerResidentStyleCache} from '@luma-contributors/kepler';
import type {KeplerResidentStyle} from '@luma-contributors/kepler';
import {getKeplerResidentStyleKey, type KeplerResidentStyleDomain} from '@luma-contributors/kepler';
import type {KeplerMemoryReservation} from '@luma-contributors/kepler';
import type {GPUVectorEvaluator} from '@luma.gl/graph/gpu-simple-graph';
import {assertExactFloat32IntegerRange} from './integer-style-range';
import {readNativeJenksThresholds} from './native-jenks-reader';
import {computeJenksBreaksInWorker} from './native-jenks-worker-client';
import type {NativeNaturalBreaks} from './native-jenks-breaks';
import {GeometryRuntime} from './geometry-runtime';
import {TripRuntime} from './trip-runtime';
import {
  H3Runtime,
  H3MetadataRuntime,
  type H3GPUVectors,
  type ResidentH3Binding
} from './h3-runtime';
import {prepareH3ViewportMesh, type H3Viewport, type PreparedH3ViewportMesh} from './h3-source';
import {prepareNativeIconGeometry, prepareNativeIconSource} from './icon-source';
import {isIconCoordinateVisible} from './icon-altitude';
import {
  IconRuntime,
  makeGPUVectorsFromNativeIcons,
  type NativeIconGPUVectors,
  type ResidentIconColumns
} from './icon-runtime';
import {GeoJSONPropertyRuntime, type GeoJSONPropertySource} from './geojson-property-runtime';
import {fieldPreparationCoordinator} from './binding3-field-preparation';
import {requestStyleFieldPreparation} from './binding4-style-field-preparation';
import {
  createNativeStylePlan,
  type NativeStyleChannel,
  type NativeStyleLayer,
  type NativeJenksThresholds,
  type NativeStylePlanContext
} from './native-style-plan';
import type {ResidentDatasetBinding} from './resident-layers';
import type {ResidentStyleLegendSnapshot} from './resident-style-legend';

export type NativeStyleResolveOptions = {
  domainMask?: GPUVector<'uint32'>;
  domainRevision?: number;
  /** Explicit native oracle thresholds; no source rows are scanned or read back by this adapter. */
  jenksThresholds?: NativeStylePlanContext['jenksThresholds'];
  zoom?: number;
  zoomOffset?: number;
  geojsonProperties?: NativeStylePlanContext['geojsonProperties'];
  tripProperties?: NativeStylePlanContext['tripProperties'];
  geojsonFixedRadius?: boolean;
  geometry?: PreparedGeometryGPUVectors;
  geometryColumn?: string;
  trips?: GeoJSONTripGPUVectors;
  tripColumn?: string;
  h3?: H3GPUVectors;
  h3Column?: string;
  h3MetadataMask?: GPUVector<'uint32'>;
  h3MetadataRevision?: number;
  h3Viewport?: H3Viewport & {resolution?: number};
  /** Explicit source-aligned XYZ upload when native Icon uses an altitude column. */
  iconPositions?: {
    positions: GPUVector<'float32x2' | 'float32x3'>;
    columns: {lat: string; lng: string; altitude?: string};
  };
  /** Explicit upload dictionary metadata, including fixtures whose uint32 codes were not Arrow dictionaries. */
  dictionaries?: Readonly<Record<string, readonly (string | number)[]>>;
};
type StyleEntry = {
  style: KeplerResidentStyle;
  key: string;
  legend?: NonNullable<ResidentStyleLegendSnapshot['legend']>;
  pending?: Promise<void>;
  readRevision?: number;
  reason?: string;
};
type GeometryEntry = {
  datasetId: string;
  runtime: GeometryRuntime;
  inputs: readonly unknown[];
  reservation: KeplerMemoryReservation;
};
type TripEntry = {
  datasetId: string;
  runtime: TripRuntime;
  inputs: readonly unknown[];
};
type PropertyEntry = {
  datasetId: string;
  runtime: GeoJSONPropertyRuntime<'float32' | 'unorm8x4'>;
  reservation: KeplerMemoryReservation;
};
type IconEntry = {
  datasetId: string;
  sourceInputs: readonly unknown[];
  retirements: Set<Promise<void>>;
  icons?: NativeIconGPUVectors;
  sourceReason?: string;
  runtime?: IconRuntime;
  runtimeInputs?: readonly unknown[];
};
type H3Entry = {
  datasetId: string;
  source: H3GPUVectors;
  mask: GPUVector<'uint32'>;
  metadata: H3MetadataRuntime;
  metadataPending?: Promise<void>;
  runtime?: H3Runtime;
  inputs?: readonly unknown[];
  viewportMesh?: PreparedH3ViewportMesh;
};
type JenksEntry = {
  key: string;
  request: Omit<Parameters<typeof readNativeJenksThresholds>[0], 'reserveMemory' | 'computeBreaks'>;
  controls?: NativeJenksThresholds;
  pending?: Promise<void>;
  reason?: string;
};
/** Float32 widening of a prepared Int32/Uint32 numeric column, owned by one layer entry. */
type WidenedColumnEntry = {
  evaluator?: GPUVectorEvaluator;
  reservation: KeplerMemoryReservation;
  vector?: GPUVector<'float32'>;
  reason?: string;
};
type LayerEntry = {
  /** Keyed by column name; prepared Int8/16 and Uint8/16 sources arrive here as Int32/Uint32. */
  widened: Map<string, WidenedColumnEntry>;
  datasetId: string;
  sourceRevision: number;
  mapped: Set<NativeStyleChannel>;
  channels: Map<NativeStyleChannel, StyleEntry>;
  properties: Map<NativeStyleChannel, PropertyEntry>;
  /** Derived native Jenks thresholds, keyed by source, field, classes and selection revision. */
  jenks: Map<NativeStyleChannel, JenksEntry>;
  colorOverride?: boolean;
  reason?: string;
  geometry?: GeometryEntry;
  trip?: TripEntry;
  icon?: IconEntry;
  h3?: H3Entry;
};
export type NativeStyleRuntimeProps = {
  session: KeplerComputeSession;
  requestUpdate(): void;
  onUnsupported?: (layerId: string, reason: string | null) => void;
  workingMemoryBudgetBytesPerStyle?: number;
  /** Pinned native Jenks implementation; defaults to the installed Kepler utility. */
  naturalBreaks?: (values: number[], classCount: number) => number[];
};

/** Native plans and resident outputs share revisions. Icon grouping reads retained Arrow once
 * per source/catalog change; subsequent interaction gathers resident channels on the GPU.
 */
export class NativeStyleRuntime {
  private readonly cache: KeplerResidentStyleCache;
  private readonly layers = new Map<string, LayerEntry>();
  private readonly retirements = new Set<Promise<void>>();
  private readonly datasetRetirements = new Map<string, Set<Promise<void>>>();
  private readonly diagnostics = new Map<string, string | null>();
  private readonly listeners = new Set<() => void>();
  private destroyed = false;
  private updateQueued = false;
  private destruction?: Promise<void>;

  constructor(private readonly props: NativeStyleRuntimeProps) {
    this.cache = new KeplerResidentStyleCache({
      session: props.session,
      workingMemoryBudgetBytesPerStyle: props.workingMemoryBudgetBytesPerStyle ?? 32 * 1024 * 1024,
      scaleWorkingMemoryToRows: true,
      maximumDomainCount: 256,
      maximumPaletteCount: 256,
      maximumLegendBytes: 1028
    });
  }

  resolve(
    layer: NativeStyleLayer,
    binding: ResidentDatasetBinding,
    options: NativeStyleResolveOptions = {}
  ): ResidentDatasetBinding {
    // Flow and Flow Field layers draw through their own runtimes and have no style channels.
    // The 3D scenegraph layer takes its look from the glTF model, not style channels.
    if (layer.type === 'flow' || layer.type === 'flowField' || (layer.type as string) === '3D')
      return binding;
    const datasetId = layer.config.dataId;
    const dataset = datasetId
      ? this.props.session.registry.getDataset<KeplerArrowDataset>(datasetId)
      : undefined;
    if (this.destroyed || !dataset)
      return this.unsupported(layer.id, binding, 'Resident style source is unavailable.');
    let entry = this.layers.get(layer.id);
    if (!entry || entry.datasetId !== dataset.id || entry.sourceRevision !== dataset.revision) {
      if (entry) this.removeLayer(layer.id, this.createCompletion());
      entry = {
        datasetId: dataset.id,
        sourceRevision: dataset.revision,
        mapped: new Set(),
        channels: new Map(),
        properties: new Map(),
        jenks: new Map(),
        widened: new Map()
      };
      this.layers.set(layer.id, entry);
    }
    entry.mapped = new Set<NativeStyleChannel>([
      ...(layer.config.colorField ? ['fillColor' as const] : []),
      ...(layer.config.strokeColorField ? ['strokeColor' as const] : []),
      ...(layer.config.sizeField ? ['size' as const] : []),
      ...(layer.config.radiusField ? ['pointSize' as const] : []),
      ...(layer.config.coverageField ? ['coverage' as const] : []),
      ...(layer.config.heightField ? ['height' as const] : []),
      ...(layer.config.elevationOffsetField ? ['elevationOffset' as const] : [])
    ]);
    entry.colorOverride = Boolean(binding.colorOverride);
    const columns: Record<string, NativeStylePlanContext['columns'][string]> = Object.fromEntries(
      Object.entries(dataset.value.columns)
        .filter(([, vector]) => Boolean(vector.format))
        .map(([name, vector]) => [
          name,
          {
            format: vector.format!,
            dictionary:
              options.dictionaries?.[name] ??
              dataset.value.analytics.dictionaries[name]?.values ??
              dataset.value.prepared.dictionaries[name]?.values
          }
        ])
    );
    try {
      if (
        options.domainMask &&
        (!Number.isSafeInteger(options.domainRevision) || (options.domainRevision ?? 0) < 1)
      )
        throw new Error('Selected style domains require a positive semantic revision.');
      // Prepared Int32/Uint32 numeric columns (including widened Int8/16, Uint8/16) are converted
      // once on the GPU to a Float32 style source after a bounded GPU min/max check rejects any
      // column outside the exact ±2^24 integer range; Int64/Uint64 are rejected earlier by Arrow
      // preparation and Float64 requires its explicit preparation policy.
      let widenedPending = false;
      // Native Trips roll, pitch and yaw datum fields are numeric marker orientation sources.
      const tripConfig = layer.config as {
        rollField?: {name: string} | null;
        pitchField?: {name: string} | null;
        yawField?: {name: string} | null;
      };
      const tripOrientationFields =
        layer.type === 'trip'
          ? [tripConfig.rollField, tripConfig.pitchField, tripConfig.yawField]
          : [];
      for (const [name, widenedEntry] of entry.widened)
        if (columns[name]?.format !== 'sint32' && columns[name]?.format !== 'uint32') {
          entry.widened.delete(name);
          this.retireWidened(widenedEntry, this.createCompletion());
        }
      for (const field of [
        layer.config.colorField,
        layer.config.strokeColorField,
        layer.config.sizeField,
        layer.config.radiusField,
        layer.config.coverageField,
        layer.config.heightField,
        layer.config.elevationOffsetField,
        ...tripOrientationFields
      ]) {
        const column = field ? columns[field.name] : undefined;
        const vector = field ? dataset.value.columns[field.name] : undefined;
        if (
          !field ||
          !vector ||
          (column?.format !== 'sint32' && column?.format !== 'uint32') ||
          (column.format === 'uint32' && column.dictionary) ||
          columns[field.name].format === 'float32'
        )
          continue;
        let widenedEntry = entry.widened.get(field.name);
        if (!widenedEntry) {
          const reservation = this.props.session.registry.reserveMemory(
            dataset.rowCount * 4,
            'gpu',
            'scratch'
          );
          widenedEntry = {reservation};
          entry.widened.set(field.name, widenedEntry);
          const created = widenedEntry;
          // Lazy import keeps the gpgpu operation graph out of unrelated style module loads.
          import('@luma.gl/graph/gpu-simple-graph')
            .then(async ({castData, GPUVectorEvaluator}) => {
              await assertExactFloat32IntegerRange(
                this.props.session.device,
                vector as GPUVector<'sint32'> | GPUVector<'uint32'>
              );
              const evaluator = GPUVectorEvaluator.fromGPUVector(vector).mapGPUData(chunk =>
                castData(chunk, {
                  inputFormat: column.format as 'sint32' | 'uint32',
                  outputFormat: 'float32'
                })
              );
              created.evaluator = evaluator;
              const result = await evaluator.evaluate(this.props.session.device);
              if (this.destroyed || entry.widened.get(field.name) !== created) {
                evaluator.destroy();
                return;
              }
              created.vector = result as GPUVector<'float32'>;
            })
            .catch(error => {
              created.reason = error instanceof Error ? error.message : String(error);
            })
            .finally(() => {
              if (!this.destroyed && this.layers.get(layer.id) === entry) this.requestUpdate();
            });
        }
        if (widenedEntry.reason) throw new Error(`${field.name}: ${widenedEntry.reason}`);
        if (widenedEntry.vector) columns[field.name] = {format: 'float32'};
        else widenedPending = true;
      }
      if (widenedPending) {
        entry.reason = 'Integer style columns are pending Float32 widening.';
        this.report(layer.id, null);
        return {...binding, enabled: false, unsupportedReason: undefined};
      }
      const domains: Partial<Record<NativeStyleChannel, KeplerResidentStyleDomain>> = {};
      const fields = {
        fillColor: layer.config.colorField,
        targetColor: layer.config.colorField,
        strokeColor: layer.config.strokeColorField,
        size: layer.config.sizeField,
        pointSize: layer.config.radiusField,
        coverage: layer.config.coverageField,
        height: layer.config.heightField,
        elevationOffset: layer.config.elevationOffsetField
      };
      for (const [channel, field] of Object.entries(fields)) {
        if (!field || columns[field.name]?.format !== 'float32') continue;
        const scale =
          channel === 'height'
            ? layer.config.heightScale
            : channel === 'elevationOffset'
              ? layer.config.elevationOffsetScale
              : channel === 'size'
                ? layer.config.sizeScale
                : channel === 'coverage'
                  ? layer.config.coverageScale
                  : channel === 'pointSize'
                    ? layer.config.radiusScale
                    : channel === 'strokeColor'
                      ? layer.config.strokeColorScale
                      : layer.config.colorScale;
        // Explicit native color thresholds own their domain; display/domain masks must not
        // turn fixed map keys into a new source distribution or quantile computation.
        if (
          (channel === 'fillColor' || channel === 'strokeColor' || channel === 'targetColor') &&
          (scale === 'custom' || scale === 'threshold')
        )
          continue;
        domains[channel as NativeStyleChannel] = options.domainMask
          ? {kind: 'selection', revision: options.domainRevision!}
          : {kind: 'source'};
      }
      const properties = options.geometry
        ? {
            fillColor: options.geometry.metadata.sourceStyleProperties.includes('fillColor')
              ? undefined
              : ('absent' as const),
            lineColor: options.geometry.metadata.sourceStyleProperties.includes('lineColor')
              ? undefined
              : ('absent' as const),
            lineWidth: options.geometry.metadata.sourceStyleProperties.includes('lineWidth')
              ? undefined
              : ('absent' as const),
            radius: options.geometry.metadata.sourceStyleProperties.includes('radius')
              ? undefined
              : ('absent' as const),
            elevation: options.geometry.metadata.sourceStyleProperties.includes('elevation')
              ? undefined
              : ('absent' as const),
            elevationOffset: options.geometry.metadata.sourceStyleProperties.includes(
              'elevationOffset'
            )
              ? undefined
              : ('absent' as const),
            ...options.geometry.properties,
            ...options.geojsonProperties
          }
        : options.geojsonProperties;
      const tripProperties = options.trips
        ? {
            lineColor: options.trips.source.sourceStyleProperties.includes('lineColor')
              ? undefined
              : ('absent' as const),
            lineWidth: options.trips.source.sourceStyleProperties.includes('lineWidth')
              ? undefined
              : ('absent' as const),
            ...options.trips.properties,
            ...options.tripProperties
          }
        : options.tripProperties;
      let h3Branch: 'polygons' | 'columns' | undefined;
      if (layer.type === 'hexagonId') {
        const source = options.h3;
        const mask = options.h3MetadataMask;
        if (!source || !mask || !options.h3Column || !options.h3MetadataRevision)
          throw new Error('H3 requires uploaded geometry and an explicit native topology scope.');
        if (!entry.h3 || entry.h3.source !== source || entry.h3.mask !== mask) {
          const previous = entry.h3;
          entry.h3 = {
            datasetId: dataset.id,
            source,
            mask,
            metadata: new H3MetadataRuntime({
              device: this.props.session.device,
              source,
              mask,
              revision: options.h3MetadataRevision,
              workingMemoryBudgetBytes: Math.max(
                1024,
                dataset.rowCount * 4 + source.validity.data.length * 256 + 1024
              ),
              memoryRegistry: this.props.session.registry
            })
          };
          if (previous) this.retireH3(previous, this.createCompletion());
          this.requestUpdate();
        }
        entry.h3.metadata.setRevision(options.h3MetadataRevision);
        const metadata = entry.h3.metadata.snapshot;
        if (!metadata) {
          // Pending topology is a normal asynchronous state, including hidden layers after replay.
          // Keep its metadata producer active, suppress stale drawing, and clear prior diagnostics.
          entry.reason = 'H3 topology is pending the current submitted selection revision.';
          this.report(layer.id, null);
          return {...binding, enabled: false, unsupportedReason: undefined};
        }
        h3Branch =
          !metadata.count ||
          options.h3Viewport?.resolution ||
          metadata.hasPentagon ||
          metadata.minimumResolution !== metadata.maximumResolution ||
          metadata.minimumResolution <= 5
            ? 'polygons'
            : 'columns';
        if (h3Branch === 'columns') {
          if (!options.h3Viewport)
            throw new Error('Native H3 columns require the current map viewport.');
          entry.h3.viewportMesh = prepareH3ViewportMesh({
            resolution: metadata.minimumResolution,
            viewport: options.h3Viewport,
            previous: entry.h3.viewportMesh
          });
        }
      } else if (entry.h3) {
        this.retireH3(entry.h3, this.createCompletion());
        entry.h3 = undefined;
      }
      const jenksThresholds: NonNullable<NativeStylePlanContext['jenksThresholds']> = {
        ...options.jenksThresholds
      };
      let jenksPending = false;
      for (const channel of ['fillColor', 'strokeColor', 'targetColor'] as const) {
        const stroke = channel === 'strokeColor';
        const field = stroke ? layer.config.strokeColorField : layer.config.colorField;
        const palette = stroke
          ? layer.config.visConfig.strokeColorRange
          : layer.config.visConfig.colorRange;
        const scale = stroke ? layer.config.strokeColorScale : layer.config.colorScale;
        const vector = field
          ? (entry.widened.get(field.name)?.vector ?? dataset.value.columns[field.name])
          : undefined;
        if (
          scale !== 'jenks' ||
          !field ||
          !palette ||
          vector?.format !== 'float32' ||
          jenksThresholds[channel] ||
          (channel === 'targetColor' && layer.type !== 'arc' && layer.type !== 'line')
        ) {
          entry.jenks.delete(channel);
          continue;
        }
        const selectionRevision = options.domainMask ? options.domainRevision : undefined;
        const request = {
          source: {datasetId: dataset.id, revision: dataset.revision},
          column: field.name,
          classCount: palette.colors.length,
          values: vector as GPUVector<'float32'>,
          validity: dataset.value.analytics.validity[field.name],
          mask: options.domainMask,
          selectionRevision
        };
        const key = JSON.stringify([
          request.source,
          request.column,
          request.classCount,
          selectionRevision
        ]);
        let derived = entry.jenks.get(channel);
        if (!derived || derived.key !== key) {
          derived = {key, request};
          entry.jenks.set(channel, derived);
        } else derived.request = request;
        if (derived.reason) throw new Error(`${channel}: ${derived.reason}`);
        if (derived.controls) jenksThresholds[channel] = derived.controls;
        else jenksPending = true;
      }
      if (jenksPending) {
        // Thresholds are derived after the submitted selection revision exists; keep drawing off.
        entry.reason = 'Native Jenks thresholds are pending the current submitted revision.';
        this.report(layer.id, null);
        return {...binding, enabled: false, unsupportedReason: undefined};
      }
      const translated = createNativeStylePlan(layer, {
        source: {datasetId: dataset.id, revision: dataset.revision},
        columns,
        domains,
        jenksThresholds,
        zoom: options.zoom ?? 0,
        zoomOffset: options.zoomOffset,
        geojsonPointFeatures: Boolean(
          options.geometry?.metadata.featureTypes.point || binding.geojson?.points
        ),
        geojsonPolygonFeatures: Boolean(
          options.geometry?.metadata.featureTypes.polygon || binding.geojson?.polygons
        ),
        geojsonFixedRadius:
          options.geojsonFixedRadius ??
          options.geometry?.metadata.fixedRadius ??
          binding.geojson?.fixedRadius,
        geojsonProperties: properties,
        tripProperties,
        h3Branch
      });
      if (!translated.supported) {
        // A scale, domain or channel field the resident copy lacks is prepared on demand.
        const preparation = requestStyleFieldPreparation(
          dataset.id,
          layer.config,
          dataset.value.columns,
          fieldPreparationCoordinator
        );
        return this.unsupported(
          layer.id,
          binding,
          [...translated.reasons, ...preparation].join(' ')
        );
      }
      const colors: Partial<Record<NativeStyleChannel, GPUVector<'unorm8x4'>>> = {};
      const sizes: Partial<Record<NativeStyleChannel, GPUVector<'float32'>>> = {};
      const constants: Partial<Record<NativeStyleChannel, number>> = {};
      const retained = new Set<NativeStyleChannel>();
      const retainedProperties = new Set<NativeStyleChannel>();
      for (const channel of Object.keys(translated.plan.channels) as NativeStyleChannel[]) {
        const channelPlan = translated.plan.channels[channel]!;
        if (channelPlan.kind === 'constant') {
          if (typeof channelPlan.value === 'number') constants[channel] = channelPlan.value;
          continue;
        }
        if (channelPlan.kind === 'property') {
          // Analysis fill is already source aligned and takes precedence over embedded properties.
          if (channel === 'fillColor' && binding.colorOverride) continue;
          const diagnostics =
            layer.type === 'trip'
              ? options.trips?.source.invalidStyleProperties
              : options.geometry?.metadata.invalidStyleProperties;
          const diagnostic = diagnostics?.find(value => value.property === channelPlan.property);
          if (diagnostic)
            throw new Error(`${layer.type} ${channelPlan.property}: ${diagnostic.reason}`);
          retainedProperties.add(channel);
          const source = channelPlan.source;
          let previous = entry.properties.get(channel);
          if (
            !previous ||
            previous.runtime.source.values !== source.values ||
            previous.runtime.source.validity !== source.validity
          ) {
            const replacement = this.createProperty(dataset.id, source, channelPlan.fallback);
            entry.properties.set(channel, replacement);
            this.requestUpdate();
            if (previous) this.retireProperty(previous, this.createCompletion());
            previous = replacement;
          }
          if (previous.runtime.setFallback(channelPlan.fallback)) this.requestUpdate();
          if (previous.runtime.colors) colors[channel] = previous.runtime.colors;
          if (previous.runtime.values) sizes[channel] = previous.runtime.values;
          continue;
        }
        retained.add(channel);
        const key = getKeplerResidentStyleKey(channelPlan.plan);
        const previous = entry.channels.get(channel);
        if (
          !previous ||
          previous.key !== key ||
          !previous.style.canUpdate(
            channelPlan.plan,
            options.domainMask,
            entry.widened.get(channelPlan.plan.column)?.vector
          )
        ) {
          const style = this.cache.set(channelPlan.plan, {
            selectionMask: options.domainMask,
            sourceOverride: entry.widened.get(channelPlan.plan.column)?.vector,
            retireAfter: previous ? this.createCompletion() : undefined
          });
          entry.channels.set(channel, {style, key});
          this.requestUpdate();
        }
        const style = entry.channels.get(channel)!.style;
        if (!style.isCurrent) throw new Error('Resident style source revision changed.');
        if (style.colors) colors[channel] = style.colors;
        if (style.values) sizes[channel] = style.values;
      }
      for (const [channel, previous] of entry.channels) {
        if (retained.has(channel)) continue;
        entry.channels.delete(channel);
        this.trackDatasetRetirement(
          dataset.id,
          this.cache.remove(previous.style.currentPlan.id, this.createCompletion())
        );
      }
      for (const [channel, previous] of entry.properties) {
        if (retainedProperties.has(channel)) continue;
        entry.properties.delete(channel);
        this.retireProperty(previous, this.createCompletion());
      }
      let geojson = binding.geojson;
      if (isPolygonFootprintLayerType(layer.type)) {
        const geometry = options.geometry;
        const column = options.geometryColumn ?? binding.geojson?.column;
        if (!geometry || !column)
          throw new Error('Prepared GeoJSON geometry is required for resident style expansion.');
        const fill = binding.colorOverride ?? colors.fillColor;
        const inputs = [
          geometry,
          column,
          binding.selectionMask,
          fill,
          colors.strokeColor,
          sizes.size,
          sizes.pointSize,
          sizes.height,
          sizes.elevationOffset
        ];
        if (
          !entry.geometry ||
          inputs.some((input, index) => input !== entry.geometry!.inputs[index])
        ) {
          const byteCapacity =
            geometry.buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0) * 4 +
            dataset.rowCount * 32 +
            4 * 1024 * 1024;
          const reservation = this.props.session.registry.reserveMemory(
            byteCapacity,
            'gpu',
            'scratch'
          );
          let runtime: GeometryRuntime | undefined;
          try {
            runtime = new GeometryRuntime({
              device: this.props.session.device,
              column,
              geometry,
              selectionMask: binding.selectionMask,
              colors: fill,
              strokeColors: colors.strokeColor,
              sizes: sizes.size,
              pointSizes: sizes.pointSize,
              elevations: sizes.height ?? constants.height,
              elevationOffsets: sizes.elevationOffset ?? constants.elevationOffset,
              stylesResolved: true
            });
            if (runtime.reason) throw new Error(runtime.reason);
            runtime.prepare();
            if (runtime.accountedResidentByteLength > byteCapacity)
              throw new Error('Geometry style expansion exceeds its reserved memory.');
          } catch (error) {
            runtime?.destroy();
            reservation.release();
            throw error;
          }
          const previous = entry.geometry;
          entry.geometry = {datasetId: dataset.id, runtime, inputs, reservation};
          if (previous) this.retireGeometry(previous, this.createCompletion());
          this.requestUpdate();
        }
        entry.geometry.runtime.setVerticalDefaults({
          elevation: constants.height,
          elevationOffset: constants.elevationOffset
        });
        geojson = {
          ...entry.geometry.runtime.binding,
          fixedRadius:
            options.geojsonFixedRadius ??
            geometry.metadata.fixedRadius ??
            binding.geojson?.fixedRadius
        };
      } else if (entry.geometry) {
        this.retireGeometry(entry.geometry, this.createCompletion());
        entry.geometry = undefined;
      }
      let trip = binding.trip;
      if (layer.type === 'trip') {
        const trips = options.trips;
        const column = options.tripColumn ?? binding.trip?.column;
        if (!trips || !column)
          throw new Error('Prepared trip geometry is required for resident style expansion.');
        const fill = binding.colorOverride ?? colors.fillColor;
        const orientationVector = (field?: {name: string} | null) => {
          const vector = field
            ? (entry.widened.get(field.name)?.vector ?? dataset.value.columns[field.name])
            : undefined;
          return vector?.format === 'float32' ? (vector as GPUVector<'float32'>) : undefined;
        };
        const orientation = {
          roll: orientationVector(tripConfig.rollField),
          pitch: orientationVector(tripConfig.pitchField),
          yaw: orientationVector(tripConfig.yawField)
        };
        const inputs = [
          trips,
          column,
          binding.selectionMask,
          fill,
          sizes.size,
          orientation.roll,
          orientation.pitch,
          orientation.yaw
        ];
        if (!entry.trip || inputs.some((input, index) => input !== entry.trip!.inputs[index])) {
          let runtime: TripRuntime | undefined;
          try {
            runtime = new TripRuntime({
              device: this.props.session.device,
              column,
              trips,
              grouping: binding.trip?.grouping,
              selectionMask: binding.selectionMask,
              colors: fill,
              widths: sizes.size,
              orientation,
              workingMemoryBudgetBytes:
                this.props.workingMemoryBudgetBytesPerStyle ?? 32 * 1024 * 1024,
              memoryRegistry: this.props.session.registry,
              growWorkingMemoryToRegistry: true
            });
            if (runtime.reason) throw new Error(runtime.reason);
            runtime.prepare();
          } catch (error) {
            if (runtime) this.track(runtime.destroy(Promise.resolve()));
            throw error;
          }
          const previous = entry.trip;
          entry.trip = {datasetId: dataset.id, runtime, inputs};
          if (previous) this.retireTrip(previous, this.createCompletion());
          this.requestUpdate();
        }
        const missing = (
          [
            ['roll', tripConfig.rollField, orientation.roll],
            ['pitch', tripConfig.pitchField, orientation.pitch],
            ['yaw', tripConfig.yawField, orientation.yaw]
          ] as const
        ).find(([, field, vector]) => field && !vector);
        trip = entry.trip.runtime.binding;
        // Kept on the stable runtime binding so the layer sees the reason without churn.
        trip.markerChannelReason = missing
          ? `Trip ${missing[0]} field ${missing[1]?.name} must be a published numeric column.`
          : undefined;
      } else if (entry.trip) {
        this.retireTrip(entry.trip, this.createCompletion());
        entry.trip = undefined;
      }
      let icon = binding.icon;
      if (layer.type === 'icon') {
        icon = this.resolveIcon(
          layer,
          entry,
          dataset.value,
          binding,
          options,
          binding.colorOverride ?? colors.fillColor,
          sizes.size
        );
      } else if (entry.icon) {
        this.retireIcon(entry.icon, this.createCompletion());
        entry.icon = undefined;
      }
      let h3: ResidentH3Binding | undefined;
      if (entry.h3?.metadata.snapshot?.count === 0 && options.h3Column) {
        if (entry.h3.runtime) {
          this.trackDatasetRetirement(
            dataset.id,
            entry.h3.runtime.destroy(this.createCompletion())
          );
          entry.h3.runtime = undefined;
        }
        h3 = {column: options.h3Column, branch: 'empty'};
      } else if (entry.h3 && h3Branch && options.h3Column) {
        const current = entry.h3;
        const fill = binding.colorOverride ?? colors.fillColor;
        const inputs = [
          current.source,
          options.h3Column,
          h3Branch,
          binding.selectionMask,
          fill,
          colors.strokeColor,
          sizes.size,
          sizes.coverage,
          h3Branch === 'columns' ? 6 : 0
        ];
        if (!current.runtime || inputs.some((input, index) => input !== current.inputs?.[index])) {
          const replacement = new H3Runtime({
            device: this.props.session.device,
            column: options.h3Column,
            source: current.source,
            branch: h3Branch,
            selectionMask: binding.selectionMask,
            colors: fill,
            strokeColors: colors.strokeColor,
            elevations: sizes.size,
            coverages: sizes.coverage,
            coverage: translated.plan.render.coverage ?? 1,
            viewportMesh: current.viewportMesh,
            workingMemoryBudgetBytes:
              this.props.workingMemoryBudgetBytesPerStyle ?? 32 * 1024 * 1024,
            memoryRegistry: this.props.session.registry
          });
          const previous = current.runtime;
          current.runtime = replacement;
          current.inputs = inputs;
          if (previous)
            this.trackDatasetRetirement(dataset.id, previous.destroy(this.createCompletion()));
          this.requestUpdate();
        }
        current.runtime.setCoverage(translated.plan.render.coverage ?? 1);
        if (h3Branch === 'columns' && current.viewportMesh)
          current.runtime.setViewportMesh(current.viewportMesh);
        h3 = current.runtime.binding;
      }
      entry.reason = undefined;
      const domainReason = this.getDomainDiagnostic(entry);
      this.report(layer.id, domainReason);
      return {
        ...binding,
        ...(domainReason ? {enabled: false} : {}),
        unsupportedReason: domainReason ?? undefined,
        colorOverride:
          isPolygonFootprintLayerType(layer.type) ||
          layer.type === 'trip' ||
          layer.type === 'icon' ||
          layer.type === 'hexagonId'
            ? undefined
            : binding.colorOverride,
        geojson,
        trip,
        icon,
        h3,
        styleRender: translated.plan.render,
        channels: {
          ...binding.channels,
          colors: colors.fillColor,
          targetColors: colors.targetColor,
          strokeColors: colors.strokeColor,
          sizes: sizes.size,
          pointSizes: sizes.pointSize
        }
      };
    } catch (error) {
      return this.unsupported(
        layer.id,
        binding,
        error instanceof Error ? error.message : String(error)
      );
    }
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Bounded viewport geometry metadata for diagnostics; this never inspects source rows. */
  getH3ViewportMesh(layerId: string): {centerCellId: string; resolution: number} | undefined {
    const entry = this.layers.get(layerId);
    if (entry?.reason) return undefined;
    const current = entry?.h3;
    const mesh = current?.viewportMesh;
    if (!mesh || !current?.metadata.snapshot || current.runtime?.binding.branch !== 'columns')
      return undefined;
    return {centerCellId: mesh.centerCellId, resolution: mesh.resolution};
  }

  /** Bounded native topology diagnostics for the current layer generation. */
  getH3Metadata(layerId: string) {
    const entry = this.layers.get(layerId);
    if (!entry?.h3) return undefined;
    return {
      ...entry.h3.metadata.status,
      branch: entry.h3.runtime?.binding.branch,
      reason: entry.reason
    };
  }

  encode(encoder: CommandEncoder): void {
    if (this.destroyed) return;
    for (const entry of this.layers.values()) {
      entry.h3?.metadata.encode(encoder);
      if (entry.reason) continue;
      for (const channel of entry.channels.values()) channel.style.encode(encoder);
      for (const property of entry.properties.values()) property.runtime.encode(encoder);
      entry.geometry?.runtime.encode(encoder);
      entry.trip?.runtime.encode(encoder);
      entry.icon?.runtime?.encode(encoder);
      entry.h3?.runtime?.encode(encoder);
    }
  }
  discardEncoding(): void {
    for (const entry of this.layers.values()) {
      entry.h3?.metadata.discardEncoding();
      entry.h3?.runtime?.discardEncoding();
      for (const channel of entry.channels.values()) channel.style.discardEncoding();
      for (const property of entry.properties.values()) property.runtime.discardEncoding();
    }
  }

  /** Call after submission; unchanged revisions perform no additional readback. */
  async readLegends(): Promise<void> {
    const reads: Promise<void>[] = [];
    for (const [layerId, entry] of this.layers) {
      const h3 = entry.h3;
      if (h3 && !h3.metadata.snapshot) {
        if (!h3.metadataPending) {
          const pending = h3.metadata
            .readMetadata()
            .then(() => {
              // A newer revision can supersede the single bounded read in flight. Schedule the
              // next frame even for its null stale result so the current summary is retried.
              if (this.layers.get(layerId) === entry && entry.h3 === h3) this.requestUpdate();
            })
            .catch(error => {
              if (entry.h3 === h3)
                this.report(layerId, error instanceof Error ? error.message : String(error));
            })
            .finally(() => {
              if (h3.metadataPending === pending) h3.metadataPending = undefined;
            });
          h3.metadataPending = pending;
        }
        reads.push(h3.metadataPending);
      }
      for (const [channel, derived] of entry.jenks) {
        if (derived.controls || derived.reason) continue;
        if (!derived.pending) {
          const key = derived.key;
          const pending = this.readJenks(derived)
            .then(controls => {
              if (this.layers.get(layerId) === entry && entry.jenks.get(channel)?.key === key)
                derived.controls = controls;
            })
            .catch(error => {
              if (entry.jenks.get(channel)?.key === key)
                derived.reason = error instanceof Error ? error.message : String(error);
            })
            .finally(() => {
              derived.pending = undefined;
              if (this.layers.get(layerId) === entry) this.requestUpdate();
            });
          derived.pending = pending;
        }
        reads.push(derived.pending);
      }
      if (entry.reason) continue;
      for (const [channel, current] of entry.channels) {
        // Scalar logarithmic domains need the same bounded validation/diagnostic as colors.
        if (current.style.currentPlan.kind !== 'color' && current.style.currentPlan.scale !== 'log')
          continue;
        const revision = current.style.currentRevision;
        if (current.readRevision === revision) continue;
        if (current.pending) {
          reads.push(current.pending);
          continue;
        }
        current.readRevision = revision;
        const pending = current.style
          .readLegend()
          .then(legend => {
            if (
              this.destroyed ||
              this.layers.get(layerId) !== entry ||
              entry.channels.get(channel) !== current ||
              current.style.currentRevision !== revision
            )
              return;
            if (legend) {
              current.legend = legend;
              current.reason = undefined;
              this.report(layerId, this.getDomainDiagnostic(entry));
            } else {
              current.readRevision = undefined;
              current.reason = 'Resident legend is pending a current submitted style revision.';
            }
            this.requestUpdate();
          })
          .catch(error => {
            if (this.destroyed || entry.channels.get(channel) !== current) return;
            current.reason = error instanceof Error ? error.message : String(error);
            this.requestUpdate();
          })
          .finally(() => {
            if (current.pending === pending) current.pending = undefined;
          });
        current.pending = pending;
        reads.push(pending);
      }
    }
    await Promise.all(reads);
  }
  /**
   * Current-revision derived Jenks thresholds, for seeding Kepler's custom-breaks editor. Edited
   * breaks switch the layer to the native `custom` scale, whose explicit controls build a new
   * revisioned resident plan and legend; this runtime never mutates the layer configuration.
   */
  getDerivedJenksThresholds(
    layerId: string,
    channel: NativeStyleChannel
  ): NativeJenksThresholds | undefined {
    const entry = this.layers.get(layerId);
    const derived = entry?.jenks.get(channel);
    return derived?.controls &&
      derived.controls.source.datasetId === entry?.datasetId &&
      derived.controls.source.revision === entry?.sourceRevision
      ? derived.controls
      : undefined;
  }
  /** Column-keyed lookup for table-bridge domains, which carry no layer identity. */
  getDerivedJenksThresholdsForColumn(
    datasetId: string,
    column: string
  ): NativeJenksThresholds | undefined {
    for (const [layerId, entry] of this.layers) {
      if (entry.datasetId !== datasetId) continue;
      for (const channel of entry.jenks.keys()) {
        const derived = this.getDerivedJenksThresholds(layerId, channel);
        if (derived?.column === column) return derived;
      }
    }
    return undefined;
  }
  getLegend(layerId: string, channel: NativeStyleChannel): ResidentStyleLegendSnapshot | null {
    const entry = this.layers.get(layerId);
    if (!entry || !entry.mapped.has(channel)) return null;
    const current = entry.channels.get(channel);
    const plan = current?.style.currentPlan;
    return {
      expected: {
        styleId: `${layerId}:${channel}`,
        revision: current?.style.currentRevision ?? 0,
        source: {datasetId: entry.datasetId, revision: entry.sourceRevision},
        ...(plan?.domain.kind === 'selection'
          ? {selectionRevision: plan.domain.revision}
          : plan?.domain.kind === 'fixed' && plan.domain.selectionRevision !== undefined
            ? {selectionRevision: plan.domain.selectionRevision}
            : {})
      },
      legend: entry.reason || current?.reason ? undefined : current?.legend,
      reason: entry.reason ?? current?.reason
    };
  }
  retainLayers(ids: readonly string[], completion: PromiseLike<unknown>): void {
    const retained = new Set(ids);
    for (const layerId of this.layers.keys())
      if (!retained.has(layerId)) this.removeLayer(layerId, completion);
  }
  /** Await before freeing a publication's borrowed geometry or source resources. */
  async retireDataset(datasetId: string, completion: PromiseLike<unknown>): Promise<void> {
    const pending = [...this.layers]
      .filter(([, entry]) => entry.datasetId === datasetId)
      .flatMap(([layerId]) => this.removeLayer(layerId, completion));
    const results = await Promise.allSettled([
      ...pending,
      ...(this.datasetRetirements.get(datasetId) ?? [])
    ]);
    const failed = results.find(result => result.status === 'rejected');
    if (failed?.status === 'rejected') throw failed.reason;
  }

  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.listeners.clear();
    for (const layerId of this.layers.keys()) this.removeLayer(layerId, completion);
    this.destruction = Promise.allSettled([
      this.cache.destroy(completion),
      ...this.retirements
    ]).then(results => {
      const failed = results.find(result => result.status === 'rejected');
      if (failed?.status === 'rejected') throw failed.reason;
    });
    return this.destruction;
  }
  private async readJenks(derived: JenksEntry): Promise<NativeJenksThresholds> {
    const injected = this.props.naturalBreaks;
    return readNativeJenksThresholds({
      ...derived.request,
      reserveMemory: byteLength =>
        this.props.session.registry.reserveMemory(byteLength, 'cpu', 'readback'),
      // Exact native breaks at every size: a module Worker, or in place without Worker support.
      computeBreaks: async (values, classCount) =>
        injected
          ? injected(Array.from(values), classCount)
          : computeJenksBreaksInWorker({values, classCount}, async () => {
              const keplerUtilities = (await import('@kepler.gl/utils')) as unknown as {
                naturalBreaks: NativeNaturalBreaks;
              };
              return keplerUtilities.naturalBreaks;
            })
    });
  }
  private unsupported(
    layerId: string,
    binding: ResidentDatasetBinding,
    reason: string
  ): ResidentDatasetBinding {
    const entry = this.layers.get(layerId);
    if (entry) entry.reason = reason;
    this.report(layerId, reason);
    return {...binding, enabled: false, unsupportedReason: reason};
  }
  private getDomainDiagnostic(entry: LayerEntry): string | null {
    const labels: Record<NativeStyleChannel, string> = {
      fillColor: 'fill color',
      targetColor: 'target color',
      strokeColor: 'stroke color',
      size: 'size',
      pointSize: 'point radius',
      coverage: 'coverage',
      height: 'polygon height',
      elevationOffset: 'polygon elevation offset'
    };
    for (const [channel, current] of entry.channels) {
      if (entry.colorOverride && (channel === 'fillColor' || channel === 'targetColor')) continue;
      if (
        current.legend?.revision === current.style.currentRevision &&
        current.legend.domainStatus === 'invalid-log-domain'
      ) {
        return `The ${labels[channel]} logarithmic domain crosses or touches zero. Choose a domain entirely above or below zero.`;
      }
    }
    return null;
  }
  private removeLayer(layerId: string, completion: PromiseLike<unknown>): Promise<void>[] {
    const entry = this.layers.get(layerId);
    if (!entry) return [];
    this.layers.delete(layerId);
    const pending = [...entry.channels.values()].map(channel =>
      this.cache.remove(channel.style.currentPlan.id, completion)
    );
    for (const retirement of pending) this.trackDatasetRetirement(entry.datasetId, retirement);
    if (entry.geometry) pending.push(this.retireGeometry(entry.geometry, completion));
    if (entry.trip) pending.push(this.retireTrip(entry.trip, completion));
    if (entry.icon) pending.push(this.retireIcon(entry.icon, completion));
    if (entry.h3) pending.push(this.retireH3(entry.h3, completion));
    for (const property of entry.properties.values())
      pending.push(this.retireProperty(property, completion));
    for (const widened of entry.widened.values())
      pending.push(this.retireWidened(widened, completion));
    this.diagnostics.delete(layerId);
    this.requestUpdate();
    return pending;
  }
  private retireH3(entry: H3Entry, completion: PromiseLike<unknown>): Promise<void> {
    const retirement = Promise.allSettled([
      entry.metadata.destroy(completion),
      ...(entry.runtime ? [entry.runtime.destroy(completion)] : [])
    ]).then(results => {
      const failed = results.find(result => result.status === 'rejected');
      if (failed?.status === 'rejected') throw failed.reason;
    });
    this.trackDatasetRetirement(entry.datasetId, retirement);
    return retirement;
  }
  private resolveIcon(
    layer: NativeStyleLayer,
    entry: LayerEntry,
    dataset: KeplerArrowDataset,
    binding: ResidentDatasetBinding,
    options: NativeStyleResolveOptions,
    colors?: GPUVector<'unorm8x4'>,
    radii?: GPUVector<'float32'>
  ): NonNullable<ResidentDatasetBinding['icon']> {
    const nativeColumns = layer.config.columns;
    const latitude = nativeColumns?.['lat']?.value;
    const longitude = nativeColumns?.['lng']?.value;
    const marker = nativeColumns?.['icon']?.value;
    const altitude = nativeColumns?.['altitude']?.value ?? undefined;
    if (!latitude || !longitude || !marker)
      throw new Error('Native Icon requires latitude, longitude and icon columns.');
    const columns: ResidentIconColumns = {
      lat: latitude,
      lng: longitude,
      icon: marker,
      ...(altitude ? {altitude} : {})
    };
    const positionBinding = options.iconPositions ?? binding.point;
    if (
      !positionBinding ||
      positionBinding.columns.lat !== latitude ||
      positionBinding.columns.lng !== longitude
    )
      throw new Error('Native Icon requires matching source-aligned resident positions.');
    if (
      altitude &&
      (!('altitude' in positionBinding.columns) ||
        positionBinding.columns.altitude !== altitude ||
        positionBinding.positions.format !== 'float32x3')
    )
      throw new Error('Native Icon altitude requires an explicit matching resident XYZ upload.');
    if (!altitude && positionBinding.positions.format === 'float32x3')
      throw new Error('Native Icon without altitude requires resident XY positions.');
    const sourceInputs = [
      dataset.source,
      latitude,
      longitude,
      marker,
      altitude,
      layer.iconGeometry,
      layer.iconGeometryVersion
    ];
    let icon = entry.icon;
    if (!icon || sourceInputs.some((input, index) => input !== icon!.sourceInputs[index])) {
      const replacement: IconEntry = {
        datasetId: entry.datasetId,
        sourceInputs,
        retirements: new Set()
      };
      try {
        // Native geometry arrives asynchronously. Its version/redraw callback re-enters resolve;
        // an absent catalog is pending and never substituted with another marker shape.
        if (!layer.iconGeometry) throw new Error('Native icon geometry is loading.');
        const catalog = prepareNativeIconGeometry(layer.iconGeometry);
        const source = prepareNativeIconSource(
          dataset.source.batches.map((batch, index) => {
            const icons = batch.getChild(marker);
            const latitudes = batch.getChild(latitude);
            const longitudes = batch.getChild(longitude);
            const altitudes = altitude ? batch.getChild(altitude) : undefined;
            if (!icons || !latitudes || !longitudes || (altitude && !altitudes))
              throw new Error('Native Icon columns are absent from the retained Arrow source.');
            const validity = new Uint32Array(batch.numRows);
            const values = Array.from({length: batch.numRows}, (_, row) => {
              // Native iconPosAccessor does not coerce nulls or numeric strings into coordinates.
              validity[row] = Number(
                isIconCoordinateVisible(
                  longitudes.get(row),
                  latitudes.get(row),
                  altitudes?.get(row),
                  Boolean(altitudes)
                )
              );
              return icons.get(row);
            });
            return {id: `${entry.datasetId}:${index}`, icons: values, validity};
          }),
          catalog
        );
        replacement.icons = makeGPUVectorsFromNativeIcons(this.props.session.device, source, {
          workingMemoryBudgetBytes: this.props.workingMemoryBudgetBytesPerStyle ?? 32 * 1024 * 1024,
          memoryRegistry: this.props.session.registry
        });
      } catch (error) {
        replacement.sourceReason = error instanceof Error ? error.message : String(error);
      }
      entry.icon = replacement;
      if (icon) this.retireIcon(icon, this.createCompletion());
      icon = replacement;
      this.requestUpdate();
    }
    if (!icon.icons || icon.sourceReason)
      throw new Error(icon.sourceReason ?? 'Native Icon source is unavailable.');
    const inputs = [icon.icons, positionBinding.positions, binding.selectionMask, colors, radii];
    if (!icon.runtime || inputs.some((input, index) => input !== icon!.runtimeInputs?.[index])) {
      let runtime: IconRuntime | undefined;
      try {
        runtime = new IconRuntime({
          device: this.props.session.device,
          columns,
          icons: icon.icons,
          positions: positionBinding.positions,
          selectionMask: binding.selectionMask,
          colors,
          radii,
          workingMemoryBudgetBytes: this.props.workingMemoryBudgetBytesPerStyle ?? 32 * 1024 * 1024,
          memoryRegistry: this.props.session.registry
        });
        runtime.prepare();
      } catch (error) {
        if (runtime) this.track(runtime.destroy(Promise.resolve()));
        throw error;
      }
      const previous = icon.runtime;
      icon.runtime = runtime;
      icon.runtimeInputs = inputs;
      if (previous) this.retireIconRuntime(icon, previous, this.createCompletion());
      this.requestUpdate();
    }
    return icon.runtime.binding;
  }
  private retireIcon(icon: IconEntry, completion: PromiseLike<unknown>): Promise<void> {
    const retirement = (async () => {
      try {
        const results = await Promise.allSettled([
          ...icon.retirements,
          ...(icon.runtime ? [icon.runtime.destroy(completion)] : [])
        ]);
        const failed = results.find(result => result.status === 'rejected');
        if (failed?.status === 'rejected') throw failed.reason;
      } finally {
        // Mesh/row-map buffers outlive every gather that borrows them, including failed fences.
        await icon.icons?.destroy(completion);
      }
    })();
    this.trackDatasetRetirement(icon.datasetId, retirement);
    return retirement;
  }
  private retireIconRuntime(
    icon: IconEntry,
    runtime: IconRuntime,
    completion: PromiseLike<unknown>
  ): void {
    const retirement = runtime.destroy(completion);
    icon.retirements.add(retirement);
    void retirement.finally(() => icon.retirements.delete(retirement)).catch(() => {});
    this.trackDatasetRetirement(icon.datasetId, retirement);
  }
  private createProperty<Format extends 'float32' | 'unorm8x4'>(
    datasetId: string,
    source: GeoJSONPropertySource<Format>,
    fallback: number | readonly number[]
  ): {
    datasetId: string;
    runtime: GeoJSONPropertyRuntime<Format>;
    reservation: KeplerMemoryReservation;
  } {
    const byteCapacity =
      4 + source.values.data.reduce((sum, chunk) => sum + Math.max(4, chunk.length * 4), 0);
    const reservation = this.props.session.registry.reserveMemory(byteCapacity, 'gpu', 'scratch');
    let runtime: GeoJSONPropertyRuntime<Format> | undefined;
    try {
      runtime = new GeoJSONPropertyRuntime(this.props.session.device, source, fallback);
      // This graph has no transient buffers; accounting must include all owned outputs and controls.
      assert(runtime.accountedResidentByteLength <= byteCapacity);
      return {datasetId, runtime, reservation};
    } catch (error) {
      runtime?.destroy();
      reservation.release();
      throw error;
    }
  }
  private retireWidened(
    widened: WidenedColumnEntry,
    completion: PromiseLike<unknown>
  ): Promise<void> {
    return Promise.resolve(completion)
      .finally(() => {
        widened.evaluator?.destroy();
        widened.reservation.release();
      })
      .then(() => undefined);
  }
  private retireProperty(property: PropertyEntry, completion: PromiseLike<unknown>): Promise<void> {
    const retirement = Promise.resolve(completion)
      .finally(() => {
        property.runtime.destroy();
        property.reservation.release();
      })
      .then(() => undefined);
    this.trackDatasetRetirement(property.datasetId, retirement);
    return retirement;
  }
  private retireTrip(trip: TripEntry, completion: PromiseLike<unknown>): Promise<void> {
    const retirement = trip.runtime.destroy(completion);
    this.trackDatasetRetirement(trip.datasetId, retirement);
    return retirement;
  }
  private retireGeometry(geometry: GeometryEntry, completion: PromiseLike<unknown>): Promise<void> {
    const retirement = Promise.resolve(completion)
      .finally(() => {
        geometry.runtime.destroy();
        geometry.reservation.release();
      })
      .then(() => undefined);
    this.trackDatasetRetirement(geometry.datasetId, retirement);
    return retirement;
  }
  private createCompletion(): Promise<void> {
    const fence = this.props.session.device.createFence();
    return Promise.resolve(fence.signaled)
      .finally(() => fence.destroy())
      .then(() => undefined);
  }
  private track(completion: Promise<void>): void {
    this.retirements.add(completion);
    void completion.finally(() => this.retirements.delete(completion)).catch(() => {});
  }
  /** Publication cleanup waits for every older generation that still borrows its source buffers. */
  private trackDatasetRetirement(datasetId: string, completion: Promise<void>): void {
    let retirements = this.datasetRetirements.get(datasetId);
    if (!retirements) {
      retirements = new Set();
      this.datasetRetirements.set(datasetId, retirements);
    }
    const retained = retirements;
    retained.add(completion);
    void completion
      .finally(() => {
        retained.delete(completion);
        if (!retained.size && this.datasetRetirements.get(datasetId) === retained)
          this.datasetRetirements.delete(datasetId);
      })
      .catch(() => {});
    this.track(completion);
  }
  private report(layerId: string, reason: string | null): void {
    if (this.diagnostics.get(layerId) === reason) return;
    this.diagnostics.set(layerId, reason);
    this.requestUpdate();
    queueMicrotask(() => {
      if (!this.destroyed && this.diagnostics.get(layerId) === reason)
        this.props.onUnsupported?.(layerId, reason);
    });
  }
  private requestUpdate(): void {
    if (this.updateQueued || this.destroyed) return;
    this.updateQueued = true;
    queueMicrotask(() => {
      this.updateQueued = false;
      if (!this.destroyed) {
        this.props.requestUpdate();
        for (const listener of this.listeners) listener();
      }
    });
  }
}
