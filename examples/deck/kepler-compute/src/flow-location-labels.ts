// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUPointLabelLayer} from '../gpu-layers/index';
import type {Layer} from '@deck.gl/core';
import type {LayersData} from '@flowmap.gl/data';
import {makePointLabelRuntime, type PointLabelRuntime} from './label-runtime';
import {preparePointLabelSource} from './label-source';
import {placeLabelsGreedy} from './label-collision';

/** Labels are admitted for at most this many in-view locations (largest circles first). */
export const FLOW_LOCATION_LABEL_LIMIT = 1024;
const MAXIMUM_LABEL_CHARACTERS = 48;
const LABEL_FIELD_NAME = 'flow-location-name';
const RETIRE_DELAY_MILLISECONDS = 2000;

export type FlowLocationLabelRows = {
  names: string[];
  /** lon, lat per label. */
  positions: Float32Array;
  /** Outer circle radius in pixels per label. */
  radii: Float32Array;
};

export type FlowLocationLabelViewport = {
  width: number;
  height: number;
  project(position: [number, number]): [number, number];
};

/**
 * Native flowmap.gl labels every layer location. This keeps the locations whose circle center is
 * inside the viewport (with a margin), drops unnamed ones, and keeps the largest circles when more
 * than `limit` remain, preserving original order for stable label identity.
 */
export function selectFlowLocationLabels(
  layersData: Pick<LayersData, 'circleAttributes'> & {locationLabels?: string[]},
  viewport: FlowLocationLabelViewport,
  limit = FLOW_LOCATION_LABEL_LIMIT,
  /** When set, overlapping labels are dropped (bounded greedy placement, largest circles first). */
  collision?: {labelSize: number; padding?: number}
): FlowLocationLabelRows {
  const names = layersData.locationLabels;
  const attributes = layersData.circleAttributes?.attributes;
  if (!names || !attributes)
    return {names: [], positions: new Float32Array(0), radii: new Float32Array(0)};
  const {getPosition, getInRadius, getOutRadius} = attributes;
  const margin = 32;
  const candidates: {index: number; radius: number}[] = [];
  for (let index = 0; index < names.length; index++) {
    const name = names[index];
    if (!name) continue;
    const stride = getPosition.size;
    const [x, y] = viewport.project([
      getPosition.value[index * stride],
      getPosition.value[index * stride + 1]
    ]);
    if (
      !(
        x >= -margin &&
        y >= -margin &&
        x <= viewport.width + margin &&
        y <= viewport.height + margin
      )
    )
      continue;
    candidates.push({index, radius: Math.max(getInRadius.value[index], getOutRadius.value[index])});
  }
  let selected = candidates;
  if (collision) {
    const {labelSize, padding = 4} = collision;
    // Labels sit below the circle, centered on it; glyph advance is approximated at 0.6 em.
    const accepted = new Set(
      placeLabelsGreedy(
        candidates.map(candidate => {
          const stride = getPosition.size;
          const [x, y] = viewport.project([
            getPosition.value[candidate.index * stride],
            getPosition.value[candidate.index * stride + 1]
          ]);
          const width =
            Math.min(names[candidate.index].length, MAXIMUM_LABEL_CHARACTERS) * labelSize * 0.6;
          return {
            rowIndex: candidate.index,
            x: x - width / 2,
            y: y + candidate.radius,
            width,
            height: labelSize * 1.2,
            priority: candidate.radius
          };
        }),
        {padding, maximumCandidates: Math.max(limit * 4, 4096)}
      )
    );
    selected = candidates.filter(candidate => accepted.has(candidate.index));
  }
  if (selected.length > limit) {
    selected = [...selected]
      .sort((a, b) => b.radius - a.radius || a.index - b.index)
      .slice(0, limit);
    selected.sort((a, b) => a.index - b.index);
  }
  const positions = new Float32Array(selected.length * 2);
  const radii = new Float32Array(selected.length);
  const rowNames: string[] = [];
  selected.forEach((candidate, row) => {
    const stride = getPosition.size;
    positions[row * 2] = getPosition.value[candidate.index * stride];
    positions[row * 2 + 1] = getPosition.value[candidate.index * stride + 1];
    radii[row] = candidate.radius;
    rowNames.push(
      names[candidate.index].replace(/[\t\r\n]+/g, ' ').slice(0, MAXIMUM_LABEL_CHARACTERS)
    );
  });
  return {names: rowNames, positions, radii};
}

export function getFlowLocationLabelSignature(rows: FlowLocationLabelRows): string {
  let signature = String(rows.names.length);
  for (let row = 0; row < rows.names.length; row++) {
    signature += `|${rows.names[row]}@${rows.positions[row * 2].toFixed(4)},${rows.positions[row * 2 + 1].toFixed(4)}`;
  }
  return signature;
}

export type FlowLocationLabelStyle = {
  color: readonly number[];
  size: number;
  outlineColor: readonly number[];
};

/**
 * Location labels for the WGSL flowmap, drawn by the same GPU text path as point labels
 * (`PointLabelRuntime` + `GPUPointLabelLayer`). The runtime is rebuilt only when the in-view label
 * set changes; the replaced runtime is destroyed after a short delay so a frame that still borrows
 * its resources can finish.
 */
export class FlowLocationLabels {
  private runtime?: PointLabelRuntime;
  private rows?: FlowLocationLabelRows;
  private signature = '';
  private resources:
    | {positions: GPUVector<'float32x2'>; radii: GPUVector<'float32'>; buffers: Buffer[]}
    | undefined;
  private ready = false;
  private failure?: string;
  private destroyed = false;
  private generation = 0;

  constructor(
    private readonly device: Device,
    private readonly requestUpdate: () => void
  ) {}

  get diagnostic(): string | undefined {
    return this.failure;
  }

  get labelCount(): number {
    return this.rows?.names.length ?? 0;
  }

  /** Idempotent per label set; returns true when a new runtime was started. */
  setRows(rows: FlowLocationLabelRows): boolean {
    if (this.destroyed) return false;
    const signature = getFlowLocationLabelSignature(rows);
    if (signature === this.signature) return false;
    this.signature = signature;
    this.retire();
    this.rows = rows;
    this.failure = undefined;
    this.ready = false;
    if (!rows.names.length) return true;
    const generation = ++this.generation;
    const buffers: Buffer[] = [];
    try {
      const positionBuffer = this.device.createBuffer({
        id: 'flow-location-label-positions',
        data: rows.positions,
        usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_DST
      });
      const radiusBuffer = this.device.createBuffer({
        id: 'flow-location-label-radii',
        data: rows.radii,
        usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_DST
      });
      buffers.push(positionBuffer, radiusBuffer);
      const positions = new GPUVector<'float32x2'>({
        type: 'data',
        name: 'flow-location-label-positions',
        format: 'float32x2',
        data: [
          new GPUData({
            format: 'float32x2',
            buffer: positionBuffer,
            length: rows.names.length,
            ownsBuffer: false
          })
        ],
        ownsData: true
      });
      const radii = new GPUVector<'float32'>({
        type: 'data',
        name: 'flow-location-label-radii',
        format: 'float32',
        data: [
          new GPUData({
            format: 'float32',
            buffer: radiusBuffer,
            length: rows.names.length,
            ownsBuffer: false
          })
        ],
        ownsData: true
      });
      this.resources = {positions, radii, buffers};
      const source = preparePointLabelSource({
        fieldName: LABEL_FIELD_NAME,
        chunks: [{id: 'flow-locations', values: rows.names}],
        maxCharactersPerLabel: MAXIMUM_LABEL_CHARACTERS,
        maxGlyphs: FLOW_LOCATION_LABEL_LIMIT * MAXIMUM_LABEL_CHARACTERS,
        maxSourceBytes: 4 * 1024 * 1024
      });
      const runtime = makePointLabelRuntime(this.device, {
        source,
        positions,
        sdf: true,
        workingMemoryBudgetBytes: 128 * 1024 * 1024
      });
      this.runtime = runtime;
      void runtime.ready.then(
        () => {
          if (generation !== this.generation || this.destroyed) return;
          this.ready = true;
          this.requestUpdate();
        },
        error => {
          if (generation === this.generation) this.failure = String(error?.message ?? error);
        }
      );
    } catch (error) {
      for (const buffer of buffers) buffer.destroy();
      this.resources = undefined;
      this.runtime = undefined;
      this.failure = error instanceof Error ? error.message : String(error);
    }
    return true;
  }

  /** The GPU label layer for the current set, or none until the atlas is ready. */
  getLayers(options: {
    id: string;
    style: FlowLocationLabelStyle;
    mapState: {longitude: number; latitude: number; zoom: number};
    visible?: boolean;
  }): Layer[] {
    if (this.destroyed || !this.ready || !this.runtime || !this.resources) return [];
    const {style} = options;
    const result = this.runtime.resolve({
      label: {
        field: {name: LABEL_FIELD_NAME, type: 'string'},
        color: style.color,
        size: style.size,
        anchor: 'middle',
        alignment: 'bottom',
        outlineWidth: 0.15,
        outlineColor: style.outlineColor
      },
      selectionMask: undefined as never,
      radius: this.resources.radii,
      radiusScale: 1,
      mapState: options.mapState
    });
    if ('diagnostic' in result) {
      this.failure = result.diagnostic;
      return [];
    }
    return [
      new GPUPointLabelLayer({
        id: options.id,
        ...result.layerProps,
        // Radii are already screen pixels.
        radiusPixelScale: 1,
        visible: options.visible ?? true
      })
    ];
  }

  private retire(): void {
    const runtime = this.runtime;
    const resources = this.resources;
    this.runtime = undefined;
    this.resources = undefined;
    if (!runtime && !resources) return;
    setTimeout(() => {
      void (runtime?.destroy(Promise.resolve()) ?? Promise.resolve()).finally(() => {
        for (const buffer of resources?.buffers ?? []) buffer.destroy();
      });
    }, RETIRE_DELAY_MILLISECONDS);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.generation++;
    this.retire();
  }
}
