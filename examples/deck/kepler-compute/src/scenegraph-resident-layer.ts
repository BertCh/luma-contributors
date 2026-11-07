// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  CompositeLayer,
  type GetPickingInfoParams,
  type Layer,
  type LayerContext,
  type LayerProps,
  type PickingInfo,
  type UpdateParameters
} from '@deck.gl/core';
import {ScenegraphLayer, type ScenegraphLayerProps} from '@deck.gl/mesh-layers';
import {load} from '@loaders.gl/core';
import {GLTFLoader} from '@loaders.gl/scene/gltf';
import {assert, Buffer, type CommandEncoder, type Device, type ShaderLayout} from '@luma.gl/core';
import {Computation, DynamicBuffer} from '@luma.gl/engine';
import {postProcessGLTF} from '@luma.gl/engine/gltf';
import {areaLighting, ibl, pbrScene, type ShaderModule} from '@luma.gl/shaders';
import type {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {toCorsSafeGcsUrl, type Layer as KeplerLayer} from '@kepler.gl/layers';

import {toRadians} from '@math.gl/core';

/**
 * Resident native Kepler "3D" layer: one instance of a glTF model per point. A compute pass reads
 * the borrowed resident positions and selection mask and writes Deck `ScenegraphLayer` instance
 * attributes (position, orientation matrix, color), so the unmodified Deck scenegraph shaders keep
 * the model's textures, PBR materials and animations while filtering never touches the CPU.
 */
export type ResidentScenegraphLayerProps = Omit<LayerProps, 'data'> & {
  /** Borrowed resident point positions (longitude, latitude and optional altitude). */
  positions: GPUVector<'float32x2' | 'float32x3'>;
  /** One nonzero flag per point; hidden points draw nothing. */
  selectionMask?: GPUVector<'uint32'>;
  /** A glTF URL that Deck's `ScenegraphLayer` loads itself. */
  scenegraph: string;
  /** Native `visConfig.sizeScale * model.scale`. */
  sizeScale: number;
  /** Native pitch, yaw and roll in degrees (Deck `getOrientation`). */
  orientation: readonly [number, number, number];
  /** Native model tint, white by default. */
  color: readonly number[];
  lighting?: 'flat' | 'pbr';
  animations?: ScenegraphLayerProps['_animations'];
};

const WORKGROUP_SIZE = 64;
/** Upper bound of instances per draw: dispatch limit of 65535 workgroups, and the storage binding limit. */
const MAXIMUM_INSTANCES_PER_DRAW = 65535 * 64;
const RECORD_WORDS = 24;
const RECORD_STRIDE = RECORD_WORDS * 4;
const RECORD_OFFSETS = {positions: 0, modelMatrix: 24, colors: 72} as const;
/** Native Kepler `_animations`. */
const KEPLER_ANIMATIONS = {'*': {speed: 5}};

const INSTANCE_SHADER = /* wgsl */ `
struct Config {
  instanceCount: u32,
  positionWordOffset: u32,
  positionWordStride: u32,
  positionComponents: u32,
  selectionEnabled: u32,
  selectionWordOffset: u32,
  selectionWordStride: u32,
  packedColor: u32,
  column0: vec4<f32>,
  column1: vec4<f32>,
  column2: vec4<f32>,
};
@group(0) @binding(0) var<storage, read_write> instanceRecords: array<u32>;
@group(0) @binding(1) var<storage, read> positionValues: array<f32>;
@group(0) @binding(2) var<storage, read> selectionValues: array<u32>;
@group(0) @binding(3) var<uniform> config: Config;

fn writeFloat(word: u32, value: f32) { instanceRecords[word] = bitcast<u32>(value); }

@compute @workgroup_size(${WORKGROUP_SIZE})
fn instanceMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= config.instanceCount) { return; }
  let base = index * ${RECORD_WORDS}u;
  for (var word = 0u; word < ${RECORD_WORDS}u; word++) { instanceRecords[base + word] = 0u; }
  if (config.selectionEnabled != 0u &&
      selectionValues[config.selectionWordOffset + index * config.selectionWordStride] == 0u) { return; }
  let first = config.positionWordOffset + index * config.positionWordStride;
  let x = positionValues[first];
  let y = positionValues[first + 1u];
  // Rows with a null or invalid coordinate arrive as NaN and draw nothing.
  if (x != x || y != y) { return; }
  var z = 0.0;
  if (config.positionComponents > 2u) {
    z = positionValues[first + 2u];
    z = select(z, 0.0, z != z);
  }
  writeFloat(base, x);
  writeFloat(base + 1u, y);
  writeFloat(base + 2u, z);
  writeFloat(base + 6u, config.column0.x);
  writeFloat(base + 7u, config.column0.y);
  writeFloat(base + 8u, config.column0.z);
  writeFloat(base + 9u, config.column1.x);
  writeFloat(base + 10u, config.column1.y);
  writeFloat(base + 11u, config.column1.z);
  writeFloat(base + 12u, config.column2.x);
  writeFloat(base + 13u, config.column2.y);
  writeFloat(base + 14u, config.column2.z);
  instanceRecords[base + 18u] = config.packedColor;
}`;

function getDataBuffer(data: GPUData): Buffer {
  return data.buffer instanceof DynamicBuffer ? data.buffer.buffer : data.buffer;
}

/** Deck `calculateTransformMatrix` with unit scale: three columns, pitch/yaw/roll in degrees. */
export function makeScenegraphOrientationColumns(
  orientation: readonly [number, number, number]
): number[] {
  const [pitch, yaw, roll] = orientation.map(degrees => toRadians(degrees));
  const [sr, sp, sw] = [Math.sin(roll), Math.sin(pitch), Math.sin(yaw)];
  const [cr, cp, cw] = [Math.cos(roll), Math.cos(pitch), Math.cos(yaw)];
  return [
    cw * cp,
    sw * cp,
    -sp,
    -sw * cr + cw * sp * sr,
    cw * cr + sw * sp * sr,
    cp * sr,
    sw * sr + cw * sp * cr,
    -cw * sr + sw * sp * cr,
    cp * cr
  ];
}

class ScenegraphInstances {
  readonly records: Buffer;
  private readonly buffers: Buffer[] = [];
  private readonly computation: Computation;
  private readonly configuration: Buffer;

  constructor(
    private readonly device: Device,
    private readonly id: string,
    private readonly positions: GPUData,
    private readonly selection: GPUData | undefined,
    private readonly components: number,
    private readonly rowStart: number,
    private readonly rowCount: number
  ) {
    assert(positions.byteOffset % 4 === 0 && positions.byteStride % 4 === 0);
    this.records = this.makeBuffer(
      'records',
      rowCount * RECORD_STRIDE,
      Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    );
    const fallback = this.makeBuffer('fallback', 16, Buffer.STORAGE | Buffer.COPY_DST);
    this.configuration = this.makeBuffer('configuration', 80, Buffer.UNIFORM | Buffer.COPY_DST);
    const types = ['storage', 'read-only-storage', 'read-only-storage', 'uniform'] as const;
    const names = ['instanceRecords', 'positionValues', 'selectionValues', 'config'];
    const shaderLayout: ShaderLayout = {
      bindings: types.map(
        (type, location) =>
          ({name: names[location], type, group: 0, location}) as ShaderLayout['bindings'][number]
      ),
      attributes: []
    };
    this.computation = new Computation(device, {
      id: `${id}-instances`,
      source: INSTANCE_SHADER,
      shaderLayout,
      bindings: {
        instanceRecords: this.records,
        positionValues: getDataBuffer(positions),
        selectionValues: selection ? getDataBuffer(selection) : fallback,
        config: this.configuration
      }
    });
  }

  encode(
    orientationColumns: readonly number[],
    color: readonly number[],
    commandEncoder?: CommandEncoder
  ): void {
    const {positions, selection} = this;
    if (this.rowCount === 0) return;
    const bytes = new ArrayBuffer(80);
    const floats = new Float32Array(bytes);
    const words = new Uint32Array(bytes);
    words[0] = this.rowCount;
    words[1] = positions.byteOffset / 4 + (this.rowStart * positions.byteStride) / 4;
    words[2] = positions.byteStride / 4;
    words[3] = this.components;
    words[4] = selection ? 1 : 0;
    words[5] = selection
      ? selection.byteOffset / 4 + (this.rowStart * selection.byteStride) / 4
      : 0;
    words[6] = selection ? selection.byteStride / 4 : 1;
    words[7] =
      (((color[3] ?? 255) << 24) |
        ((color[2] ?? 255) << 16) |
        ((color[1] ?? 255) << 8) |
        (color[0] ?? 255)) >>>
      0;
    for (let column = 0; column < 3; column++) {
      floats.set(orientationColumns.slice(column * 3, column * 3 + 3), 8 + column * 4);
    }
    this.configuration.write(new Uint8Array(bytes));
    const workgroups = Math.ceil(this.rowCount / WORKGROUP_SIZE);
    assert(workgroups <= 65535); // Instance draws are split below the dispatch limit
    const encoder = commandEncoder ?? this.device.createCommandEncoder({id: `${this.id}-encode`});
    const computePass = encoder.beginComputePass({});
    this.computation.dispatch(computePass, workgroups);
    computePass.end();
    if (!commandEncoder) this.device.submit(encoder.finish());
  }

  destroy(): void {
    this.computation.destroy();
    for (const buffer of this.buffers) buffer.destroy();
    this.buffers.length = 0;
  }

  private makeBuffer(name: string, byteLength: number, usage: number): Buffer {
    const buffer = this.device.createBuffer({
      id: `${this.id}-${name}`,
      byteLength: Math.max(16, byteLength),
      usage
    });
    this.buffers.push(buffer);
    return buffer;
  }
}

type ScenegraphChunkLayerProps = ScenegraphLayerProps<unknown> & {
  encodeInstances: () => void;
  rowIndexBase: number;
};

/** The unmodified Deck scenegraph shader and materials, fed by compute-written instances. */
class ResidentScenegraphChunkLayer extends ScenegraphLayer<unknown, ScenegraphChunkLayerProps> {
  static override layerName = 'ResidentScenegraphChunkLayer';

  override getShaders() {
    const shaders = super.getShaders();
    // Deck 9.4's WGSL PBR wrapper keeps only the `lighting` dependency; the luma.gl PBR source
    // also calls the area-lighting, image-based-lighting and scene modules.
    shaders.modules = shaders.modules.map((shaderModule: ShaderModule) =>
      shaderModule.name === 'pbrMaterial' && shaderModule.dependencies
        ? {
            ...shaderModule,
            dependencies: [...shaderModule.dependencies, areaLighting, ibl, pbrScene]
          }
        : shaderModule
    );
    return shaders;
  }

  override draw(parameters: Parameters<ScenegraphLayer['draw']>[0]): void {
    if ((this.state as {scenegraph?: unknown}).scenegraph) this.props.encodeInstances();
    super.draw(parameters);
  }
}

type ChunkState = {
  instances: ScenegraphInstances;
  chunkIndex: number;
  rowIndexBase: number;
  data: {length: number; attributes: Record<string, unknown>};
  encoded: boolean;
};

type ResidentScenegraphState = {chunks: ChunkState[]; boundInputs: unknown[]};

/** Mirrors native `fetchGltf`: CORS-safe URLs, then luma glTF post-processing. */
function fetchScenegraphModel(url: string, {propName}: {propName?: string} = {}): Promise<unknown> {
  if (propName !== 'scenegraph')
    return fetch(toCorsSafeGcsUrl(url)).then(response => response.json());
  return load(url, GLTFLoader, {
    fetch: (requestUrl: string, init?: RequestInit) => fetch(toCorsSafeGcsUrl(requestUrl), init)
  } as never).then(gltf => postProcessGLTF(gltf as never));
}

export class ResidentScenegraphPointLayer extends CompositeLayer<ResidentScenegraphLayerProps> {
  static override layerName = 'ResidentScenegraphPointLayer';
  static override defaultProps = {lighting: 'pbr', animations: KEPLER_ANIMATIONS};

  override initializeState(): void {
    this.setState({chunks: [], boundInputs: []} satisfies ResidentScenegraphState);
  }

  override updateState({props}: UpdateParameters<Layer<ResidentScenegraphLayerProps>>): void {
    const {boundInputs} = this.state as ResidentScenegraphState;
    const inputs = [props.positions, props.selectionMask];
    if (inputs.some((input, index) => input !== boundInputs[index])) {
      this.destroyChunks();
      this.createChunks();
    }
  }

  override renderLayers(): Layer[] {
    const {chunks} = this.state as ResidentScenegraphState;
    const {scenegraph, sizeScale, lighting, animations} = this.props;
    return chunks.map(
      chunk =>
        new ResidentScenegraphChunkLayer(this.getSubLayerProps({id: `chunk-${chunk.chunkIndex}`}), {
          data: chunk.data,
          numInstances: chunk.data.length,
          scenegraph,
          fetch: fetchScenegraphModel,
          _lighting: lighting,
          _animations: animations,
          sizeScale,
          rowIndexBase: chunk.rowIndexBase,
          encodeInstances: () => this.drawInstances(chunk)
        } as unknown as ScenegraphChunkLayerProps)
    );
  }

  override getPickingInfo({info, sourceLayer}: GetPickingInfoParams): PickingInfo {
    if (info.index >= 0) {
      info.index += (sourceLayer?.props as Partial<ScenegraphChunkLayerProps>).rowIndexBase ?? 0;
    }
    return info;
  }

  override finalizeState(context: LayerContext): void {
    this.destroyChunks();
    super.finalizeState(context);
  }

  /** Records the instance pass after the frame's selection compute; the next draw reuses it. */
  encodeInstances(commandEncoder: CommandEncoder): void {
    const {chunks} = this.state as ResidentScenegraphState;
    for (const chunk of chunks) {
      this.encodeChunk(chunk, commandEncoder);
      chunk.encoded = true;
    }
  }

  private drawInstances(chunk: ChunkState): void {
    if (chunk.encoded) {
      chunk.encoded = false;
      return;
    }
    this.encodeChunk(chunk);
  }

  private encodeChunk(chunk: ChunkState, commandEncoder?: CommandEncoder): void {
    chunk.instances.encode(
      makeScenegraphOrientationColumns(this.props.orientation),
      this.props.color,
      commandEncoder
    );
  }

  private createChunks(): void {
    const {device} = this.context;
    assert(device.type === 'webgpu'); // ResidentScenegraphPointLayer requires WebGPU
    const {positions, selectionMask} = this.props;
    const components = positions.format === 'float32x3' ? 3 : 2;
    const chunks: ChunkState[] = [];
    const maximumInstances = Math.max(
      1,
      Math.min(
        MAXIMUM_INSTANCES_PER_DRAW,
        Math.floor(device.limits.maxStorageBufferBindingSize / RECORD_STRIDE),
        Math.floor(device.limits.maxBufferSize / RECORD_STRIDE)
      )
    );
    let rowIndexBase = 0;
    let chunkIndex = 0;
    for (const [sourceIndex, chunkPositions] of positions.data.entries()) {
      for (let rowStart = 0; rowStart < chunkPositions.length; rowStart += maximumInstances) {
        const rowCount = Math.min(maximumInstances, chunkPositions.length - rowStart);
        const instances = new ScenegraphInstances(
          device,
          `${this.id}-${chunkIndex}`,
          chunkPositions,
          selectionMask?.data[sourceIndex],
          components,
          rowStart,
          rowCount
        );
        const layout = {
          buffer: instances.records,
          stride: RECORD_STRIDE,
          bytesPerElement: 4,
          type: 'float32'
        };
        chunks.push({
          encoded: false,
          instances,
          chunkIndex,
          rowIndexBase,
          data: {
            length: rowCount,
            attributes: {
              instancePositions: {...layout, size: 3, offset: RECORD_OFFSETS.positions},
              instanceModelMatrix: {...layout, size: 12, offset: RECORD_OFFSETS.modelMatrix},
              instanceColors: {
                ...layout,
                size: 4,
                type: 'unorm8',
                bytesPerElement: 1,
                normalized: true,
                offset: RECORD_OFFSETS.colors
              }
            }
          }
        });
        rowIndexBase += rowCount;
        chunkIndex++;
      }
    }
    this.setState({
      chunks,
      boundInputs: [positions, selectionMask]
    } satisfies ResidentScenegraphState);
  }

  private destroyChunks(): void {
    const state = this.state as ResidentScenegraphState;
    for (const chunk of state.chunks) chunk.instances.destroy();
    this.setState({chunks: [], boundInputs: []} satisfies ResidentScenegraphState);
  }
}

/** Native `ScenegraphLayer` members that the renderer reads. */
type NativeScenegraphLayer = KeplerLayer & {
  getScenegraph(): {url: string | null; scale: number};
  getOrientation(scenegraph: {angles: number[]}, globeMode: boolean): [number, number, number];
  getColor(): readonly number[];
};

/** Everything `createResidentScenegraphRenderer` needs from the resident layer factory. */
export type ResidentScenegraphContext = {
  prepare: (
    layer: KeplerLayer,
    renderOptions: object
  ) =>
    | {
        point?: {positions: GPUVector<'float32x2' | 'float32x3'>; columns: object};
        selectionMask: GPUVector<'uint32'>;
      }
    | undefined;
  baseProps: (layer: KeplerLayer, binding: never, renderOptions: never) => LayerProps;
  report: (layer: KeplerLayer, reason: string | null) => void;
};

/**
 * Returns the resident native 3D layers: one model instance per resident point, honoring native
 * model, size scale, angles, tint, globe orientation, mask and source-row picking.
 */
export function renderResidentScenegraph(
  context: ResidentScenegraphContext,
  layer: KeplerLayer,
  renderOptions: {mapState?: {globe?: {enabled?: boolean}; layerParameters?: object}} & object
): Layer[] {
  const binding = context.prepare(layer, renderOptions);
  if (!binding) return [];
  const native = layer as NativeScenegraphLayer;
  const config = native.config as {columnMode?: string; visConfig: {sizeScale?: number}};
  const geometry = binding.point;
  const scenegraph = native.getScenegraph();
  const reason = !geometry
    ? 'No resident point geometry is bound to this dataset.'
    : config.columnMode && config.columnMode !== 'points'
      ? 'The 3D layer needs point columns for a resident binding.'
      : null;
  context.report(layer, reason);
  if (reason || !geometry || !scenegraph.url) return [];
  const globeMode = Boolean(renderOptions.mapState?.globe?.enabled);
  const properties = context.baseProps(layer, binding as never, renderOptions as never);
  return [
    new ResidentScenegraphPointLayer({
      ...properties,
      // Native scenegraphs draw with depth testing; the native layer opacity multiplies the model alpha.
      parameters: {...properties.parameters, depthCompare: 'less-equal', depthWriteEnabled: true},
      positions: geometry.positions,
      selectionMask: binding.selectionMask,
      scenegraph: scenegraph.url,
      sizeScale: (config.visConfig.sizeScale ?? 1) * scenegraph.scale,
      orientation: native.getOrientation(scenegraph as never, globeMode),
      color: native.getColor()
    } as ResidentScenegraphLayerProps)
  ];
}
