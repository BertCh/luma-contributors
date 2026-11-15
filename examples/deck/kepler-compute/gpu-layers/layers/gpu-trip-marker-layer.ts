// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  CompositeLayer,
  type Color,
  type GetPickingInfoParams,
  type Layer,
  type LayerContext,
  type LayerProps,
  type PickingInfo,
  type UpdateParameters
} from '@deck.gl/core';
import {ScenegraphLayer, type ScenegraphLayerProps} from '@deck.gl/mesh-layers';
import {assert, type CommandEncoder} from '@luma.gl/core';
import {areaLighting, ibl, pbrScene, type ShaderModule} from '@luma.gl/shaders';
import type {GPUVector, VertexList} from '@luma.gl/graph/gpu-data';
import {
  getGPUVectorPickingProvenance,
  type GPUVectorLayerPickingInfo
} from './gpu-vector-layer-utils';
import {validateGPUGeometryChunks, validateGPUGeometrySelection} from './gpu-geometry-selection';
import {validateGPUVectorPickingCapacity} from './gpu-vector-layer-utils';
import {
  GPUTripMarkerInstances,
  type GPUTripMarkerInstanceFrame,
  GPU_TRIP_MARKER_RECORD_OFFSETS,
  GPU_TRIP_MARKER_RECORD_STRIDE
} from './gpu-trip-marker-instances';

/**
 * Props for resident Trips scenegraph markers. The path, timestamp, selection and channel vectors
 * are borrowed. Marker position, orientation and color are evaluated by GPU compute passes at draw
 * time and drawn by Deck's `ScenegraphLayer`, so the glTF model keeps its textures, PBR materials
 * and animations while animation, filtering and grouped-trip filtered path copies never touch the
 * CPU.
 */
export type GPUTripMarkerLayerProps = Omit<LayerProps, 'data'> & {
  /** One nonzero flag per trip; hidden trips draw no marker. */
  selectionMask?: GPUVector<'uint32'>;
  getPath: GPUVector<VertexList<'float32x3'>>;
  /** Float32 timestamps aligned with getPath, in the same unit and origin as `currentTime`. */
  getTimestamps: GPUVector<VertexList<'float32'>>;
  currentTime: number;
  /**
   * A post-processed `@luma.gl/gltf` document, or a URL Deck's `ScenegraphLayer` loads itself.
   * It is instantiated once per physical trip chunk.
   */
  scenegraph: unknown;
  /** Multiplier applied to model meters (native `model.scale * 2^sizeScale`). */
  sizeScale?: number;
  /** Column-major 3x3 applied after the orientation: native adjust angles and model angles. */
  baseRotation?: readonly number[];
  /**
   * Multiplies the model's material colors: constant, or one `unorm8x4` color per trip. White by
   * default, like native.
   */
  getColor?: Color | GPUVector<'unorm8x4'>;
  /**
   * One `unorm8x4` color per path vertex (native table-mode trails): the marker takes the color
   * of the vertex at the bisected time. Wins over `getColor`.
   */
  getVertexColor?: GPUVector<'unorm8x4'>;
  /**
   * One flag per path vertex. Filtered tabular trips keep unselected vertices as repeats of the
   * previous selected one; per-vertex channels follow the same rule.
   */
  vertexSelection?: GPUVector<'uint32'>;
  /** Native roll, pitch and yaw datum fields in degrees, one value per path vertex. */
  getVertexRoll?: GPUVector<'float32'>;
  getVertexPitch?: GPUVector<'float32'>;
  getVertexYaw?: GPUVector<'float32'>;
  invertRoll?: boolean;
  invertPitch?: boolean;
  invertYaw?: boolean;
  /** Native Kepler passes `'pbr'`. */
  lighting?: 'flat' | 'pbr';
  /** Deck `_animations`; native Kepler plays every animation at speed 5. */
  animations?: ScenegraphLayerProps['_animations'];
};

type GPUTripScenegraphLayerProps = ScenegraphLayerProps<unknown> & {
  encodeInstances: () => void;
  rowIndexBase: number;
};

/** The unmodified Deck scenegraph shader and materials, fed by compute-written instances. */
class GPUTripScenegraphLayer extends ScenegraphLayer<unknown, GPUTripScenegraphLayerProps> {
  static override layerName = 'GPUTripScenegraphLayer';

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
    // Deck 10 builds the globe tangent basis east-correct, so the scenegraph shader needs no
    // east/north sign compensation here (deck 9 required one).
    return shaders;
  }

  override draw(parameters: Parameters<ScenegraphLayer['draw']>[0]): void {
    // The scenegraph is created asynchronously; instances are only needed once it can draw.
    if ((this.state as {scenegraph?: unknown}).scenegraph) this.props.encodeInstances();
    super.draw(parameters);
  }
}

type GPUTripMarkerChunk = {
  instances: GPUTripMarkerInstances;
  chunkIndex: number;
  rowIndexBase: number;
  data: {length: number; attributes: Record<string, unknown>};
  /** Set by `encodeInstances` and consumed by the next draw, which then skips its own encode. */
  encoded: boolean;
};

type GPUTripMarkerState = {
  chunks: GPUTripMarkerChunk[];
  boundInputs: unknown[];
};

const DEFAULT_BASE_ROTATION = [1, 0, 0, 0, 1, 0, 0, 0, 1];
/** Native Kepler `_animations`. */
const KEPLER_ANIMATIONS = {'*': {speed: 5}};

/**
 * Resident Trips scenegraph markers: one instance of the glTF model per trip at its position and
 * heading for the current time. Compute passes bisect each trip's resident timestamps and write
 * Deck `ScenegraphLayer` instance attributes (position, model matrix, color); the native Deck
 * scenegraph shaders then draw the model with its textures, PBR materials and animations.
 */
export class GPUTripMarkerLayer extends CompositeLayer<GPUTripMarkerLayerProps> {
  static override layerName = 'GPUTripMarkerLayer';
  static override defaultProps = {lighting: 'pbr', animations: KEPLER_ANIMATIONS};

  override initializeState(): void {
    this.setState({chunks: [], boundInputs: []} satisfies GPUTripMarkerState);
  }

  override updateState({props}: UpdateParameters<Layer<GPUTripMarkerLayerProps>>): void {
    const {boundInputs} = this.state as GPUTripMarkerState;
    const inputs = this.getInputs(props);
    if (inputs.some((input, index) => input !== boundInputs[index])) {
      this.destroyChunks();
      this.createChunks();
    }
  }

  override renderLayers(): Layer[] {
    const {chunks} = this.state as GPUTripMarkerState;
    const {scenegraph, lighting, animations} = this.props;
    return chunks.map(
      chunk =>
        new GPUTripScenegraphLayer(this.getSubLayerProps({id: `chunk-${chunk.chunkIndex}`}), {
          data: chunk.data,
          numInstances: chunk.data.length,
          scenegraph: scenegraph as ScenegraphLayerProps['scenegraph'],
          _lighting: lighting,
          _animations: animations,
          sizeScale: 1,
          rowIndexBase: chunk.rowIndexBase,
          encodeInstances: () => this.drawInstances(chunk)
        } as GPUTripScenegraphLayerProps)
    );
  }

  override getPickingInfo({info, sourceLayer}: GetPickingInfoParams): PickingInfo {
    const result = info as GPUVectorLayerPickingInfo;
    if (result.index >= 0) {
      result.index +=
        (sourceLayer?.props as Partial<GPUTripScenegraphLayerProps>).rowIndexBase ?? 0;
    }
    result.gpuVector = getGPUVectorPickingProvenance(this.props.getPath, result.index);
    return result;
  }

  override finalizeState(context: LayerContext): void {
    this.destroyChunks();
    super.finalizeState(context);
  }

  private getInputs(props: GPUTripMarkerLayerProps): unknown[] {
    return [
      props.getPath,
      props.getTimestamps,
      props.selectionMask,
      Array.isArray(props.getColor) ? undefined : props.getColor,
      props.getVertexColor,
      props.vertexSelection,
      props.getVertexRoll,
      props.getVertexPitch,
      props.getVertexYaw
    ];
  }

  /**
   * Records the instance compute passes into `commandEncoder`, after the caller's own compute (for
   * example a selection filter) that the instances depend on. Call it before the frame's render
   * pass; the next draw then reuses the result. Without it every draw submits its own compute
   * work, which cannot observe selection or style results encoded later in the same frame.
   */
  encodeInstances(commandEncoder: CommandEncoder): void {
    const {chunks} = this.state as GPUTripMarkerState;
    for (const chunk of chunks) {
      chunk.instances.encode(this.getInstanceFrame(), commandEncoder);
      chunk.encoded = true;
    }
  }

  private drawInstances(chunk: GPUTripMarkerChunk): void {
    if (chunk.encoded) {
      chunk.encoded = false;
      return;
    }
    chunk.instances.encode(this.getInstanceFrame());
  }

  private getInstanceFrame(): GPUTripMarkerInstanceFrame {
    const {props} = this;
    return {
      currentTime: props.currentTime,
      sizeScale: props.sizeScale ?? 1,
      baseRotation: props.baseRotation ?? DEFAULT_BASE_ROTATION,
      constantColor: Array.isArray(props.getColor) ? props.getColor : undefined,
      invertRoll: props.invertRoll,
      invertPitch: props.invertPitch,
      invertYaw: props.invertYaw
    };
  }

  private createChunks(): void {
    const {device} = this.context;
    assert(device.type === 'webgpu'); // GPUTripMarkerLayer requires WebGPU
    const {props} = this;
    const {getPath, getTimestamps, selectionMask, getVertexColor, vertexSelection} = props;
    const tripColors = isGPUVector(props.getColor) ? props.getColor : undefined;
    const channels = [props.getVertexRoll, props.getVertexPitch, props.getVertexYaw];
    validateGPUVectorPickingCapacity(getPath.length, Boolean(props.pickable));
    validateGPUGeometrySelection(device, getPath, selectionMask);
    validateGPUGeometryChunks(getPath, [getTimestamps, tripColors]);
    for (const vector of [getVertexColor, vertexSelection, ...channels]) {
      // Per-vertex channels share the path's physical chunks.
      assert(!vector || vector.data.length === getPath.data.length);
    }
    const chunks: GPUTripMarkerChunk[] = [];
    let rowIndexBase = 0;
    try {
      for (const [chunkIndex, path] of getPath.data.entries()) {
        if (path.length > 0) {
          const instances = new GPUTripMarkerInstances(device, `${this.id}-${chunkIndex}`, {
            path,
            timestamps: getTimestamps.data[chunkIndex],
            selection: selectionMask?.data[chunkIndex],
            tripColors: tripColors?.data[chunkIndex],
            vertexColors: getVertexColor?.data[chunkIndex],
            vertexSelection: vertexSelection?.data[chunkIndex],
            vertexRoll: channels[0]?.data[chunkIndex],
            vertexPitch: channels[1]?.data[chunkIndex],
            vertexYaw: channels[2]?.data[chunkIndex]
          });
          const layout = {
            buffer: instances.records,
            stride: GPU_TRIP_MARKER_RECORD_STRIDE,
            bytesPerElement: 4,
            type: 'float32'
          };
          chunks.push({
            encoded: false,
            instances,
            chunkIndex,
            rowIndexBase,
            data: {
              length: path.length,
              // Deck binary attributes, keyed by attribute id: the interleaved compute records.
              attributes: {
                instancePositions: {
                  ...layout,
                  size: 3,
                  offset: GPU_TRIP_MARKER_RECORD_OFFSETS.positions
                },
                instanceModelMatrix: {
                  ...layout,
                  size: 12,
                  offset: GPU_TRIP_MARKER_RECORD_OFFSETS.modelMatrix
                },
                instanceColors: {
                  ...layout,
                  size: 4,
                  type: 'unorm8',
                  bytesPerElement: 1,
                  normalized: true,
                  offset: GPU_TRIP_MARKER_RECORD_OFFSETS.colors
                }
              }
            }
          });
        }
        rowIndexBase += path.length;
      }
    } catch (error) {
      for (const chunk of chunks) chunk.instances.destroy();
      throw error;
    }
    this.setState({chunks, boundInputs: this.getInputs(props)} satisfies GPUTripMarkerState);
  }

  private destroyChunks(): void {
    const state = this.state as GPUTripMarkerState;
    for (const chunk of state.chunks) chunk.instances.destroy();
    this.setState({chunks: [], boundInputs: []} satisfies GPUTripMarkerState);
  }
}

function isGPUVector(value: unknown): value is GPUVector<'unorm8x4'> {
  return Boolean(value && typeof value === 'object' && 'data' in value && 'format' in value);
}
