// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The 30-file closure of the former @deck.gl-community/gpu-layers package, copied into the Kepler
// host example for deck.gl 10. Example code only; never published.

export {GPUScatterplotLayer, type GPUScatterplotLayerProps} from './layers/gpu-scatterplot-layer';
export {GPULineLayer} from './layers/gpu-line-layer';
export {GPUArrowHeadLayer} from './layers/gpu-arrow-head-layer';
export {GPUMeshIconLayer} from './layers/gpu-mesh-icon-layer';
export {GPUArcLayer} from './layers/gpu-arc-layer';
export {GPUMeshColumnLayer} from './layers/gpu-mesh-column-layer';
export {GPUTripMarkerLayer} from './layers/gpu-trip-marker-layer';
export {GPUPathLayer, type GPUPathLayerProps} from './layers/gpu-path-layer';
export {
  GPUSolidPolygonLayer,
  type GPUSolidPolygonLayerProps
} from './layers/gpu-solid-polygon-layer';
export {FlowParticleLayer, type FlowParticleLayerProps} from './layers/flow-particle-layer';
export {GPUGlobeDepthDiskLayer} from './layers/gpu-globe-depth-disk-layer';
export {
  getGlobeTileParameters,
  insertGlobeDepthDisk,
  isGlobeTileLayer
} from './layers/globe-tile-occlusion';
export {
  GPUPointLabelLayer,
  type GPUPointLabelLayerProps,
  type GPUPointLabelBatch
} from './layers/gpu-point-label-layer';
export {GPUFlowLinesLayer} from './layers/gpu-flow-lines-layer';
export {GPUFlowCirclesLayer} from './layers/gpu-flow-circles-layer';
export {
  WGSLShadowLightingEffect,
  type WGSLShadowLightingEffectOptions
} from './effects/wgsl-shadow-effect';
export {installScenegraphShadowPatch} from './effects/wgsl-shadow-scenegraph';
