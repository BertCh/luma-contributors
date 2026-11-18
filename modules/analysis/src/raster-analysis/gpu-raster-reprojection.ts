// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterStorageBindings} from './raster-grid/raster-storage-preflight';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {CompiledProjection, ProjectionShader} from '@luma.gl/graph/gpu-project';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  getRasterBufferBandNodes,
  getRasterBandViews,
  snapshotRasterBand,
  validateRasterGrid,
  validateRasterTextureBand,
  validateRasterBandBelongsToGraph,
  validateParameterTarget,
  validateRasterParameters
} from './raster-band-utils';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUParameterBuffer,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  getGraphOutputViews as getRasterPreflightOutputViews
} from '../analysis-core/kernels/gpu-contributor-utils';
import {resolveGPURasterContext} from '../analysis-core/contracts/raster-grid';
import {IS_FINITE_VALUE_WGSL} from '../analysis-core/kernels/wgsl-common';
import {
  requireRasterLength,
  validateRasterExpansionViews,
  validateRasterExpansionGraph
} from './raster-expansion-utils';

export type GPURasterReprojectionProps = {
  id?: string;
  sourceGrid: GPURasterMetadata;
  destinationGrid: GPURasterMetadata;
  input: GPURasterBand;
  /** Explicit destination-world to source-world program. Omit only for grids in the same CRS. */
  inverseProjection?: CompiledProjection;
  method?: 'nearest' | 'bilinear';
  noDataPolicy?: 'strict' | 'renormalize';
  /** Per-frame affines packed by getGPURasterReprojectionParameterValues; shapes stay fixed. */
  parameters?: GraphDataView<'float32'>;
  output: {values: GraphDataView<'float32'>; validity: GraphDataView<'uint32'>};
};

/** Number of float32 words in the reprojection georeference. */
export const GPU_RASTER_REPROJECTION_PARAMETER_LENGTH = 20;

/** Layout version stored in word 17; older manually packed layouts must be repacked. */
export const GPU_RASTER_REPROJECTION_PARAMETER_VERSION = 3;

/** Per-frame grids; dimensions, CRS and pixel interpretation must match the contributor. */
export type GPURasterReprojectionSettings = {
  sourceGrid: GPURasterMetadata;
  destinationGrid: GPURasterMetadata;
  /** Same compiled inverse projection used by the contributor. */
  inverseProjection?: CompiledProjection;
};

/**
 * Packs layout version 3, composing same-CRS affines in float64 before rounding once.
 * Layout: `[a, b, d, e, originX, originY, inverseA, inverseB, inverseD, inverseE, 0,
 * sampleShiftX, sampleShiftY, sourceOriginXHigh, sourceOriginXLow, sourceOriginYHigh,
 * sourceOriginYLow, version, anchorColumn, anchorRow]`.
 * Without a projection, subtract the destination pixel anchor in words 18..19 before applying
 * words 0..5 to obtain source sample coordinates. The anchor is near source support, preserving
 * cancellation between large scales and translations. Area/point shifts are already included.
 * With a projection, words 0..5 are the destination world affine, words 6..9 are the source
 * inverse linear affine computed in float64, and words 11..12 shift local-f32 projection output
 * relative to the source origin. Double-single results subtract the split source origin first.
 * Projected packed inverse matrices must remain invertible. Versions 1 and 2 must be repacked.
 *
 * @param settings Grids and the contributor's optional inverse projection.
 * @param target Optional destination of at least 20 words.
 */
export function getGPURasterReprojectionParameterValues(
  settings: GPURasterReprojectionSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_REPROJECTION_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterReprojectionParameterValues';
  validateParameterTarget(id, target, GPU_RASTER_REPROJECTION_PARAMETER_LENGTH);
  const {sourceGrid: source, destinationGrid: destination, inverseProjection} = settings;
  resolveGPURasterContext(id, {
    grid: source,
    localOrigin: [source.affine[2], source.affine[5]]
  });
  const destinationContext = resolveGPURasterContext(id, {
    grid: destination,
    localOrigin: [destination.affine[2], destination.affine[5]]
  });
  const projectionOrigin = inverseProjection?.destinationOrigin;
  const projected = inverseProjection !== undefined;
  const sourceOrigin =
    inverseProjection?.precision === 'double-single'
      ? [source.affine[2], source.affine[5]]
      : [0, 0];
  const sourceOriginHigh = sourceOrigin.map(Math.fround);
  const [sourceA, sourceB, sourceOriginX, sourceD, sourceE, sourceOriginY] = source.affine;
  const [
    destinationA,
    destinationB,
    destinationOriginX,
    destinationD,
    destinationE,
    destinationOriginY
  ] = destination.affine;
  const determinant = sourceA * sourceE - sourceB * sourceD;
  // Divide after forming each numerator: identical axes then produce exactly 1 and 0,
  // even when the source affine is ill-conditioned.
  const composedA = (sourceE * destinationA - sourceB * destinationD) / determinant;
  const composedB = (sourceE * destinationB - sourceB * destinationE) / determinant;
  const composedD = (sourceA * destinationD - sourceD * destinationA) / determinant;
  const composedE = (sourceA * destinationE - sourceD * destinationB) / determinant;
  const differenceX = destinationOriginX - sourceOriginX;
  const differenceY = destinationOriginY - sourceOriginY;
  const destinationShift = destination.pixelInterpretation === 'area' ? 0.5 : 0;
  const sourceShift = source.pixelInterpretation === 'area' ? 0.5 : 0;
  const translationColumn =
    (sourceE * differenceX - sourceB * differenceY) / determinant +
    (composedA + composedB) * destinationShift -
    sourceShift;
  const translationRow =
    (sourceA * differenceY - sourceD * differenceX) / determinant +
    (composedD + composedE) * destinationShift -
    sourceShift;
  // Locate source sample (0, 0) in the destination, then use a nearby destination pixel as
  // the evaluation origin. This keeps a large translation from cancelling a rounded product
  // inside source support (e.g. 3355446.4 * 3 - 10066335.2 must stay at column 4).
  const destinationDeterminant = destinationA * destinationE - destinationB * destinationD;
  const anchorWorldX = sourceShift * (sourceA + sourceB) - differenceX;
  const anchorWorldY = sourceShift * (sourceD + sourceE) - differenceY;
  const anchorColumn = projected
    ? 0
    : Math.min(
        destination.width - 1,
        Math.max(
          0,
          Math.round(
            (destinationE * anchorWorldX - destinationB * anchorWorldY) / destinationDeterminant -
              destinationShift
          )
        )
      );
  const anchorRow = projected
    ? 0
    : Math.min(
        destination.height - 1,
        Math.max(
          0,
          Math.round(
            (destinationA * anchorWorldY - destinationD * anchorWorldX) / destinationDeterminant -
              destinationShift
          )
        )
      );
  target.set([
    ...(projected
      ? [
          ...destinationContext.parameterValues.subarray(2, 6),
          destinationOriginX,
          destinationOriginY
        ]
      : [
          composedA,
          composedB,
          composedD,
          composedE,
          composedA * anchorColumn + composedB * anchorRow + translationColumn,
          composedD * anchorColumn + composedE * anchorRow + translationRow
        ]),
    ...(projected
      ? [
          sourceE / determinant,
          -sourceB / determinant,
          -sourceD / determinant,
          sourceA / determinant
        ]
      : [0, 0, 0, 0]),
    0,
    projected && inverseProjection.precision !== 'double-single'
      ? (projectionOrigin?.[0] ?? 0) - source.affine[2]
      : 0,
    projected && inverseProjection.precision !== 'double-single'
      ? (projectionOrigin?.[1] ?? 0) - source.affine[5]
      : 0,
    sourceOriginHigh[0],
    sourceOrigin[0] - sourceOriginHigh[0],
    sourceOriginHigh[1],
    sourceOrigin[1] - sourceOriginHigh[1],
    GPU_RASTER_REPROJECTION_PARAMETER_VERSION,
    anchorColumn,
    anchorRow
  ]);
  validateRequirement(
    id,
    target.subarray(0, GPU_RASTER_REPROJECTION_PARAMETER_LENGTH).every(Number.isFinite),
    'grid affine origins and differences must be finite in float32'
  );
  if (projected) {
    validateRequirement(
      id,
      target[6] * target[9] - target[7] * target[8] !== 0,
      'sourceGrid.affine inverse must remain invertible in float32'
    );
  }
  return target;
}

/**
 * Destination-driven grid alignment/reprojection. Generates destination centers and projects them
 * inside the sampling kernel, without full-destination scratch buffers. It inverts the source affine
 * and samples one explicitly supplied complete source tile. Invalid projection domains and absent
 * source support remain nodata. Rotated/sheared affines are supported.
 *
 * Precision. Without `inverseProjection` the destination-to-source sample transform, including
 * origin differences and area/point shifts, is composed in float64 on the CPU. Evaluation is
 * anchored near source support before rounding, preserving large scale/translation cancellation.
 * Identical grids therefore sample identically on wide sheared grids within float32 index precision.
 * With `inverseProjection` the source inverse linear affine is computed in float64 before packing;
 * destination centres are the float32 input of the projection, so give `destinationGrid` a local frame (a small origin, with the
 * offset to the absolute frame inside the projection program). For local-f32 output, set its
 * `destinationOrigin` near the source grid origin: the projected result is relative to that
 * origin and the kernel adds `destinationOrigin - sourceGrid origin`, differenced in float64.
 * The source tile must contain the entire requested interpolation support. This adapter
 * does not select/fetch source tiles. Call destroy after the compiled graph to release projection
 * parameters when an inverseProjection is provided. Double-single projection outputs are absolute:
 * the source origin is subtracted in double-single before converting to a tile-relative float32.
 *
 * Compile-time: grid shapes, CRS, pixel interpretation, input/output views, inverseProjection,
 * method and noDataPolicy. Per-frame: affines through parameters, source values and validity.
 * Without parameters an owned buffer holds the initial affines; call destroy after the graph. Invalid samples publish NaN and validity=0;
 * there is no bounded search, compact output or iteration requiring a scalar status port.
 */
export class GPURasterReprojection implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterReprojectionProps;
  private readonly projectionShader?: ProjectionShader;
  private projectionParameters?: GPUParameterBuffer<'uint32'>;
  private projectionRegistered = false;
  private affineParameters?: GPUParameterBuffer<'float32'>;
  constructor(props: GPURasterReprojectionProps) {
    props = {...props, input: snapshotRasterBand(props.input)};
    this.id = props.id ?? 'raster-reprojection';
    this.props = props;
    validateRasterTextureBand(this.id, props.input, props.sourceGrid);
    const count = validateRasterExpansionViews(
      this.id,
      props.destinationGrid,
      [...getRasterBandViews(props.input), ...(props.parameters ? [props.parameters] : [])],
      Object.values(props.output)
    );
    validateRasterGrid(this.id, props.sourceGrid);
    if (props.parameters) {
      validateRasterParameters(this.id, props.parameters, GPU_RASTER_REPROJECTION_PARAMETER_LENGTH);
    }
    requireRasterLength(props.output.values, count, this.id, 'output.values');
    requireRasterLength(props.output.validity, count, this.id, 'output.validity');
    const affine = props.sourceGrid.affine;
    validateRequirement(
      this.id,
      affine[0] * affine[4] - affine[1] * affine[3] !== 0,
      'sourceGrid.affine must be invertible (nonzero determinant)'
    );
    validateRequirement(
      this.id,
      !props.inverseProjection || props.inverseProjection.inputFormat === 'float32x2',
      "inverseProjection.inputFormat must be 'float32x2'"
    );
    if (props.inverseProjection) {
      if (props.inverseProjection.precision === 'double-single') {
        // The high word of an absolute double-single origin must be finite float32.
        validateFiniteNumber(this.id, 'sourceGrid.affine[2]', props.sourceGrid.affine[2]);
        validateFiniteNumber(this.id, 'sourceGrid.affine[5]', props.sourceGrid.affine[5]);
      }
      const shader = props.inverseProjection.getShader({namespace: 'raster'});
      // The kernel builder declares/binds storage from its binding list. Keep the published
      // callable unchanged, supplying its one parameter declaration through that same list.
      const declaration = `@group(0) @binding(auto) var<storage, read> ${shader.bindingName}: array<u32>;`;
      validateRequirement(
        this.id,
        shader.source.includes(declaration),
        'inverseProjection.getShader() must declare its uint32 parameter storage binding'
      );
      this.projectionShader = {...shader, source: shader.source.replace(declaration, '')};
    } else {
      validateRequirement(
        this.id,
        JSON.stringify(props.sourceGrid.coordinateReferenceSystem ?? null) ===
          JSON.stringify(props.destinationGrid.coordinateReferenceSystem ?? null),
        'sourceGrid and destinationGrid must share a coordinate reference system unless inverseProjection is given'
      );
    }
    getGPURasterReprojectionParameterValues(props);
  }
  destroy(): void {
    this.projectionParameters?.destroy();
    this.projectionParameters = undefined;
    this.affineParameters?.destroy();
    this.affineParameters = undefined;
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const storageProps = this.props;
    const storageCellCount = storageProps.sourceGrid.width * storageProps.sourceGrid.height;
    validateRasterStorageBindings(
      graph,
      this.id,
      [
        ...getRasterBandViews(storageProps.input),
        storageProps.parameters,
        ...getRasterPreflightOutputViews(storageProps.output)
      ],
      {
        'canonical band planes': storageCellCount * 4,
        'projection parameters': storageProps.inverseProjection
          ? storageProps.inverseProjection.packParameters().byteLength
          : 0,
        'affine parameters': GPU_RASTER_REPROJECTION_PARAMETER_LENGTH * 4
      }
    );

    // Projection parameters belong to one graph; rebuilding requires a new contributor.
    validateRequirement(
      this.id,
      !this.projectionRegistered,
      'getCommandNodes can run once; construct a new contributor for another graph'
    );
    const {id, props} = this;
    const source = props.sourceGrid,
      destination = props.destinationGrid;
    const count = destination.width * destination.height;
    validateRasterBandBelongsToGraph(id, graph, props.input, []);
    validateRasterExpansionGraph(id, graph, [
      ...getRasterBandViews(props.input),
      ...(props.parameters ? [props.parameters] : []),
      ...Object.values(props.output)
    ]);
    // Canonicalization allocates one float32 and one uint32 source-sized binding. Check before
    // creating any nodes; the fused sampler has no row-proportional destination scratch.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'sourceGrid canonical band',
      source.width * source.height * Float32Array.BYTES_PER_ELEMENT,
      'reduce sourceGrid dimensions'
    );
    const shader = this.projectionShader;
    const projectionValues = shader ? props.inverseProjection!.packParameters() : undefined;
    if (projectionValues) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'inverseProjection parameters',
        projectionValues.byteLength,
        'reduce the inverseProjection program size'
      );
    }
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'affine parameters',
      GPU_RASTER_REPROJECTION_PARAMETER_LENGTH * Float32Array.BYTES_PER_ELEMENT,
      'use a device with at least 80 bytes per storage binding'
    );
    let parameterView: GraphDataView<'uint32'> | undefined;
    if (shader && projectionValues) {
      this.projectionParameters = new GPUParameterBuffer(graph.device, {
        id: `${id}-projection-parameters`,
        format: 'uint32',
        length: projectionValues.length,
        values: projectionValues
      });
      parameterView = this.projectionParameters.importToGraph(graph);
      this.projectionRegistered = true;
    }
    const band = getRasterBufferBandNodes(graph, id, props.input, source, true);
    let affineParameters = props.parameters;
    if (!affineParameters) {
      this.affineParameters ??= new GPUParameterBuffer(graph.device, {
        id: `${id}-affine-parameters`,
        format: 'float32',
        length: GPU_RASTER_REPROJECTION_PARAMETER_LENGTH,
        values: getGPURasterReprojectionParameterValues(props)
      });
      affineParameters = this.affineParameters.importToGraph(graph);
    }
    const destinationPosition = `let destinationColumn=f32(index%${destination.width}u)${shader ? `+${destination.pixelInterpretation === 'area' ? '0.5' : '0.0'}` : '-readAffine(18u)'};
let destinationRow=f32(index/${destination.width}u)${shader ? `+${destination.pixelInterpretation === 'area' ? '0.5' : '0.0'}` : '-readAffine(19u)'};
let destinationPosition=vec2f(readAffine(0u)*destinationColumn+readAffine(1u)*destinationRow+readAffine(4u), readAffine(2u)*destinationColumn+readAffine(3u)*destinationRow+readAffine(5u));`;
    const projectedPosition = shader
      ? `let projected=${shader.entryPoint}(destinationPosition, 1u);
${
  props.inverseProjection!.precision === 'double-single'
    ? `let relativeX=sub_fp64(projected.position.xy, vec2f(readAffine(13u), readAffine(14u)));
let relativeY=sub_fp64(projected.position.zw, vec2f(readAffine(15u), readAffine(16u)));
let position=vec2f(relativeX.x+relativeX.y, relativeY.x+relativeY.y);`
    : `let position=projected.position+vec2f(readAffine(11u), readAffine(12u));`
}`
      : 'let position=destinationPosition;';
    return [
      ...band.nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-sample`,
        operation: 'GPURasterReprojection',
        invocationCount: count,
        bindings: [
          ...(shader && parameterView
            ? [
                {
                  name: shader.bindingName,
                  view: parameterView,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'affine', view: affineParameters, type: 'f32', access: 'read'},
          {name: 'source', view: band.band.storage.values, type: 'f32', access: 'read'},
          {name: 'sourceValidity', view: band.band.validity!, type: 'u32', access: 'read'},
          {name: 'values', view: props.output.values, type: 'f32', access: 'read_write'},
          {name: 'validity', view: props.output.validity, type: 'u32', access: 'read_write'}
        ],
        declarations: `${shader?.source ?? ''}\n${IS_FINITE_VALUE_WGSL}\nfn readAffine(slot: u32) -> f32 { return affine[affineOffset + slot]; }`,
        modules: shader?.modules,
        defines: shader?.defines,
        body: `${destinationPosition}\n${projectedPosition}\n${shader ? `let column=readAffine(6u)*position.x+readAffine(7u)*position.y-${source.pixelInterpretation === 'area' ? '0.5' : '0.0'};let row=readAffine(8u)*position.x+readAffine(9u)*position.y-${source.pixelInterpretation === 'area' ? '0.5' : '0.0'};` : 'let column=position.x;let row=position.y;'}
var valid=readAffine(17u)==${GPU_RASTER_REPROJECTION_PARAMETER_VERSION}.0&&isFiniteValue(column)&&isFiniteValue(row)${shader ? '&&projected.valid!=0u' : ''};var total=0.0;var weightSum=0.0;
${props.method === 'nearest' ? `let sampleColumn=i32(floor(column))+select(0,1,fract(column)>=0.5);let sampleRow=i32(floor(row))+select(0,1,fract(row)>=0.5);if(sampleColumn<0||sampleRow<0||sampleColumn>=${source.width}||sampleRow>=${source.height}){valid=false;}else{let sampleIndex=u32(sampleRow)*${source.width}u+u32(sampleColumn);valid=valid&&sourceValidity[sourceValidityOffset+sampleIndex]!=0u;total=source[sourceOffset+sampleIndex];weightSum=1.0;}` : `let baseColumn=i32(floor(column));let baseRow=i32(floor(row));let fraction=vec2<f32>(fract(column),fract(row));for(var rowOffset=0;rowOffset<2;rowOffset++){for(var columnOffset=0;columnOffset<2;columnOffset++){let weight=select(1.0-fraction.x,fraction.x,columnOffset==1)*select(1.0-fraction.y,fraction.y,rowOffset==1);if(weight==0.0){continue;}let sampleColumn=baseColumn+columnOffset;let sampleRow=baseRow+rowOffset;var sampleValid=sampleColumn>=0&&sampleRow>=0&&sampleColumn<${source.width}&&sampleRow<${source.height};if(sampleValid){let sampleIndex=u32(sampleRow)*${source.width}u+u32(sampleColumn);sampleValid=sourceValidity[sourceValidityOffset+sampleIndex]!=0u;if(sampleValid){total+=weight*source[sourceOffset+sampleIndex];weightSum+=weight;}}${props.noDataPolicy === 'renormalize' ? '' : 'valid=valid&&sampleValid;'}}}`}
valid=valid&&weightSum>0.0;let result=total/select(1.0,weightSum,weightSum>0.0);valid=valid&&isFiniteValue(result);values[valuesOffset+index]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),result,valid);validity[validityOffset+index]=u32(valid);`
      })
    ];
  }
}
