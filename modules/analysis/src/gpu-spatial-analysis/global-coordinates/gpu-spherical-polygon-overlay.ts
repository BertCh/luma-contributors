// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {vec3} from '@math.gl/core';
import {assertUnitVector} from './assert-unit-vector';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUPolygonOverlay,
  type GPUPolygonOverlayProps,
  type GPUPolygonOverlayOutput
} from '../polygon-overlay/gpu-polygon-overlay';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateSinglePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_OVERLAY_MINIMUM_COSINE,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
} from './spherical-limits';

/** Value of `output.complement` when the result is the polygon region itself. */
export const GPU_SPHERICAL_POLYGON_OVERLAY_COMPLEMENT_NONE = 0;
/** Value of `output.complement` when the result is the sphere minus the assembled polygon region. */
export const GPU_SPHERICAL_POLYGON_OVERLAY_COMPLEMENT_SPHERE = 1;

export type GPUSphericalPolygonOperand = Omit<GPUPolygonOverlayProps['left'], 'positions'> & {
  positions: GraphDataView<'float32x3'>;
};
export type GPUSphericalPolygonOverlayProps = {
  id?: string;
  /**
   * Coordinate space of the unit vectors: longitude-latitude (their embedding on the sphere) with
   * great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.
   * The planar overlay inside the chart is an implementation detail and takes no context.
   */
  spatialContext?: GPUSpatialContext;
  left: GPUSphericalPolygonOperand;
  right: GPUSphericalPolygonOperand;
  operation: 'intersection' | 'union' | 'difference' | 'symmetric-difference';
  leftComplement?: boolean;
  rightComplement?: boolean;
  /** Unit vector inside a common hemisphere containing every supplied boundary vertex. */
  chartCenter: readonly [number, number, number];
  /** Reject vertices closer to the chart horizon; defaults to 0.01. */
  minimumCosine?: number;
  /** Gnomonic chart units. Overlay is exact on its documented snap lattice, not exact spherical predicates. */
  vertexTolerance: number;
  capacity: GPUPolygonOverlayProps['capacity'];
  output: Omit<GPUPolygonOverlayOutput, 'geometry'> & {
    geometry: Omit<GPUPolygonOverlayOutput['geometry'], 'positions'> & {
      positions: GraphDataView<'float32x3'>;
    };
    /**
     * One GPU word: {@link GPU_SPHERICAL_POLYGON_OVERLAY_COMPLEMENT_SPHERE} (1) means the sphere
     * minus the assembled polygon region, {@link GPU_SPHERICAL_POLYGON_OVERLAY_COMPLEMENT_NONE} (0)
     * the region itself.
     */
    complement: GraphDataView<'uint32'>;
  };
};

/**
 * General nonconvex shell/hole Boolean overlay in a supplied spherical hemisphere chart.
 * Gnomonic projection maps minor great-circle arcs to straight lines; the existing snap-rounded
 * arrangement handles noding, coincident edges and ring assembly. Complements use Boolean duality,
 * so their unbounded exterior is represented explicitly rather than clipped at the chart horizon.
 */
export class GPUSphericalPolygonOverlay implements GPUCommandNodeProducer {
  readonly id: string;
  /** Validated coordinate space of the unit vectors. */
  readonly spatialContext: GPUSpatialContext;
  readonly complement: boolean;
  private readonly planarOperation: GPUPolygonOverlayProps['operation'];
  private readonly swap: boolean;
  private readonly horizontal: number[];
  private readonly vertical: number[];
  constructor(readonly props: GPUSphericalPolygonOverlayProps) {
    this.id = props.id ?? 'spherical-polygon-overlay';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
      GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
    );
    assertUnitVector(props.chartCenter, this.id, 'chartCenter');
    validateRequirement(
      this.id,
      Number.isFinite(props.minimumCosine ?? 0.01) &&
        (props.minimumCosine ?? 0.01) >= GPU_SPHERICAL_OVERLAY_MINIMUM_COSINE &&
        (props.minimumCosine ?? 0.01) < 1,
      `minimumCosine must be a finite number in [${GPU_SPHERICAL_OVERLAY_MINIMUM_COSINE}, 1)`
    );
    for (const view of [
      props.left.positions,
      props.right.positions,
      props.output.geometry.positions
    ])
      validateSinglePackedView(view, ['float32x3'], this.id);
    validateSinglePackedView(props.output.complement, ['uint32'], this.id);
    validateRequirement(
      this.id,
      props.output.complement.length === 1,
      'output.complement must hold one entry'
    );
    const geometry = props.output.geometry;
    const outputs = [
      geometry.positions,
      geometry.ringOffsets,
      geometry.polygonOffsets,
      geometry.featureOffsets,
      ...Object.values(props.output.status),
      props.output.boundary.endpoints,
      props.output.boundary.operandIds,
      props.output.boundary.sourceFeatureIds,
      ...Object.values(props.output.boundary.status)
    ];
    const inputs = [
      props.left.positions,
      props.left.ringOffsets,
      props.left.polygonOffsets,
      props.left.featureOffsets,
      props.right.positions,
      props.right.ringOffsets,
      props.right.polygonOffsets,
      props.right.featureOffsets
    ];
    validateGraphOutputsDisjointFromInputs(this.id, [...outputs, props.output.complement], inputs);
    // This final writer must never overwrite geometry, provenance or an earlier failure flag.
    validateGraphOutputsDisjointFromInputs(this.id, [props.output.complement], outputs);
    const [centerX, centerY, centerZ] = props.chartCenter;
    const center = [centerX, centerY, centerZ];
    const axis = Math.abs(centerZ) < 0.9 ? [0, 0, 1] : [0, 1, 0];
    const horizontal = vec3.cross([0, 0, 0], axis, center);
    vec3.normalize(horizontal, horizontal);
    this.horizontal = horizontal;
    this.vertical = vec3.cross([0, 0, 0], center, horizontal);
    const truth = (left: boolean, right: boolean) => {
      const first = left !== Boolean(props.leftComplement);
      const second = right !== Boolean(props.rightComplement);
      return props.operation === 'intersection'
        ? first && second
        : props.operation === 'union'
          ? first || second
          : props.operation === 'difference'
            ? first && !second
            : first !== second;
    };
    this.complement = truth(false, false);
    const mask = [truth(false, true), truth(true, false), truth(true, true)]
      .map(value => Number(value !== this.complement))
      .join('');
    this.swap = mask === '100';
    this.planarOperation =
      mask === '001'
        ? 'intersection'
        : mask === '111'
          ? 'union'
          : mask === '110'
            ? 'symmetric-difference'
            : 'difference';
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.left.positions,
      props.right.positions,
      props.output.geometry.positions,
      props.output.complement
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const frame = `const CHART_CENTER = vec3f(${props.chartCenter.join(',')}); const CHART_HORIZONTAL = vec3f(${this.horizontal.join(',')}); const CHART_VERTICAL = vec3f(${this.vertical.join(',')});`;
    const project = (
      name: string,
      operand: GPUSphericalPolygonOperand
    ): GPUPolygonOverlayProps['left'] => {
      const positions = createTransientView(
        graph,
        `${id}-${name}-chart`,
        'float32x2',
        operand.positions.length
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-${name}-project`,
          operation: 'GPUSphericalPolygonOverlay',
          variant: 'gnomonic-project',
          invocationCount: operand.positions.length,
          bindings: [
            {name: 'input', view: operand.positions, type: 'f32', access: 'read'},
            {name: 'output', view: positions, type: 'f32', access: 'read_write'}
          ],
          declarations: frame,
          body: `let offset = inputOffset + index * 3u; let position = vec3f(input[offset],input[offset+1u],input[offset+2u]); let cosine = dot(position,CHART_CENTER); let squared = dot(position,position);
var projected = vec2f(bitcast<f32>(0x7fc00000u | (index & 0u)));
if ((bitcast<u32>(squared) & 0x7f800000u) != 0x7f800000u && abs(squared-1.0) <= 1e-4 && cosine >= ${props.minimumCosine ?? 0.01}) { projected = vec2f(dot(position,CHART_HORIZONTAL),dot(position,CHART_VERTICAL))/cosine; }
output[outputOffset+index*2u]=projected.x; output[outputOffset+index*2u+1u]=projected.y;`
        })
      );
      return {...operand, positions};
    };
    const left = project('left', props.left);
    const right = project('right', props.right);
    const positions = createTransientView(
      graph,
      `${id}-result-chart`,
      'float32x2',
      props.output.geometry.positions.length
    );
    nodes.push(
      ...new GPUPolygonOverlay({
        id: `${id}-arrangement`,
        left: this.swap ? right : left,
        right: this.swap ? left : right,
        operation: this.planarOperation,
        vertexTolerance: props.vertexTolerance,
        capacity: props.capacity,
        output: {...props.output, geometry: {...props.output.geometry, positions}}
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-lift`,
        operation: 'GPUSphericalPolygonOverlay',
        variant: 'gnomonic-lift',
        invocationCount: positions.length,
        bindings: [
          {name: 'input', view: positions, type: 'f32', access: 'read'},
          {name: 'output', view: props.output.geometry.positions, type: 'f32', access: 'read_write'}
        ],
        declarations: frame,
        body: `let point=normalize(CHART_CENTER+input[inputOffset+index*2u]*CHART_HORIZONTAL+input[inputOffset+index*2u+1u]*CHART_VERTICAL); output[outputOffset+index*3u]=point.x; output[outputOffset+index*3u+1u]=point.y; output[outputOffset+index*3u+2u]=point.z;`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-complement`,
        operation: 'GPUSphericalPolygonOverlay',
        variant: 'complement',
        invocationCount: 1,
        bindings: [
          {name: 'complement', view: props.output.complement, type: 'u32', access: 'read_write'}
        ],
        body: `complement[complementOffset]=${Number(this.complement)}u;`
      })
    );
    if (this.swap)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-operand-provenance`,
          operation: 'GPUSphericalPolygonOverlay',
          variant: 'operand-provenance',
          invocationCount: props.output.boundary.operandIds.length,
          bindings: [
            {
              name: 'operands',
              view: props.output.boundary.operandIds,
              type: 'u32',
              access: 'read_write'
            },
            {name: 'count', view: props.output.boundary.status.count, type: 'u32', access: 'read'}
          ],
          body: 'if(index<count[countOffset]) {operands[operandsOffset+index]=1u-operands[operandsOffset+index];}'
        })
      );
    return nodes;
  }
}
