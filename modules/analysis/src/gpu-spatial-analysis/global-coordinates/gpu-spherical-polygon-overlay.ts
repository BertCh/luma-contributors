// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSphericalOverlayOperation} from './spherical-overlay-options';
import {vec3} from '@math.gl/core';
import {createOverlaySourceIdentityNodes} from '../polygon-overlay/overlay-source-identities';
import {validateUnitVector} from './assert-unit-vector';
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
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {validateOutlineOutputPasses} from '../outline-geometry/validate-outline-output-passes';
import {validateGeometryReadWriteViews} from '../geometry-edit/geometry-storage-validation';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateOverlayScratchByteLengths,
  getOverlayOutputViews
} from '../polygon-overlay/overlay-storage-validation';
import {
  validateSinglePackedView,
  getGraphOutputViews,
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

const RESULT_STATUS_FIELDS = [
  'count',
  'overflow',
  'candidateOverflow',
  'uncertainCount',
  'ringCount',
  'requiredRingCount',
  'nodedSegmentRequiredCount'
];
const BOUNDARY_STATUS_FIELDS = ['count', 'requiredCount', 'overflow', 'candidateOverflow'];
const RETIRED_STATUS_FIELDS = [
  'requiredCount',
  'iterationCount',
  'converged',
  'iterationLimitReached',
  'invalidCount',
  'approximation',
  'uncertainCount'
];

function snapshotSphericalOverlayFields<Properties extends object>(
  properties: Properties,
  declaredFields: readonly string[]
): Properties {
  return snapshotConstructionFields(properties, declaredFields as (keyof Properties)[]);
}

function snapshotSphericalOverlayStatus<Properties extends object>(
  properties: Properties,
  supportedFields: readonly string[]
): Properties {
  // Status is the only descriptor with rejected output keys. Inspect supplied own keys and
  // explicitly known retired names through the prototype chain, without touching unrelated
  // getters on ordinary input/metadata descriptors.
  const fields = new Set([
    ...supportedFields,
    ...RETIRED_STATUS_FIELDS,
    ...Object.getOwnPropertyNames(properties)
  ]);
  return snapshotConstructionFields(properties, [...fields] as (keyof Properties)[]);
}

function validateSphericalOverlayStatus(
  id: string,
  name: string,
  status: object,
  supportedFields: readonly string[]
): void {
  for (const [field, view] of Object.entries(status)) {
    const reason =
      field === 'requiredCount'
        ? 'only the unclamped ring requirement is known after rings are dropped'
        : field === 'iterationCount' || field === 'converged' || field === 'iterationLimitReached'
          ? 'the arrangement has fixed passes and no solver stopping criterion or iteration budget'
          : field === 'invalidCount'
            ? 'input rejection is reported as incomplete noding without a separate rejected-row count'
            : field === 'approximation'
              ? 'exactness is defined on the chart snap lattice'
              : field === 'uncertainCount'
                ? 'uncertified predicates are counted on the result rather than the selected boundary'
                : `this port publishes only ${supportedFields.join(', ')}`;
    const alternative =
      field === 'requiredCount'
        ? 'output.status.requiredRingCount for ring sizing'
        : field === 'uncertainCount'
          ? 'output.status.uncertainCount for uncertified segment pairs'
          : field === 'invalidCount'
            ? 'output.status.candidateOverflow for incomplete input'
            : field === 'approximation'
              ? 'vertexTolerance to control the chart snap lattice'
              : 'output.status.overflow and output.status.candidateOverflow for completeness';
    validateRequirement(
      id,
      view === undefined || supportedFields.includes(field),
      `${name}.${field} is unsupported because ${reason}; use ${alternative}`
    );
  }
}

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
 * The fixed-pass arrangement has no solver iteration or convergence outputs. Unsupported status
 * fields are refused at construction rather than silently leaving caller buffers unwritten.
 * The result publisher requires its count/overflow/candidateOverflow buffers separate from
 * geometry.featureOffsets and boundary overflow/candidateOverflow, even for disjoint slices.
 * Assembly completeness also separates boundary count reads from result candidateOverflow.
 * Co-written result, boundary and optional ring status fields need disjoint aligned bindings.
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
  readonly props: GPUSphericalPolygonOverlayProps;
  constructor(props: GPUSphericalPolygonOverlayProps) {
    // Snapshot nested descriptors before validating so a caller cannot add an unvalidated output
    // or replace a validated view before deferred graph construction. Buffer contents stay mutable.
    props = snapshotSphericalOverlayFields(props, [
      'id',
      'spatialContext',
      'left',
      'right',
      'operation',
      'leftComplement',
      'rightComplement',
      'chartCenter',
      'minimumCosine',
      'vertexTolerance',
      'capacity',
      'output'
    ]);
    const output = snapshotSphericalOverlayFields(props.output, [
      'geometry',
      'complement',
      'status',
      'boundary',
      'featureSources'
    ]);
    const boundary = snapshotSphericalOverlayFields(output.boundary, [
      'endpoints',
      'operandIds',
      'sourceFeatureIds',
      'status'
    ]);
    const operandFields = [
      'kind',
      'positions',
      'ringOffsets',
      'polygonOffsets',
      'featureOffsets',
      'sourceIds'
    ];
    props = Object.freeze({
      ...props,
      chartCenter: Object.freeze([...props.chartCenter]) as readonly [number, number, number],
      capacity: snapshotSphericalOverlayFields(props.capacity, ['intersections', 'nodedSegments']),
      left: snapshotSphericalOverlayFields(props.left, operandFields),
      right: snapshotSphericalOverlayFields(props.right, operandFields),
      output: Object.freeze({
        ...output,
        geometry: snapshotSphericalOverlayFields(output.geometry, operandFields),
        status: snapshotSphericalOverlayStatus(output.status, RESULT_STATUS_FIELDS),
        boundary: Object.freeze({
          ...boundary,
          status: snapshotSphericalOverlayStatus(boundary.status, BOUNDARY_STATUS_FIELDS)
        }),
        ...(output.featureSources
          ? {
              featureSources: snapshotSphericalOverlayFields(output.featureSources, [
                'sourceRows',
                'operandIds',
                'mixed'
              ])
            }
          : {})
      })
    });
    this.id = props.id ?? 'spherical-polygon-overlay';
    validateSphericalOverlayOperation(this.id, props.operation);
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      snapshotConstructionSpatialContext(
        props.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT
      ),
      GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
    );
    props = Object.freeze({...props, spatialContext: this.spatialContext});
    this.props = props;
    validateUnitVector(props.chartCenter, this.id, 'chartCenter');
    validateRequirement(
      this.id,
      Number.isFinite(props.minimumCosine ?? 0.01) &&
        (props.minimumCosine ?? 0.01) >= GPU_SPHERICAL_OVERLAY_MINIMUM_COSINE &&
        (props.minimumCosine ?? 0.01) < 1,
      `minimumCosine must be a finite number in [${GPU_SPHERICAL_OVERLAY_MINIMUM_COSINE}, 1)`
    );
    validateSphericalOverlayStatus(
      this.id,
      'output.status',
      props.output.status,
      RESULT_STATUS_FIELDS
    );
    validateSphericalOverlayStatus(
      this.id,
      'output.boundary.status',
      props.output.boundary.status,
      BOUNDARY_STATUS_FIELDS
    );
    for (const [name, operand] of [
      ['left', props.left],
      ['right', props.right]
    ] as const) {
      if (operand.sourceIds) {
        validateSinglePackedView(operand.sourceIds, ['uint32'], this.id);
        validateRequirement(
          this.id,
          operand.sourceIds.length >= operand.featureOffsets.length - 1,
          `${name}.sourceIds must hold at least ${operand.featureOffsets.length - 1} feature identities`
        );
      }
    }
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
    const outputViews = {
      'output.geometry.positions': geometry.positions,
      'output.geometry.ringOffsets': geometry.ringOffsets,
      'output.geometry.polygonOffsets': geometry.polygonOffsets,
      'output.geometry.featureOffsets': geometry.featureOffsets,
      'output.complement': props.output.complement,
      'output.boundary.endpoints': props.output.boundary.endpoints,
      'output.boundary.operandIds': props.output.boundary.operandIds,
      'output.boundary.sourceFeatureIds': props.output.boundary.sourceFeatureIds,
      ...Object.fromEntries(
        Object.entries(props.output.status).map(([name, view]) => [`output.status.${name}`, view])
      ),
      ...Object.fromEntries(
        Object.entries(props.output.boundary.status).map(([name, view]) => [
          `output.boundary.status.${name}`,
          view
        ])
      ),
      ...Object.fromEntries(
        Object.entries(props.output.featureSources ?? {}).map(([name, view]) => [
          `output.featureSources.${name}`,
          view
        ])
      )
    };
    const inputs = [
      props.left.positions,
      props.left.ringOffsets,
      props.left.polygonOffsets,
      props.left.featureOffsets,
      props.right.positions,
      props.right.ringOffsets,
      props.right.polygonOffsets,
      props.right.featureOffsets,
      ...(props.left.sourceIds ? [props.left.sourceIds] : []),
      ...(props.right.sourceIds ? [props.right.sourceIds] : [])
    ];
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(outputViews), inputs);
    // Compare actual public XYZ ranges, rather than the temporary planar chart positions.
    // Separate dispatches may safely write disjoint columns of one physical allocation.
    validateOutlineOutputPasses(this.id, outputViews, [
      ['output.status.count', 'output.status.overflow', 'output.status.candidateOverflow'],
      [
        'output.boundary.status.count',
        'output.boundary.status.requiredCount',
        'output.boundary.status.overflow',
        'output.boundary.status.candidateOverflow'
      ],
      ['output.status.ringCount', 'output.status.requiredRingCount']
    ]);
    // The nested planar publisher reads boundary loss and feature offsets while
    // writing result status. Check its physical usage before creating chart streams.
    validateGeometryReadWriteViews(
      this.id,
      [
        geometry.featureOffsets,
        props.output.boundary.status.overflow,
        props.output.boundary.status.candidateOverflow
      ],
      [
        props.output.status.count,
        props.output.status.overflow,
        props.output.status.candidateOverflow
      ]
    );
    // Assembly completeness atomically augments the result loss flag while reading
    // the retained boundary count, in a separate dispatch from the result publisher.
    validateGeometryReadWriteViews(
      this.id,
      [props.output.boundary.status.count],
      [props.output.status.candidateOverflow]
    );
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
    if (this.swap)
      validateGeometryReadWriteViews(
        this.id,
        [props.output.boundary.status.count],
        [props.output.boundary.operandIds]
      );
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
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [
          ...getGraphOutputViews(props.left),
          ...getGraphOutputViews(props.right),
          ...getOverlayOutputViews({
            ...props.output,
            geometry: {
              ...props.output.geometry,
              positions: props.output.geometry.positions as unknown as GraphDataView<'float32x2'>
            }
          }),
          props.output.complement
        ].map((view, index) => [`binding.${index}`, view])
      )
    );
    // Chart positions are two words per vertex, bounded by the three-word caller views.
    // Validate the entire nested planar arrangement before creating either chart stream.
    const inputRingCount = props.left.ringOffsets.length + props.right.ringOffsets.length - 2;
    validateOverlayScratchByteLengths(graph, id, {
      expandedVertexCount:
        props.left.positions.length + props.right.positions.length + inputRingCount,
      inputRingCount,
      intersections: props.capacity.intersections,
      nodedSegments: props.capacity.nodedSegments,
      boundaryCapacity: props.output.boundary.endpoints.length,
      outputRingCapacity: props.output.geometry.ringOffsets.length - 1,
      outputVertexCapacity: props.output.geometry.positions.length
    });
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
      // Internal provenance must remain row-valued even when the shared planar publisher also
      // supports caller identities. The outer publisher restores identities only after reduction.
      return {...operand, sourceIds: undefined, positions};
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
    if (this.swap) {
      const restoreOperands = createWGSLKernelNode<Parameters>(graph, {
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
      });
      // Representative ordering is in the caller's operand space. Restore edge ownership before
      // the planar feature reducer reads it; flipping only its final representative loses the
      // correct source row on mixed-owner features.
      const reductionIndex = nodes.findIndex(
        node => node.id === `${id}-arrangement-reduce-feature-sources`
      );
      nodes.splice(reductionIndex < 0 ? nodes.length : reductionIndex, 0, restoreOperands);
    }
    // The planar arrangement's packed provenance contains feature rows. Keep those rows until
    // representative reduction is complete, then publish caller identities at the outer port.
    nodes.push(
      ...createOverlaySourceIdentityNodes(graph, {
        id,
        operation: 'GPUSphericalPolygonOverlay',
        rows: props.output.boundary.sourceFeatureIds,
        operands: props.output.boundary.operandIds,
        count: props.output.boundary.status.count,
        leftIds: props.left.sourceIds,
        rightIds: props.right.sourceIds
      })
    );
    return nodes;
  }
}
