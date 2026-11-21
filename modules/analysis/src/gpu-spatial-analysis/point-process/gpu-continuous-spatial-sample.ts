// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validatePointProcessView} from './point-process-utils';
import {getSamplingReplayParameters, validateSamplingSeed} from './sampling-seed';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotContinuousSampleFrame} from './continuous-sample-frame-construction';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {PHILOX_PURPOSES, PHILOX_WGSL} from '../../analysis-core/kernels/philox-random';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Most frame rows: each lattice candidate scans the frame serially inside one wave invocation. */
const MAXIMUM_FRAME_ROW_COUNT = 4096;
/** Most waves: each wave is one serial invocation and one status row. */
const MAXIMUM_WAVE_COUNT = 4096;
/** Most output slots per wave, bounding resident outputs of waves times capacity rows. */
const MAXIMUM_WAVE_CAPACITY = 65536;
/** Default and largest lattice candidates examined per wave, bounding the serial loop. */
const DEFAULT_MAXIMUM_CANDIDATES = 65536;
const MAXIMUM_CANDIDATE_COUNT = 1048576;
/** Most fixed legacy sites: each candidate is tested against all of them. */
const MAXIMUM_LEGACY_SITE_COUNT = 256;
/**
 * Most serial inner-loop iterations of one wave invocation (see `getWaveWork`). Each wave is a
 * single invocation, so without this bound the separate maxima above allow about 1.7e9 iterations
 * and a lost device. 2^24 is the scalar work budget the bounded reference kernels already use
 * (`MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK` and the `GPUPointPatternEnvelope` rank comparisons, both
 * 16777216). In this family `GPUInhomogeneousPointPattern` bounds one invocation's overlap loop at
 * 1048576 edge pairs, each a triangle clip costing several of this loop's point-in-triangle or
 * segment steps, which is of the same order. It admits the default 65536 candidates against
 * about 250 frame primitives without legacy sites (more with a smaller budget or a smaller area
 * lattice).
 */
const MAXIMUM_WAVE_WORK = 16777216;

export type GPUContinuousSpatialFrame =
  | {
      /** Connected polyline. Separation is along its cumulative arc length, including on folded paths. */
      type: 'line';
      positions: GraphDataView<'float32x2'>;
      /** Fixed legacy locations in cumulative arc-length units. */
      legacySites?: GraphDataView<'float32'>;
    }
  | {
      /** Union of nondegenerate triangles, three positions each. Holes are gaps in this supplied frame. */
      type: 'area';
      positions: GraphDataView<'float32x2'>;
      bounds: readonly [number, number, number, number];
      /** Fixed legacy xy sites inside the triangle union. */
      legacySites?: GraphDataView<'float32x2'>;
    };

export type GPUContinuousSpatialSampleProps = {
  id?: string;
  /** Planar only; longitude-latitude is rejected. Default planar. */
  spatialContext?: GPUSpatialContext;
  frame: GPUContinuousSpatialFrame;
  /** Positive spacing; Euclidean for area, arc length for line. Legacy exclusion uses the same metric. */
  spacing: number;
  /** Scalar seed, required only without parameters. Resident words override this hint. */
  seed?: number;
  /** First stream index; independent replacement waves increment this value. */
  replicate?: number;
  /** Primary wave followed by independently randomized complete replacement waves; default 1. */
  waves?: number;
  /** Fixed output slots per wave, including legacy rows. */
  capacity: number;
  /**
   * Maximum examined lattice candidates per wave, default 65536. An area frame must cover its whole
   * `bounds` lattice, so a smaller value throws at construction; a line frame reports flag 4. The
   * serial work of one wave, which grows with this budget, is bounded at 2^24 iterations.
   */
  maximumCandidates?: number;
  /** Resident [seed, first replicate] overrides; permits replay and changed inputs without recompilation. */
  parameters?: GraphDataView<'uint32'>;
  output: {
    positions: GraphDataView<'float32x2'>;
    /** Inverse first-order sampling intensity: spacing for lines, spacing² for areas. Legacy rows have zero weight. */
    designWeights: GraphDataView<'float32'>;
    /** Per slot [source segment/triangle or legacy ID, stream replicate, candidate ordinal, role]. Role 0 primary, 1 replacement, 2 legacy. */
    provenance: GraphDataView<'uint32x4'>;
    /** Per wave [written, required so far, random accepted so far, flags]. Flags: 1 invalid, 2 capacity, 4 candidate budget. */
    status: GraphDataView<'uint32x4'>;
  };
};

/**
 * Continuous random-shift probability sampling. A uniform phase translates an arc-length lattice
 * or a planar square lattice; every eligible infinitesimal frame element has the same known
 * first-order sampling intensity. Fixed legacy exclusion does not alter that intensity outside
 * excluded neighborhoods. Count is random; this is not fixed-size GRTS or an iid design.
 * Replacement waves are complete independent designs, not an assertion that arbitrary selective
 * substitution or response-dependent truncation preserves design weights. No CPU sampling occurs.
 */
export class GPUContinuousSpatialSample implements GPUCommandNodeProducer {
  readonly id: string;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;
  readonly samplingIntensity: number;
  readonly provenance: Readonly<{
    design: 'random-shift-lattice';
    frame: 'line' | 'area';
    separationMetric: 'arc-length' | 'euclidean';
    randomPhaseBits: 24;
    replacement: 'independent-complete-waves';
    generator: 'Philox4x32-10';
    parameterSource: 'constructor' | 'resident-view';
    effectiveSeedSource: 'scalar' | 'resident-parameters';
  }>;
  readonly props: GPUContinuousSpatialSampleProps;
  constructor(props: GPUContinuousSpatialSampleProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'frame',
      'spacing',
      'seed',
      'replicate',
      'waves',
      'capacity',
      'maximumCandidates',
      'parameters',
      'output'
    ]);
    const id = captured.id ?? 'continuous-spatial-sample';
    props = Object.freeze({
      ...captured,
      frame: snapshotContinuousSampleFrame(id, captured.frame),
      output: snapshotConstructionFields(captured.output, [
        'positions',
        'designWeights',
        'provenance',
        'status'
      ])
    });
    this.props = props;
    this.id = id;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    const {frame} = props;
    validateRequirement(
      id,
      ['line', 'area'].includes(frame.type),
      `frame.type must be line or area; received ${frame.type}`
    );
    validateSinglePackedView(frame.positions, ['float32x2'], id);
    validateCapacity(
      id,
      'frame.positions rows',
      frame.positions.length,
      frame.type === 'area' ? 3 : 2,
      MAXIMUM_FRAME_ROW_COUNT
    );
    validateRequirement(
      id,
      frame.type !== 'area' || frame.positions.length % 3 === 0,
      'area frame.positions must hold whole triangles (a multiple of 3 rows)'
    );
    validateRequirement(
      id,
      Number.isFinite(Math.fround(props.spacing)) &&
        Math.fround(props.spacing) > 0 &&
        Number.isFinite(Math.fround(props.spacing ** 2)) &&
        Math.fround(props.spacing ** 2) > 0,
      'spacing must be positive with a positive finite float32 square'
    );
    const waves = props.waves ?? 1;
    validateCapacity(id, 'waves', waves, 1, MAXIMUM_WAVE_COUNT);
    validateCapacity(id, 'capacity', props.capacity, 1, MAXIMUM_WAVE_CAPACITY);
    const maximumCandidates = props.maximumCandidates ?? DEFAULT_MAXIMUM_CANDIDATES;
    validateCapacity(id, 'maximumCandidates', maximumCandidates, 1, MAXIMUM_CANDIDATE_COUNT);
    validateSamplingSeed(id, props.seed, props.parameters !== undefined);
    validateCapacity(id, 'replicate', props.replicate ?? 0, 0, 0xffffffff - waves + 1);
    if (frame.legacySites) {
      validateSinglePackedView(
        frame.legacySites,
        [frame.type === 'line' ? 'float32' : 'float32x2'],
        id
      );
      validateCapacity(
        id,
        'frame.legacySites rows',
        frame.legacySites.length,
        0,
        MAXIMUM_LEGACY_SITE_COUNT
      );
    }
    // Line frames examine up to maximumCandidates; their arc length is per-frame data.
    let candidates = maximumCandidates;
    if (frame.type === 'area') {
      validateRequirement(
        id,
        frame.bounds.every(value => Number.isFinite(Math.fround(value))),
        'frame.bounds must be finite in float32'
      );
      validateRequirement(
        id,
        frame.bounds[2] > frame.bounds[0] && frame.bounds[3] > frame.bounds[1],
        'frame.bounds must have a positive extent on both axes'
      );
      const columns = Math.ceil((frame.bounds[2] - frame.bounds[0]) / props.spacing) + 1;
      const rows = Math.ceil((frame.bounds[3] - frame.bounds[1]) / props.spacing) + 1;
      validateRequirement(
        id,
        Number.isSafeInteger(columns * rows) && columns * rows < 0xffffffff,
        'frame.bounds divided by spacing must give fewer than 2^32 lattice cells'
      );
      // Bounds, spacing and the budget are compile-time, so a lattice the budget cannot cover
      // would always end in flag 4. Reject it here; flag 4 remains for line frames.
      candidates = columns * rows;
      validateRequirement(
        id,
        candidates <= maximumCandidates,
        `maximumCandidates (${maximumCandidates}) must be at least the area lattice of ${columns} x ${rows} = ${candidates} cells given by frame.bounds and spacing (largest maximumCandidates ${MAXIMUM_CANDIDATE_COUNT}); raise maximumCandidates or spacing`
      );
    }
    const waveWork = getWaveWork(
      frame.type,
      frame.positions.length,
      frame.legacySites?.length ?? 0,
      candidates,
      props.capacity
    );
    validateRequirement(
      id,
      waveWork <= MAXIMUM_WAVE_WORK,
      `serial work per wave, (candidates + legacy sites) * (frame primitives + legacy sites * legacy-site cost + 1) + frame rows + capacity = ${waveWork}, must be at most ${MAXIMUM_WAVE_WORK}; lower maximumCandidates, frame rows or legacy sites, or raise an area frame's spacing`
    );
    if (props.parameters) {
      validateSinglePackedView(props.parameters, ['uint32'], id);
      validateViewLength(id, 'parameters', props.parameters, 2);
    }
    for (const [path, view, format, expectedRows] of [
      ['output.positions', props.output.positions, 'float32x2', waves * props.capacity],
      ['output.designWeights', props.output.designWeights, 'float32', waves * props.capacity],
      ['output.provenance', props.output.provenance, 'uint32x4', waves * props.capacity],
      ['output.status', props.output.status, 'uint32x4', waves]
    ] as const)
      validatePointProcessView(id, path, view, [format], expectedRows);
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), [
      frame.positions,
      frame.legacySites,
      props.parameters
    ]);
    // Output fields are independent records and may not overwrite one another.
    validateRequirement(
      id,
      new Set(Object.values(props.output).map(view => view.buffer)).size === 4,
      'output views must use four distinct buffers'
    );
    this.samplingIntensity = 1 / (frame.type === 'line' ? props.spacing : props.spacing ** 2);
    this.provenance = Object.freeze({
      design: 'random-shift-lattice',
      frame: frame.type,
      separationMetric: frame.type === 'line' ? 'arc-length' : 'euclidean',
      randomPhaseBits: 24,
      replacement: 'independent-complete-waves',
      generator: 'Philox4x32-10',
      parameterSource: props.parameters ? 'resident-view' : 'constructor',
      effectiveSeedSource: props.parameters ? 'resident-parameters' : 'scalar'
    });
  }
  /** Record the uploaded resident words from this encoding, or the active scalar defaults. */
  getReplayParameters(words?: Uint32Array): Uint32Array {
    return getSamplingReplayParameters(
      this.id,
      this.props.seed,
      this.props.replicate,
      this.props.parameters !== undefined,
      words
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {frame} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      frame.positions,
      frame.legacySites,
      props.parameters,
      ...Object.values(props.output)
    ]);
    const legacyCount = frame.legacySites?.length ?? 0;
    validateGraphChunkBindingRanges(graph, id, {
      framePositions: frame.positions,
      legacySites: frame.legacySites,
      parameters: props.parameters,
      ...props.output
    });
    const area = frame.type === 'area';
    const columns = area ? Math.ceil((frame.bounds[2] - frame.bounds[0]) / props.spacing) + 1 : 0;
    const rows = area ? Math.ceil((frame.bounds[3] - frame.bounds[1]) / props.spacing) + 1 : 0;
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUContinuousSpatialSample',
        variant: frame.type,
        invocationCount: props.waves ?? 1,
        bindings: [
          {name: 'framePositions', view: frame.positions, type: 'f32', access: 'read'},
          ...(frame.legacySites
            ? [
                {
                  name: 'legacy',
                  view: frame.legacySites,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          ...(props.parameters
            ? [
                {
                  name: 'parameters',
                  view: props.parameters,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'positions', view: props.output.positions, type: 'f32', access: 'read_write'},
          {name: 'weights', view: props.output.designWeights, type: 'f32', access: 'read_write'},
          {name: 'provenance', view: props.output.provenance, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${PHILOX_WGSL}
const SAMPLE_SPACING: f32 = ${getWGSLFloatLiteral(props.spacing)};
const SAMPLE_CAPACITY: u32 = ${props.capacity}u;
${IS_FINITE_VALUE_WGSL}
fn samplePosition(row:u32)->vec2f{let offset=framePositionsOffset+2u*row;return vec2f(framePositions[offset],framePositions[offset+1u]);}
fn sampleCross(first:vec2f,second:vec2f)->f32{return first.x*second.y-first.y*second.x;}
fn samplePrimitive(point:vec2f)->u32 {
${
  area
    ? `for(var triangle=0u;triangle<${frame.positions.length / 3}u;triangle++){
 let first=samplePosition(triangle*3u);let second=samplePosition(triangle*3u+1u);let third=samplePosition(triangle*3u+2u);
 let orientation=sign(sampleCross(second-first,third-first));
 if(sampleCross(second-first,point-first)*orientation>=0.0&&sampleCross(third-second,point-second)*orientation>=0.0&&sampleCross(first-third,point-third)*orientation>=0.0){return triangle;}
}`
    : ''
}
 return 0xffffffffu;
}
fn sampleLine(measure:f32)->vec3f {
 var consumed=0.0;
 for(var segment=0u;segment<${frame.positions.length - 1}u;segment++){
  let first=samplePosition(segment);let second=samplePosition(segment+1u);let extent=distance(first,second);
  if(extent>0.0&&measure<=consumed+extent){return vec3f(mix(first,second,clamp((measure-consumed)/extent,0.0,1.0)),f32(segment));}consumed+=extent;
 }
 return vec3f(samplePosition(${frame.positions.length - 1}u),f32(${frame.positions.length - 2}u));
}
${frame.legacySites ? `fn sampleLegacy(row:u32)->vec3f{${area ? 'let offset=legacyOffset+row*2u;return vec3f(legacy[offset],legacy[offset+1u],0.0);' : 'return vec3f(sampleLine(legacy[legacyOffset+row]).xy,legacy[legacyOffset+row]);'}}` : ''}
fn sampleWrite(wave:u32,slot:u32,point:vec2f,weight:f32,source:u32,replicate:u32,ordinal:u32,role:u32){
 if(slot>=SAMPLE_CAPACITY){return;}let row=wave*SAMPLE_CAPACITY+slot;
 positions[positionsOffset+row*2u]=point.x;positions[positionsOffset+row*2u+1u]=point.y;weights[weightsOffset+row]=weight;
 provenance[provenanceOffset+row*4u]=source;provenance[provenanceOffset+row*4u+1u]=replicate;provenance[provenanceOffset+row*4u+2u]=ordinal;provenance[provenanceOffset+row*4u+3u]=role;
}`,
        body: `var failure=0u;var required=0u;var randomAccepted=0u;var measure=0.0;
let seed=${props.parameters ? 'parameters[parametersOffset]' : `${props.seed}u`};let initialReplicate=${props.parameters ? 'parameters[parametersOffset+1u]' : `${props.replicate ?? 0}u`};
if(initialReplicate>0xffffffffu-index){failure=1u;}
let replicate=initialReplicate+index;
let words=philox4x32(vec4u(replicate,0u,0u,0u),vec2u(seed,${PHILOX_PURPOSES.continuousSpatialSampling.id}u));
let phase=vec2f(philoxUnitFloat(words.x),philoxUnitFloat(words.y))*SAMPLE_SPACING;
for(var vertex=0u;vertex<${frame.positions.length}u;vertex++){
 let point=samplePosition(vertex);if(!isFiniteValue(point.x)||!isFiniteValue(point.y)||max(abs(point.x),abs(point.y))*0.00000095367431640625>SAMPLE_SPACING){failure=1u;}
 ${area ? `if(any(point<vec2f(${frame.bounds.slice(0, 2).map(getWGSLFloatLiteral).join(',')}))||any(point>vec2f(${frame.bounds.slice(2).map(getWGSLFloatLiteral).join(',')}))){failure=1u;}` : `if(vertex+1u<${frame.positions.length}u){measure+=distance(point,samplePosition(vertex+1u));}`}
}
${area ? `for(var triangle=0u;triangle<${frame.positions.length / 3}u;triangle++){let first=samplePosition(triangle*3u);let determinant=sampleCross(samplePosition(triangle*3u+1u)-first,samplePosition(triangle*3u+2u)-first);if(!isFiniteValue(determinant)||determinant==0.0){failure=1u;}}` : 'if(!isFiniteValue(measure)||measure<=0.0){failure=1u;}'}
${
  frame.legacySites
    ? `for(var fixed=0u;fixed<${legacyCount}u;fixed++){
 let point=sampleLegacy(fixed);
 if(!isFiniteValue(point.x)||!isFiniteValue(point.y)||!isFiniteValue(point.z)${area ? '||samplePrimitive(point.xy)==0xffffffffu' : '||point.z<0.0||point.z>measure'}){failure=1u;}
 for(var previous=0u;previous<fixed;previous++){let other=sampleLegacy(previous);if(${area ? 'distance(point.xy,other.xy)' : 'abs(point.z-other.z)'}<SAMPLE_SPACING){failure=1u;}}
}`
    : ''
}
if(failure==0u){
 ${frame.legacySites ? `for(var fixed=0u;fixed<${legacyCount}u;fixed++){sampleWrite(index,required,sampleLegacy(fixed).xy,0.0,fixed,replicate,fixed,2u);required++;}` : ''}
 var candidate=0u;
 loop {
  ${area ? `if(candidate>=${columns * rows}u){break;}` : 'let distanceAlong=phase.x+f32(candidate)*SAMPLE_SPACING;if(distanceAlong>=measure){break;}'}
  if(candidate>=${props.maximumCandidates ?? 65536}u){failure|=4u;break;}
  ${
    area
      ? `let point=vec2f(${frame.bounds.slice(0, 2).map(getWGSLFloatLiteral).join(',')})+phase+vec2f(f32(candidate%${columns}u),f32(candidate/${columns}u))*SAMPLE_SPACING;
 let source=samplePrimitive(point);var accepted=source!=0xffffffffu&&all(point<vec2f(${frame.bounds.slice(2).map(getWGSLFloatLiteral).join(',')}));`
      : 'let location=sampleLine(distanceAlong);let point=location.xy;let source=u32(location.z);var accepted=true;'
  }
 ${frame.legacySites ? `for(var fixed=0u;fixed<${legacyCount}u;fixed++){let other=sampleLegacy(fixed);if(${area ? 'distance(point,other.xy)' : 'abs(distanceAlong-other.z)'}<SAMPLE_SPACING){accepted=false;}}` : ''}
  if(accepted){sampleWrite(index,required,point,${getWGSLFloatLiteral(area ? props.spacing ** 2 : props.spacing)},source,replicate,candidate,select(1u,0u,index==0u));required++;randomAccepted++;}
  candidate++;
 }
}
if(required>SAMPLE_CAPACITY){failure|=2u;}
let written=min(required,SAMPLE_CAPACITY);
for(var slot=written;slot<SAMPLE_CAPACITY;slot++){sampleWrite(index,slot,vec2f(0.0),0.0,0xffffffffu,replicate,0xffffffffu,0xffffffffu);}
status[statusOffset+index*4u]=written;status[statusOffset+index*4u+1u]=required;status[statusOffset+index*4u+2u]=randomAccepted;status[statusOffset+index*4u+3u]=failure;`
      })
    ];
  }
}

/**
 * Upper bound on the serial inner-loop iterations of one wave invocation. A candidate scans every
 * frame primitive (triangles, or segments to locate a line position) and every legacy site; a line
 * legacy site is located by scanning the segments, an area site is read directly. Legacy
 * validation and writes cost at most one candidate's work per site. Vertex checks and slot padding
 * add the frame rows and the capacity.
 */
function getWaveWork(
  type: 'line' | 'area',
  frameRowCount: number,
  legacySiteCount: number,
  candidateCount: number,
  capacity: number
): number {
  const primitiveCount = type === 'area' ? frameRowCount / 3 : frameRowCount - 1;
  const legacySiteCost = type === 'area' ? 1 : primitiveCount;
  return (
    (candidateCount + legacySiteCount) * (primitiveCount + legacySiteCount * legacySiteCost + 1) +
    frameRowCount +
    capacity
  );
}

/** Caller-owned output map for {@link GPUContinuousSpatialSample}. */
export type GPUContinuousSpatialSampleOutput = GPUContinuousSpatialSampleProps['output'];
