// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  createWGSLActiveCountDispatch,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUOffsetCurveComponentsProps} from './gpu-offset-curve-components';
import {
  boundOffsetAssemblyPasses,
  createBoundedOffsetAssemblySequences
} from './offset-assembly-sequence';
import {
  getOffsetAssemblyWorkgroupWGSL,
  MAXIMUM_OFFSET_WORKGROUP_FRAGMENT_CAPACITY,
  MAXIMUM_OFFSET_WORKGROUP_ROUNDS_PER_DISPATCH,
  OFFSET_ASSEMBLY_WORKGROUP_SIZE
} from './offset-workgroup-assembly';

/** Incoming-edge census visits at most this many other fragments per invocation and dispatch. */
export const MAXIMUM_OFFSET_INCOMING_FRAGMENTS_PER_DISPATCH = 256;
/** Smaller tables can keep the original dispatch schedule. */
const MINIMUM_OFFSET_WORKGROUP_FRAGMENT_CAPACITY = 8192;

/** Exact greedy cancellation/seed/successor order, with parallel searches and constant-work commits. */
export function createOffsetComponentAssemblyNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    capacity: number;
    fragments: GraphDataView<'float32x4'>;
    metadata: GraphDataView<'uint32'>;
    output: GPUOffsetCurveComponentsProps['output'];
  }
): readonly GPUCommandNode<Parameters>[] {
  const {id, capacity, fragments, metadata, output} = props;
  const transient = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const flags = transient('flags', capacity);
  const ranks = transient('ranks', capacity);
  const rows = transient('rows', capacity);
  const live = transient('live', 1);
  // Words 10 and 11 are the iteration-limit flag and the live-row count for batched traversal.
  const control = transient('control', 16);
  const incoming = transient('incoming', capacity);
  const active = createWGSLActiveCountDispatch(graph, {
    id: `${id}-active`,
    operation: 'GPUOffsetCurveComponents',
    count: live,
    maximumItemCount: capacity
  });
  const binding = (
    name: string,
    view: WGSLKernelBinding['view'],
    access: 'read' | 'read_write' = 'read',
    atomic = false
  ): WGSLKernelBinding => ({
    name,
    view,
    type: atomic ? 'atomic<u32>' : view.format.startsWith('float') ? 'f32' : 'u32',
    access
  });
  const common = `const NONE:u32=0xffffffffu;
fn original(row:u32)->u32{return rows[rowsOffset+row];}
fn startPoint(row:u32)->vec2f{let base=fragmentsOffset+original(row)*4u;return vec2f(fragments[base],fragments[base+1u]);}
fn endPoint(row:u32)->vec2f{let base=fragmentsOffset+original(row)*4u;return vec2f(fragments[base+2u],fragments[base+3u]);}
fn line(row:u32)->u32{return metadata[metadataOffset+original(row)*4u];}
fn source(row:u32)->u32{return metadata[metadataOffset+original(row)*4u+1u];}
fn state(row:u32)->u32{return metadata[metadataOffset+original(row)*4u+2u];}
fn matches(first:vec2f,last:vec2f)->bool{return all(first==last);}
fn follows(previous:u32,next:u32)->bool{return line(previous)==line(next) && matches(endPoint(previous),startPoint(next));}`;
  const cancellationWGSL = `
fn isOpposite(candidate:u32,row:u32)->bool {
 return row>candidate && state(row)==1u && line(row)==line(candidate) && matches(startPoint(candidate),endPoint(row)) && matches(endPoint(candidate),startPoint(row));
}
fn shouldCancel(candidate:u32,row:u32,opposite:u32)->bool {
 if(row==candidate) {return opposite!=NONE;}
 return row>candidate && state(row)==1u && line(row)==line(candidate) &&
   (row==opposite || (row<opposite && matches(startPoint(candidate),startPoint(row)) && matches(endPoint(candidate),endPoint(row))));
}`;
  const geometry = [
    binding('fragments', fragments),
    binding('metadata', metadata),
    binding('rows', rows)
  ];
  const node = (
    name: string,
    bindings: WGSLKernelBinding[],
    body: string,
    declarations = '',
    parallel = false
  ) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${name}`,
      operation: 'GPUOffsetCurveComponents',
      variant: name,
      bindings:
        parallel && !bindings.some(value => value.name === 'live')
          ? [...bindings, binding('live', live)]
          : bindings,
      body: parallel ? 'if(index>=live[liveOffset]) {return;}\n' + body : body,
      declarations,
      invocationCount: parallel ? capacity : 1,
      ...(parallel ? {condition: active.condition} : {})
    });
  const prepareCancel = () =>
    node(
      'cancel-prepare',
      [
        binding('metadata', metadata),
        binding('rows', rows),
        binding('live', live),
        binding('control', control, 'read_write')
      ],
      `let candidate=control[controlOffset];control[controlOffset+1u]=candidate;control[controlOffset+2u]=0u;control[controlOffset+3u]=0xffffffffu;
if(candidate<live[liveOffset]) {control[controlOffset+2u]=select(0u,1u,metadata[metadataOffset+rows[rowsOffset+candidate]*4u+2u]==1u);}
control[controlOffset]=candidate+1u;`
    );
  const findOpposite = () =>
    node(
      'cancel-opposite',
      [...geometry, binding('control', control, 'read_write', true)],
      `if(atomicLoad(&control[controlOffset+2u])==0u) {return;}
let candidate=atomicLoad(&control[controlOffset+1u]);
if(isOpposite(candidate,index)) {atomicMin(&control[controlOffset+3u],index);}`,
      common + cancellationWGSL,
      true
    );
  const applyCancel = () =>
    node(
      'cancel-apply',
      [
        binding('fragments', fragments),
        binding('metadata', metadata, 'read_write'),
        binding('rows', rows),
        binding('control', control)
      ],
      `if(control[controlOffset+2u]==0u) {return;}let candidate=control[controlOffset+1u];let opposite=control[controlOffset+3u];
if(shouldCancel(candidate,index,opposite)) {metadata[metadataOffset+original(index)*4u+2u]=0u;}`,
      common + cancellationWGSL,
      true
    );
  const census = node(
    'incoming-census',
    [
      ...geometry,
      binding('control', control),
      binding('live', live),
      binding('incoming', incoming, 'read_write')
    ],
    `if(state(index)!=1u) {return;}var count=incoming[incomingOffset+index];
for(var local=0u;local<${MAXIMUM_OFFSET_INCOMING_FRAGMENTS_PER_DISPATCH}u;local++) {let other=control[controlOffset]+local;if(other>=live[liveOffset]) {break;}
if(other!=index && state(other)==1u && follows(other,index)) {count++;}}
incoming[incomingOffset+index]=count;`,
    common,
    true
  );
  const advanceCensus = node(
    'incoming-next',
    [binding('control', control, 'read_write')],
    `control[controlOffset]=control[controlOffset]+${MAXIMUM_OFFSET_INCOMING_FRAGMENTS_PER_DISPATCH}u;`
  );
  const clearSelection = () =>
    node(
      'select-clear',
      [binding('control', control, 'read_write')],
      `control[controlOffset+1u]=0xffffffffu;control[controlOffset+2u]=0xffffffffu;control[controlOffset+3u]=0xffffffffu;control[controlOffset+4u]=0xffffffffu;`
    );
  const select = () =>
    node(
      'select',
      [...geometry, binding('incoming', incoming), binding('control', control, 'read_write', true)],
      `if(state(index)!=1u) {return;}
atomicMin(&control[controlOffset+3u],index);if(incoming[incomingOffset+index]==0u) {atomicMin(&control[controlOffset+2u],index);}
let current=atomicLoad(&control[controlOffset]);if(current<live[liveOffset] && follows(current,index)) {atomicMin(&control[controlOffset+4u],index);}`,
      common,
      true
    );
  const commitBindings = [
    binding('fragments', fragments),
    binding('metadata', metadata, 'read_write'),
    binding('rows', rows),
    binding('control', control, 'read_write'),
    binding('positions', output.positions, 'read_write'),
    binding('sources', output.sourceEdges, 'read_write'),
    binding('components', output.components, 'read_write')
  ];
  const commitWGSL = `
fn commitSelection(previous:u32,seed:u32,firstActive:u32,next:u32,limit:u32)->u32 {
 if(control[controlOffset+10u]!=0u) {return NONE;}
 var current=previous;
 if(current!=NONE && (current>=limit || state(current)!=2u)) {control[controlOffset+10u]=1u;return NONE;}
 if(current!=NONE && next==NONE) {finishComponent();current=NONE;}
if(current==NONE) {
 current=seed;if(current==NONE) {current=firstActive;}if(current==NONE) {return NONE;}
 if(current>=limit || state(current)!=1u) {control[controlOffset+10u]=1u;return NONE;}
 control[controlOffset+5u]=control[controlOffset+6u];writeVertex(startPoint(current),source(current));
} else {
 if(next>=limit || state(next)!=1u || !follows(current,next)) {control[controlOffset+10u]=1u;return NONE;}
 current=next;
}
writeVertex(endPoint(current),source(current));metadata[metadataOffset+original(current)*4u+2u]=2u;
control[controlOffset]=current;return current;
}
fn writeVertex(point:vec2f,edge:u32) {let vertex=control[controlOffset+6u];if(vertex<${output.positions.length}u) {positions[positionsOffset+vertex*2u]=point.x;positions[positionsOffset+vertex*2u+1u]=point.y;sources[sourcesOffset+vertex]=edge;}control[controlOffset+6u]=vertex+1u;}
fn finishComponent() {let component=control[controlOffset+7u];let vertex=control[controlOffset+6u];let start=control[controlOffset+5u];
if(component<${output.components.length}u && vertex<=${output.positions.length}u) {let base=componentsOffset+component*4u;components[base]=start;components[base+1u]=vertex-start;components[base+2u]=line(control[controlOffset]);components[base+3u]=0u;control[controlOffset+9u]=control[controlOffset+9u] + 1u;control[controlOffset+8u]=vertex;}
control[controlOffset+7u]=component+1u;control[controlOffset]=NONE;}`;
  const commit = () =>
    node(
      'commit',
      [...commitBindings, binding('live', live)],
      `control[controlOffset+1u]=commitSelection(control[controlOffset],control[controlOffset+2u],control[controlOffset+3u],control[controlOffset+4u],min(live[liveOffset],${capacity}u));`,
      common + commitWGSL
    );
  const removeIncoming = () =>
    node(
      'incoming-remove',
      [...geometry, binding('control', control), binding('incoming', incoming, 'read_write')],
      `let used=control[controlOffset+1u];if(used==NONE || state(index)!=1u) {return;}
if(follows(used,index)) {incoming[incomingOffset+index]=incoming[incomingOffset+index] - 1u;}`,
      common,
      true
    );
  const batched =
    capacity >= MINIMUM_OFFSET_WORKGROUP_FRAGMENT_CAPACITY &&
    capacity <= MAXIMUM_OFFSET_WORKGROUP_FRAGMENT_CAPACITY;
  const workgroup = getOffsetAssemblyWorkgroupWGSL(capacity);
  const workgroupNode = (
    name: string,
    bindings: WGSLKernelBinding[],
    body: string,
    declarations: string
  ) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${name}`,
      operation: 'GPUOffsetCurveComponents',
      variant: name,
      invocationCount: OFFSET_ASSEMBLY_WORKGROUP_SIZE,
      workgroupSize: OFFSET_ASSEMBLY_WORKGROUP_SIZE,
      guardIndex: false,
      bindings,
      body,
      declarations: common + workgroup.declarations + declarations
    });
  const cancellation = batched
    ? {
        nodes: [
          workgroupNode(
            'cancel-workgroup',
            [...commitBindings.slice(0, 4), binding('live', live)],
            workgroup.cancelBody,
            cancellationWGSL + workgroup.cancelDeclarations
          )
        ],
        repetitions: Math.ceil(capacity / MAXIMUM_OFFSET_WORKGROUP_ROUNDS_PER_DISPATCH)
      }
    : {nodes: [prepareCancel(), findOpposite(), applyCancel()], repetitions: capacity};
  const traversal = batched
    ? {
        nodes: [
          workgroupNode(
            'traverse-workgroup',
            [...commitBindings, binding('incoming', incoming, 'read_write')],
            workgroup.traverseBody,
            commitWGSL + workgroup.traverseDeclarations
          )
        ],
        repetitions: Math.ceil((capacity + 1) / MAXIMUM_OFFSET_WORKGROUP_ROUNDS_PER_DISPATCH)
      }
    : {nodes: [clearSelection(), select(), commit(), removeIncoming()], repetitions: capacity + 1};
  return boundOffsetAssemblyPasses(id, metadata, [
    createWGSLKernelNode(graph, {
      id: `${id}-clear-components`,
      operation: 'GPUOffsetCurveComponents',
      invocationCount: output.components.length * 4,
      bindings: [binding('components', output.components, 'read_write')],
      body: 'components[componentsOffset+index]=0u;'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-flags`,
      operation: 'GPUOffsetCurveComponents',
      invocationCount: capacity,
      bindings: [binding('metadata', metadata), binding('flags', flags, 'read_write')],
      body: 'flags[flagsOffset+index]=select(0u,1u,metadata[metadataOffset+index*4u+2u]==1u);'
    }),
    ...new GPUScan({
      id: `${id}-scan`,
      input: flags,
      output: ranks,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-compact`,
      operation: 'GPUOffsetCurveComponents',
      invocationCount: capacity,
      bindings: [
        binding('flags', flags),
        binding('ranks', ranks),
        binding('rows', rows, 'read_write'),
        binding('live', live, 'read_write')
      ],
      body: `if(flags[flagsOffset+index]!=0u) {rows[rowsOffset+ranks[ranksOffset+index]]=index;}
if(index==${capacity - 1}u) {live[liveOffset]=ranks[ranksOffset+index]+flags[flagsOffset+index];}`
    }),
    active.updateNode,
    node('cancel-init', [binding('control', control, 'read_write')], 'control[controlOffset]=0u;'),
    ...createBoundedOffsetAssemblySequences(`${id}-cancellation`, [cancellation]),
    createWGSLKernelNode(graph, {
      id: `${id}-incoming-clear`,
      operation: 'GPUOffsetCurveComponents',
      invocationCount: capacity,
      bindings: [binding('incoming', incoming, 'read_write')],
      body: 'incoming[incomingOffset+index]=0u;'
    }),
    node(
      'incoming-init',
      [binding('control', control, 'read_write')],
      'control[controlOffset]=0u;'
    ),
    ...createBoundedOffsetAssemblySequences(`${id}-census`, [
      {
        nodes: [census, advanceCensus],
        repetitions: Math.ceil(capacity / MAXIMUM_OFFSET_INCOMING_FRAGMENTS_PER_DISPATCH)
      }
    ]),
    createWGSLKernelNode(graph, {
      id: `${id}-traversal-init`,
      operation: 'GPUOffsetCurveComponents',
      invocationCount: 16,
      bindings: [
        binding('control', control, 'read_write'),
        ...(batched ? [binding('live', live)] : [])
      ],
      body: `control[controlOffset+index]=select(0u,0xffffffffu,index==0u);
${batched ? 'if(index==11u) {control[controlOffset+11u]=live[liveOffset];}' : ''}`
    }),
    ...createBoundedOffsetAssemblySequences(`${id}-traversal`, [traversal]),
    node(
      'status',
      [binding('control', control), binding('status', output.status, 'read_write')],
      `let vertices=control[controlOffset+6u];let components=control[controlOffset+7u];
status[statusOffset]=control[controlOffset+9u];status[statusOffset+1u]=control[controlOffset+8u];status[statusOffset+2u]=components;status[statusOffset+3u]=vertices;
status[statusOffset+4u]=select(0u,1u,components>${output.components.length}u || vertices>${output.positions.length}u);status[statusOffset+5u]=0u;
status[statusOffset+6u]=select(0u,1u,control[controlOffset+10u]!=0u || control[controlOffset]!=0xffffffffu);status[statusOffset+7u]=0u;`
    )
  ]);
}
