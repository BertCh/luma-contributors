// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Data flowing along a graph: glowing nodes and particles that travel the edges, with a wave that
// sweeps the schedule left to right like a dispatch. Used by LensGraphPanel and the pipeline
// diagram. The caller lays the graph out in SVG units and the canvas covers exactly that box.

import type {Buffer, Device} from '@luma.gl/core';
import type {Model} from '@luma.gl/engine';
import type {GpuScene, GpuSceneLoader} from '../../../gpu/types';
import {
  assertWebGpu,
  createQuadModel,
  createStorageBuffer,
  createUniformBuffer,
  getSceneColors,
  ScenePointer,
  STILL_FRAME_MILLISECONDS
} from './scene-kit';

export type FlowNode = {
  x: number;
  y: number;
  width: number;
  height: number;
  /** 0 compute (accent), 1 copy (signal), 2 mixed, 3 neutral ink. */
  kind: 0 | 1 | 2 | 3;
  /** Position in the schedule, 0..1; the wave lights nodes in this order. */
  progress: number;
};

export type FlowEdge = {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  controlX0: number;
  controlY0: number;
  controlX1: number;
  controlY1: number;
  /** Relative thickness, 0..1; sets how many particles travel the edge. Zero draws none. */
  weight: number;
  /** Schedule position at the start and end of the edge, 0..1. */
  progressStart: number;
  progressEnd: number;
  /** Index into `graph.nodes` of the node this edge leaves. Needed for dependency highlighting. */
  fromNode?: number;
  /** Index into `graph.nodes` of the node this edge enters. Needed for dependency highlighting. */
  toNode?: number;
};

export type FlowSceneOptions = {
  /**
   * 'none' is a passive picture. 'dependencies' lights the node under the pointer and its
   * dependencies. 'scrub' lets the pointer drive the schedule wave.
   */
  interaction?: 'none' | 'dependencies' | 'scrub';
  /**
   * Which nodes 'dependencies' lights. 'direct' lights the hovered node and its immediate
   * neighbours in either direction. 'transitive' lights the hovered node and everything
   * downstream of it, following edge direction (fromNode to toNode). An edge means "depends on" or
   * "a call descends", so hovering a package lights everything it builds on; following both
   * directions would light the whole of a linear chain and show nothing.
   */
  dependencyDepth?: 'direct' | 'transitive';
};

export type FlowGraph = {width: number; height: number; nodes: FlowNode[]; edges: FlowEdge[]};

const LOOK_FLOATS = 28;
const WAVE_PERIOD_SECONDS = 9;
const HOVERED_FOCUS: [number, number] = [1, 1];
const LIT_FOCUS: [number, number] = [0.5, 1];
const DIMMED_FOCUS: [number, number] = [0, 0.3];
const HIGHLIGHTED_EDGE_FOCUS: [number, number] = [1.8, 1.35];
const DIMMED_EDGE_FOCUS: [number, number] = [0.25, 1];
const MAXIMUM_PARTICLES = 1400;

const SHARED_WGSL = /* wgsl */ `
struct Look {
  world: vec2f,
  size: vec2f,
  clock: vec4f,
  background: vec4f,
  accent: vec4f,
  signal: vec4f,
  ink: vec4f,
  pointer: vec4f
};
@group(0) @binding(0) var<uniform> look: Look;

fn toClip(position: vec2f) -> vec2f {
  return vec2f(position.x / look.world.x * 2.0 - 1.0, 1.0 - position.y / look.world.y * 2.0);
}

// The wave front moves through schedule progress 0..1 and pauses past the end.
fn waveGlow(progress: f32) -> f32 {
  let distance = (progress - (look.clock.y * 1.3 - 0.1)) / 0.12;
  return exp(-distance * distance);
}
`;

const NODE_WGSL = /* wgsl */ `
${SHARED_WGSL}
@group(0) @binding(1) var<storage, read> rects: array<vec4f>;
@group(0) @binding(2) var<storage, read> kinds: array<vec2f>;
@group(0) @binding(3) var<storage, read> nodeFocus: array<vec2f>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) offset: vec2f,
  @location(1) extent: vec2f,
  @location(2) kind: f32,
  @location(3) glow: f32,
  @location(4) alpha: f32
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let rect = rects[instanceIndex];
  let center = rect.xy + rect.zw * 0.5;
  let halo = 22.0;
  let extent = rect.zw * 0.5 + vec2f(halo);
  let corner = corners[vertexIndex];
  var output: VertexOutput;
  output.position = vec4f(toClip(center + corner * extent), 0.0, 1.0);
  output.offset = corner * extent;
  output.extent = rect.zw * 0.5;
  output.kind = kinds[instanceIndex].x;
  let focus = nodeFocus[instanceIndex];
  // Lit nodes glow at least focus.x; dimmed ones (alpha < 1) also lose the schedule wave.
  let wave = waveGlow(kinds[instanceIndex].y);
  output.glow = mix(wave, max(wave, focus.x) * focus.y, look.pointer.z);
  output.alpha = mix(1.0, focus.y, look.pointer.z);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let edge = abs(input.offset) - input.extent + vec2f(6.0);
  let distance = length(max(edge, vec2f(0.0))) + min(max(edge.x, edge.y), 0.0) - 6.0;
  var color = look.accent.rgb;
  if (input.kind > 0.5 && input.kind < 1.5) {
    color = look.signal.rgb;
  } else if (input.kind > 2.5) {
    color = look.ink.rgb;
  } else if (input.kind > 1.5) {
    color = mix(look.accent.rgb, look.signal.rgb, 0.5);
  }
  var strength = 0.16 + 0.84 * input.glow;
  if (input.kind > 2.5) {
    strength = strength * 0.45;
  }
  let halo = exp(-max(distance, 0.0) / 9.0) * strength * 0.55;
  let body = (1.0 - smoothstep(-1.0, 1.0, distance)) * (0.05 + 0.4 * input.glow);
  return vec4f(color, clamp(halo + body, 0.0, 1.0) * input.alpha);
}
`;

const PARTICLE_WGSL = /* wgsl */ `
${SHARED_WGSL}
@group(0) @binding(1) var<storage, read> curves: array<vec4f>;
@group(0) @binding(2) var<storage, read> particles: array<vec4f>;
@group(0) @binding(3) var<storage, read> edgeFocus: array<vec2f>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) offset: vec2f,
  @location(1) brightness: f32,
  @location(2) tint: f32
};

fn bezier(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
  let u = 1.0 - t;
  return u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3;
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let particle = particles[instanceIndex];
  let edgeIndex = u32(particle.x);
  let ends = curves[edgeIndex * 3u];
  let controls = curves[edgeIndex * 3u + 1u];
  let schedule = curves[edgeIndex * 3u + 2u];
  let t = fract(particle.y + look.clock.x * particle.z);
  let position = bezier(ends.xy, controls.xy, controls.zw, ends.zw, t);
  let corner = corners[vertexIndex];
  let focus = edgeFocus[edgeIndex];
  let radius = particle.w * (0.8 + 0.5 * sin(t * 3.14159)) * mix(1.0, focus.y, look.pointer.z);
  var output: VertexOutput;
  output.position = vec4f(toClip(position + corner * radius), 0.0, 1.0);
  output.offset = corner;
  output.brightness =
    (0.5 + 0.5 * waveGlow(mix(schedule.x, schedule.y, t))) * mix(1.0, focus.x, look.pointer.z);
  output.tint = schedule.z;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let distance = length(input.offset);
  if (distance > 1.0) {
    discard;
  }
  let core = 1.0 - smoothstep(0.0, 1.0, distance);
  let color = mix(look.accent.rgb, look.signal.rgb, input.tint);
  return vec4f(color, min(core * core * input.brightness, 1.0));
}
`;

/** Builds a scene for a laid-out graph. The loader form is what <GpuCanvas scene> expects. */
export function createFlowSceneLoader(
  graph: FlowGraph,
  options: FlowSceneOptions = {}
): GpuSceneLoader {
  return async () => ({default: createFlowScene(graph, options)});
}

export function createFlowScene(graph: FlowGraph, options: FlowSceneOptions = {}): GpuScene {
  return ({device, theme, width, height, reducedMotion}) =>
    new FlowSceneInstance(device, graph, options, theme, width, height, reducedMotion);
}

type ScrubPoint = {x: number; y: number; progress: number};

/** One point per distinct progress value (mean node center), ascending. Follows snaking layouts. */
function makeScrubPolyline(nodes: FlowNode[]): ScrubPoint[] {
  const groups = new Map<number, {x: number; y: number; count: number}>();
  for (const node of nodes) {
    const group = groups.get(node.progress) ?? {x: 0, y: 0, count: 0};
    group.x += node.x + node.width / 2;
    group.y += node.y + node.height / 2;
    group.count++;
    groups.set(node.progress, group);
  }
  return [...groups.entries()]
    .sort(([progressA], [progressB]) => progressA - progressB)
    .map(([progress, group]) => ({x: group.x / group.count, y: group.y / group.count, progress}));
}

/** Schedule progress of the polyline point closest to (x, y). */
function findScrubProgress(polyline: ScrubPoint[], x: number, y: number): number {
  if (polyline.length === 0) return 0;
  let bestProgress = polyline[0].progress;
  let bestDistance = Infinity;
  for (let index = 0; index + 1 < polyline.length; index++) {
    const start = polyline[index];
    const end = polyline[index + 1];
    const deltaX = end.x - start.x;
    const deltaY = end.y - start.y;
    const lengthSquared = deltaX * deltaX + deltaY * deltaY;
    const fraction =
      lengthSquared === 0
        ? 0
        : Math.min(
            1,
            Math.max(0, ((x - start.x) * deltaX + (y - start.y) * deltaY) / lengthSquared)
          );
    const distance = Math.hypot(
      x - (start.x + fraction * deltaX),
      y - (start.y + fraction * deltaY)
    );
    if (distance < bestDistance) {
      bestDistance = distance;
      bestProgress = start.progress + fraction * (end.progress - start.progress);
    }
  }
  return bestProgress;
}

/** Index of the node nearest (x, y) by distance to its rectangle, so there are no dead zones. */
function findNearestNode(nodes: FlowNode[], x: number, y: number): number {
  let nearestIndex = -1;
  let nearestDistance = Infinity;
  nodes.forEach((node, index) => {
    const distance = Math.hypot(
      Math.max(node.x - x, 0, x - (node.x + node.width)),
      Math.max(node.y - y, 0, y - (node.y + node.height))
    );
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestIndex = index;
    }
  });
  return nearestIndex;
}

class FlowSceneInstance {
  private readonly device: Device;
  private readonly graph: FlowGraph;
  private readonly colors: ReturnType<typeof getSceneColors>;
  private readonly look = new Float32Array(LOOK_FLOATS);
  private readonly lookBuffer: Buffer;
  private readonly buffers: Buffer[] = [];
  private readonly nodeModel: Model;
  private readonly particleModel: Model;
  private readonly reducedMotion: boolean;
  private readonly interaction: 'none' | 'dependencies' | 'scrub';
  private readonly dependencyDepth: 'direct' | 'transitive';
  private readonly pointer: ScenePointer;
  private readonly scrubPolyline: ScrubPoint[];
  private readonly nodeFocus: Float32Array;
  private readonly edgeFocus: Float32Array;
  private readonly nodeFocusBuffer: Buffer;
  private readonly edgeFocusBuffer: Buffer;
  private hoveredNode = -1;
  private wasActive = false;
  private phaseOffsetSeconds = 0;
  private lastPointerWave = 0;
  private width: number;
  private height: number;

  constructor(
    device: Device,
    graph: FlowGraph,
    options: FlowSceneOptions,
    theme: 'light' | 'dark',
    width: number,
    height: number,
    reducedMotion: boolean
  ) {
    assertWebGpu(device);
    this.device = device;
    this.graph = graph;
    this.colors = getSceneColors(theme);
    this.width = width;
    this.height = height;
    this.reducedMotion = reducedMotion;
    this.interaction = options.interaction ?? 'none';
    this.dependencyDepth = options.dependencyDepth ?? 'direct';
    this.pointer = new ScenePointer(reducedMotion);
    this.scrubPolyline = makeScrubPolyline(graph.nodes);
    this.lookBuffer = createUniformBuffer(device, LOOK_FLOATS);

    const rects = new Float32Array(Math.max(1, graph.nodes.length) * 4);
    const kinds = new Float32Array(Math.max(1, graph.nodes.length) * 2);
    graph.nodes.forEach((node, index) => {
      rects.set([node.x, node.y, node.width, node.height], index * 4);
      kinds.set([node.kind, node.progress], index * 2);
    });

    const curves = new Float32Array(Math.max(1, graph.edges.length) * 12);
    const particleList: number[] = [];
    const budget = Math.min(MAXIMUM_PARTICLES, graph.edges.length * 5);
    const totalWeight = graph.edges.reduce((total, edge) => total + 0.25 + edge.weight, 0) || 1;
    graph.edges.forEach((edge, index) => {
      curves.set([edge.x0, edge.y0, edge.x1, edge.y1], index * 12);
      curves.set([edge.controlX0, edge.controlY0, edge.controlX1, edge.controlY1], index * 12 + 4);
      curves.set(
        [edge.progressStart, edge.progressEnd, edge.weight > 0.66 ? 0.8 : 0, 0],
        index * 12 + 8
      );
      if (edge.weight <= 0) return;
      const count = Math.max(1, Math.round((budget * (0.25 + edge.weight)) / totalWeight));
      for (let particle = 0; particle < count; particle++) {
        // Deterministic spread: golden-ratio phases keep particles evenly spaced along the edge.
        const phase = (particle / count + ((index * 0.61803) % 1)) % 1;
        const speed = 0.16 + (0.1 * ((index * 7 + particle * 3) % 5)) / 5;
        particleList.push(index, phase, speed, 2.6 + 3.4 * edge.weight);
      }
    });
    const particles = new Float32Array(Math.max(4, particleList.length));
    particles.set(particleList);

    // Per node (lift, alpha) and per edge (brightness, size); identity until a node is hovered.
    this.nodeFocus = new Float32Array(Math.max(1, graph.nodes.length) * 2);
    for (let index = 0; index < this.nodeFocus.length; index += 2)
      this.nodeFocus.set([0, 1], index);
    this.edgeFocus = new Float32Array(Math.max(1, graph.edges.length) * 2).fill(1);

    const rectsBuffer = createStorageBuffer(device, rects);
    const kindsBuffer = createStorageBuffer(device, kinds);
    const curvesBuffer = createStorageBuffer(device, curves);
    const particlesBuffer = createStorageBuffer(device, particles);
    this.nodeFocusBuffer = createStorageBuffer(device, this.nodeFocus);
    this.edgeFocusBuffer = createStorageBuffer(device, this.edgeFocus);
    this.buffers.push(
      rectsBuffer,
      kindsBuffer,
      curvesBuffer,
      particlesBuffer,
      this.nodeFocusBuffer,
      this.edgeFocusBuffer
    );

    this.nodeModel = createQuadModel(device, {
      id: 'flow-nodes',
      source: NODE_WGSL,
      bindings: {
        look: this.lookBuffer,
        rects: rectsBuffer,
        kinds: kindsBuffer,
        nodeFocus: this.nodeFocusBuffer
      },
      instanceCount: graph.nodes.length,
      theme
    });
    this.particleModel = createQuadModel(device, {
      id: 'flow-particles',
      source: PARTICLE_WGSL,
      bindings: {
        look: this.lookBuffer,
        curves: curvesBuffer,
        particles: particlesBuffer,
        edgeFocus: this.edgeFocusBuffer
      },
      instanceCount: particleList.length / 4,
      theme
    });
    // Zero nodes or edges still need a valid draw; hide the empty model instead.
    if (graph.nodes.length === 0) this.nodeModel.setInstanceCount(0);
    if (graph.edges.length === 0) this.particleModel.setInstanceCount(0);
  }

  setPointer(position: [number, number] | null): void {
    this.pointer.set(position);
  }

  /** Rewrites both focus arrays for the hovered node and uploads them. */
  private writeFocus(hoveredNode: number): void {
    const {nodes, edges} = this.graph;
    const lit = new Set<number>([hoveredNode]);
    if (this.dependencyDepth === 'direct') {
      for (const edge of edges) {
        if (edge.fromNode === hoveredNode && edge.toNode !== undefined) lit.add(edge.toNode);
        if (edge.toNode === hoveredNode && edge.fromNode !== undefined) lit.add(edge.fromNode);
      }
    } else {
      const queue = [hoveredNode];
      for (let head = 0; head < queue.length; head++) {
        for (const edge of edges) {
          if (edge.fromNode === queue[head] && edge.toNode !== undefined && !lit.has(edge.toNode)) {
            lit.add(edge.toNode);
            queue.push(edge.toNode);
          }
        }
      }
    }
    nodes.forEach((_, index) => {
      const focus =
        index === hoveredNode ? HOVERED_FOCUS : lit.has(index) ? LIT_FOCUS : DIMMED_FOCUS;
      this.nodeFocus.set(focus, index * 2);
    });
    edges.forEach((edge, index) => {
      const isHighlighted =
        edge.fromNode !== undefined &&
        edge.toNode !== undefined &&
        (this.dependencyDepth === 'direct'
          ? edge.fromNode === hoveredNode || edge.toNode === hoveredNode
          : lit.has(edge.fromNode) && lit.has(edge.toNode));
      this.edgeFocus.set(isHighlighted ? HIGHLIGHTED_EDGE_FOCUS : DIMMED_EDGE_FOCUS, index * 2);
    });
    this.nodeFocusBuffer.write(this.nodeFocus);
    this.edgeFocusBuffer.write(this.edgeFocus);
  }

  render(timeMilliseconds: number): void {
    const time = (this.reducedMotion ? STILL_FRAME_MILLISECONDS : timeMilliseconds) / 1000;
    const look = this.look;
    const pointer = this.pointer;
    pointer.update(timeMilliseconds / 1000);
    const worldX = (pointer.x * this.graph.width) / this.width;
    const worldY = (pointer.y * this.graph.height) / this.height;

    // The wave loops every 9 seconds; it overshoots the graph slightly so the pause is dark.
    let wave = (time % WAVE_PERIOD_SECONDS) / WAVE_PERIOD_SECONDS;
    if (this.interaction === 'scrub') {
      if (pointer.presence > 0) {
        const progress = findScrubProgress(this.scrubPolyline, worldX, worldY);
        this.lastPointerWave = (progress + 0.1) / 1.3;
      }
      // Resume autoplay from the scrubbed position so leaving does not jump the wave.
      // Reduced motion has no autoplay to resume, so it returns to the still frame.
      if (this.wasActive && !pointer.isActive && !this.reducedMotion) {
        this.phaseOffsetSeconds =
          this.lastPointerWave * WAVE_PERIOD_SECONDS - (time % WAVE_PERIOD_SECONDS);
      }
      this.wasActive = pointer.isActive;
      const autoWave =
        ((((time + this.phaseOffsetSeconds) % WAVE_PERIOD_SECONDS) + WAVE_PERIOD_SECONDS) %
          WAVE_PERIOD_SECONDS) /
        WAVE_PERIOD_SECONDS;
      wave = autoWave + (this.lastPointerWave - autoWave) * pointer.presence;
    }

    if (this.interaction === 'dependencies' && pointer.presence > 0) {
      const hoveredNode = findNearestNode(this.graph.nodes, worldX, worldY);
      if (hoveredNode !== this.hoveredNode) {
        this.hoveredNode = hoveredNode;
        this.writeFocus(hoveredNode);
      }
    }

    const dependencyPresence = this.interaction === 'dependencies' ? pointer.presence : 0;
    look.set([this.graph.width, this.graph.height, this.width, this.height], 0);
    look.set([time, wave, 0, 0], 4);
    look.set([0, 0, dependencyPresence, 0], 24);
    look.set(this.colors.background, 8);
    look.set(this.colors.accent, 12);
    look.set(this.colors.signal, 16);
    look.set(this.colors.ink, 20);
    this.lookBuffer.write(look);
    const renderPass = this.device.beginRenderPass({clearColor: this.colors.background});
    this.nodeModel.draw(renderPass);
    this.particleModel.draw(renderPass);
    renderPass.end();
    this.device.submit();
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }

  destroy(): void {
    this.nodeModel.destroy();
    this.particleModel.destroy();
    this.lookBuffer.destroy();
    for (const buffer of this.buffers) buffer.destroy();
  }
}

export default createFlowScene({width: 1, height: 1, nodes: [], edges: []});
