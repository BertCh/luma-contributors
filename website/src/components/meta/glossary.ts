// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The vocabulary of the how-it-works pages. One sentence per term, who defines it, and the page
// that explains it. <Term id="..."> in page-kit.tsx links here; the glossary page lists them all.

import type {TopicSlug} from './topics';

/** Who defines the term: the WebGPU standard, published luma.gl, or this library. */
export type GlossaryOwner = 'webgpu' | 'luma.gl' | 'gridlight';

export type GlossaryTerm = {
  id: string;
  term: string;
  /** One sentence, present tense. */
  definition: string;
  owner: GlossaryOwner;
  /** The page that explains the term. */
  topic: TopicSlug;
  /** The symbol or syntax a reader will meet in code, when there is one. */
  code?: string;
};

export const GLOSSARY_OWNER_LABELS: Record<GlossaryOwner, string> = {
  webgpu: 'WebGPU',
  'luma.gl': 'luma.gl',
  gridlight: 'this library'
};

export const GLOSSARY: readonly GlossaryTerm[] = [
  {
    id: 'buffer',
    term: 'Buffer',
    definition:
      'A block of GPU memory. A graph imports a buffer the caller owns, or allocates a transient one as scratch.',
    owner: 'luma.gl',
    topic: 'data-on-the-gpu',
    code: 'graph.importBuffer()'
  },
  {
    id: 'view',
    term: 'View',
    definition:
      'A typed, non-owning range of a graph buffer: a format and a row count. Every input and output of an operation is a view.',
    owner: 'luma.gl',
    topic: 'data-on-the-gpu',
    code: 'GraphDataView'
  },
  {
    id: 'format',
    term: 'Format',
    definition: 'How the bytes of one row are laid out, such as two 32-bit floats per point.',
    owner: 'luma.gl',
    topic: 'data-on-the-gpu',
    code: "'float32x2'"
  },
  {
    id: 'offsets',
    term: 'Offsets',
    definition:
      'An array that says where each variable-length row starts in a flat array of values. Polygons, neighbor lists and networks all use it.',
    owner: 'gridlight',
    topic: 'data-on-the-gpu',
    code: 'ringOffsets, offsets'
  },
  {
    id: 'mask',
    term: 'Mask and validity',
    definition:
      'One uint32 per row. A nonzero mask keeps an input row; a nonzero validity marks an output row as computed.',
    owner: 'gridlight',
    topic: 'data-on-the-gpu',
    code: 'mask, validity'
  },
  {
    id: 'compact-output',
    term: 'Compact output',
    definition:
      'A variable-length result in a fixed-capacity buffer, with the count written, the count required and an overflow flag.',
    owner: 'gridlight',
    topic: 'data-on-the-gpu',
    code: 'GPUCompactOutput'
  },
  {
    id: 'kernel',
    term: 'Kernel',
    definition:
      'One WGSL compute program. The GPU runs it once per element, for thousands of elements at the same time.',
    owner: 'luma.gl',
    topic: 'kernels',
    code: 'Kernel'
  },
  {
    id: 'invocation',
    term: 'Invocation',
    definition: 'One run of a kernel. It receives an index and computes the element at that index.',
    owner: 'webgpu',
    topic: 'kernels'
  },
  {
    id: 'workgroup',
    term: 'Workgroup',
    definition:
      'A block of invocations launched together. The library picks a power-of-two size from 32 to 256.',
    owner: 'webgpu',
    topic: 'kernels',
    code: '@workgroup_size'
  },
  {
    id: 'dispatch',
    term: 'Dispatch',
    definition: 'The command that launches a kernel over a grid of workgroups.',
    owner: 'webgpu',
    topic: 'kernels',
    code: 'dispatchWorkgroups'
  },
  {
    id: 'index-guard',
    term: 'Index guard',
    definition:
      'The first line of every kernel: invocations past the element count return without touching memory.',
    owner: 'gridlight',
    topic: 'kernels',
    code: 'if (index >= count) { return; }'
  },
  {
    id: 'contributor',
    term: 'Contributor',
    definition:
      'A class for one operation. Its constructor validates the props; its getCommandNodes returns the kernels.',
    owner: 'gridlight',
    topic: 'contract',
    code: 'GPUKernelDensity'
  },
  {
    id: 'recipe',
    term: 'Recipe',
    definition:
      'A function that adds several contributors to a graph in order and returns their outputs, intermediates and status.',
    owner: 'gridlight',
    topic: 'workflows',
    code: 'addHotSpotAnalysisRecipe'
  },
  {
    id: 'command-graph',
    term: 'Command graph',
    definition:
      'The list of kernels of a computation and the buffers each one reads and writes. It compiles once and encodes every frame.',
    owner: 'luma.gl',
    topic: 'command-graph',
    code: 'GPUCommandGraph'
  },
  {
    id: 'command-node',
    term: 'Command node',
    definition: 'One step of a command graph: a compute kernel, a copy or a render pass.',
    owner: 'luma.gl',
    topic: 'command-graph'
  },
  {
    id: 'hazard',
    term: 'Hazard',
    definition:
      'An ordering the compiler infers when two kernels touch the same buffer: read after write, write after read, or write after write.',
    owner: 'luma.gl',
    topic: 'command-graph'
  },
  {
    id: 'compile',
    term: 'Compile',
    definition:
      'Orders the kernels by their hazards, plans buffer lifetimes and creates the pipelines. It happens once.',
    owner: 'luma.gl',
    topic: 'scheduling',
    code: 'graph.compile()'
  },
  {
    id: 'encode',
    term: 'Encode',
    definition:
      'Records the compiled passes into a command encoder. It happens every frame and does no analysis.',
    owner: 'luma.gl',
    topic: 'scheduling',
    code: 'compiled.encode()'
  },
  {
    id: 'parameter-buffer',
    term: 'Parameter buffer',
    definition:
      'A small GPU buffer of per-frame numbers such as a bandwidth. Writing it changes the result without recompiling.',
    owner: 'gridlight',
    topic: 'scheduling',
    code: 'GPUParameterBuffer'
  },
  {
    id: 'reduction',
    term: 'Reduction',
    definition: 'Folds a column to one value, such as a sum or a maximum, in a tree of passes.',
    owner: 'luma.gl',
    topic: 'building-blocks',
    code: 'GPUReduction'
  },
  {
    id: 'scan',
    term: 'Scan',
    definition:
      'A prefix sum: element i becomes the total of everything before it. It turns counts into write positions.',
    owner: 'luma.gl',
    topic: 'building-blocks',
    code: 'GPUScan'
  },
  {
    id: 'compaction',
    term: 'Compaction',
    definition:
      'Keeps the flagged rows and packs them to the front, using a scan to place each one.',
    owner: 'luma.gl',
    topic: 'building-blocks',
    code: 'GPUCompaction'
  },
  {
    id: 'sort',
    term: 'Sort',
    definition:
      'A radix sort: for each digit, count, scan and scatter. Stable, so ties keep row order.',
    owner: 'luma.gl',
    topic: 'building-blocks',
    code: 'GPUSort'
  },
  {
    id: 'grid-index',
    term: 'Grid index',
    definition:
      'Points bucketed by cell, so a kernel visits only the neighbors near its element and never every pair.',
    owner: 'luma.gl',
    topic: 'building-blocks',
    code: 'GPUGridIndex'
  },
  {
    id: 'round',
    term: 'Round',
    definition:
      'One pass of an iterative method. Rounds are separate kernels added when the graph is built, up to a budget.',
    owner: 'gridlight',
    topic: 'iteration',
    code: 'maximumIterations'
  },
  {
    id: 'gate',
    term: 'Gate',
    definition:
      'A dispatch whose size is read from a GPU buffer, through an indirect dispatch of luma.gl. A round with nothing left to do runs zero workgroups.',
    owner: 'gridlight',
    topic: 'iteration',
    code: 'createWGSLDispatchGate'
  },
  {
    id: 'status',
    term: 'Status',
    definition:
      'Named outputs a kernel writes about its own run, such as overflow, converged or invalidCount. The caller decides whether to read them.',
    owner: 'gridlight',
    topic: 'contract',
    code: 'output.status'
  },
  {
    id: 'readback',
    term: 'Readback',
    definition:
      'Copying a GPU buffer to the CPU. It is asynchronous, the application owns it, and no operation needs it to finish.',
    owner: 'luma.gl',
    topic: 'anatomy',
    code: 'GPUReadbackRing'
  },
  {
    id: 'snapshot-tap',
    term: 'Snapshot tap',
    definition:
      'A copy node that captures chosen buffers of a run so a record can show their values.',
    owner: 'gridlight',
    topic: 'lenses',
    code: 'GPUSnapshotTap'
  },
  {
    id: 'record',
    term: 'Computation record',
    definition:
      'Plain data that describes one compiled graph and its captured values: kernels, buffers, hazards, formulas, choices.',
    owner: 'gridlight',
    topic: 'lenses',
    code: 'createGPUComputationRecord'
  },
  {
    id: 'lens',
    term: 'Lens',
    definition:
      'A recorded scenario of one operation, drawn as its graph, its formula and the data it produced.',
    owner: 'gridlight',
    topic: 'lenses'
  }
];

export function getGlossaryTerm(id: string): GlossaryTerm | undefined {
  return GLOSSARY.find(term => term.id === id);
}
