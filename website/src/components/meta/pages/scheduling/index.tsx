// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import {formatBytes, formatCount} from '../../../../lens/format';
import {FEATURED_LENS_SCENARIOS} from '../../../../lens/featured';
import type {LensViewModel} from '../../../../lens/types';
import {Section} from '../../../ui/section';
import {findExampleLines} from '../../examples';
import {LensGraphPanel} from '../../lens-graph-panel';
import {getLiveBytes, getPeakLiveBytes} from '../../lifetime-chart';
import {
  CodeWalk,
  type CodeWalkStep,
  Figure,
  InShort,
  MissingRun,
  Prose,
  RunLocator,
  StatStrip,
  Term,
  TopicShell
} from '../../page-kit';
import type {TopicPageProps} from '../../topic-types';
import {type TimelineStep, TimelineFigure} from './timeline-figure';
import './scheduling.css';

/**
 * What each operation fixes at compile time and what it reads per frame, as its class documentation
 * states: gpu-kernel-density.ts:211-214 and gpu-local-moran.ts:104-106 (dev-docs/contributor-contract.md
 * rule 5).
 */
const LIFETIMES_OF_PROPS = [
  {
    operation: 'GPUKernelDensity',
    perFrame: 'bandwidth and study window (the parameter buffer), contents of every input buffer',
    compileTime:
      'grid size, point count, indexGridSize, indexBounds, kernel, edgeCorrection, normalization, which optional views exist'
  },
  {
    operation: 'GPULocalMoran',
    perFrame: 'contents of weights, values, mask and parameters (significance level)',
    compileTime: 'row count, islands, minimumNeighbors, multipleTesting, which optional views exist'
  }
] as const;

/** The kernel whose inputs come from the most distinct earlier kernels, with those writers. */
function getBusiestReader(model: LensViewModel) {
  let best: {index: number; writers: number[]} = {index: model.nodes.length - 1, writers: []};
  model.nodes.forEach((node, index) => {
    if (node.kind !== 'compute') return;
    const writers = new Set<number>();
    for (const buffer of node.reads ?? []) {
      for (let earlier = index - 1; earlier >= 0; earlier--) {
        if (model.nodes[earlier].writes?.includes(buffer)) {
          writers.add(earlier);
          break;
        }
      }
    }
    // Ties go to the later kernel: the end of the run gathers the most.
    if (writers.size >= best.writers.length && writers.size > 0)
      best = {index, writers: [...writers].sort((a, b) => a - b)};
  });
  return best;
}

/** The step (kernels run) at which the most logical bytes are alive. */
function getPeakStep(model: LensViewModel): number {
  const lifetimes = model.lifetimes ?? [];
  let peakStep = model.nodes.length;
  let peak = -1;
  for (let step = 1; step <= model.nodes.length; step++) {
    const live = getLiveBytes(lifetimes, step);
    if (live > peak) {
      peak = live;
      peakStep = step;
    }
  }
  return peakStep;
}

/** Runs of consecutive compute kernels, as [first, last] indices: each run can share one pass. */
function getComputeRuns(model: LensViewModel): [number, number][] {
  const runs: [number, number][] = [];
  model.nodes.forEach((node, index) => {
    if (node.kind !== 'compute') return;
    const last = runs.at(-1);
    if (last && last[1] === index - 1) last[1] = index;
    else runs.push([index, index]);
  });
  return runs;
}

export function Page({topic, models, code}: TopicPageProps) {
  const density = models[FEATURED_LENS_SCENARIOS.kernelDensity];
  const example = code.compileOnce;
  const [active, setActive] = useState<TimelineStep>('write');

  const steps: CodeWalkStep[] = example
    ? [
        {
          id: 'construct',
          title: 'Construct',
          text: 'The operation checks its props once, here. A bad prop throws before any GPU work.',
          lines: findExampleLines(example.code, 'const parameters', 'graph.add(')
        },
        {
          id: 'compile',
          title: 'Compile, once',
          text: 'Orders the kernels by their hazards and creates the pipelines. The graph freezes.',
          lines: findExampleLines(example.code, 'const compiled')
        },
        {
          id: 'encode',
          title: 'Encode every frame',
          text: 'Records the fixed order into a command encoder; you submit it. Nothing is planned again.',
          lines: findExampleLines(example.code, 'function encodeFrame', 'device.submit')
        },
        {
          id: 'write',
          title: 'Write a parameter',
          text: 'A new bandwidth is a small buffer write. The next frame uses it; nothing recompiles.',
          lines: findExampleLines(example.code, 'function setBandwidth', 'parameters.write')
        }
      ]
    : [];

  if (!density) {
    return (
      <TopicShell topic={topic}>
        <MissingRun scenarioId={FEATURED_LENS_SCENARIOS.kernelDensity} />
      </TopicShell>
    );
  }

  const busiest = getBusiestReader(density);
  const busiestNode = density.nodes[busiest.index];
  const lifetimes = density.lifetimes ?? [];
  const peakStep = getPeakStep(density);
  const peakBytes = getPeakLiveBytes(lifetimes, density.nodes.length);
  const transientCount = lifetimes.filter(lifetime => lifetime.ownership === 'transient').length;
  const computeRuns = getComputeRuns(density);
  const computeCount = density.nodes.filter(node => node.kind === 'compute').length;
  const parameterBytes = density.bufferBytes?.['kde-parameters'];

  return (
    <TopicShell
      topic={topic}
      tryIt={[
        {kind: 'function', id: 'kernel-density', note: 'The recorded run on this page.'},
        {kind: 'function', id: 'local-morans-i', note: 'Per-frame and compile-time props listed.'},
        {kind: 'topic', slug: 'command-graph', note: 'Where the order comes from.'},
        {kind: 'topic', slug: 'kernels', note: 'What one step of the order does on the GPU.'}
      ]}
    >
      <RunLocator stages={['compile', 'encode']} />
      <InShort
        points={[
          'Compile orders the kernels and creates the pipelines once. The graph cannot change after that.',
          'Each frame only encodes that fixed order and submits it.',
          'Numbers that do not change a buffer size or a loop bound live in a small parameter buffer: a write changes the answer without compiling again.'
        ]}
      />

      <Section title="Once, then every frame">
        <Prose>
          <Term id="compile">Compiling</Term> a graph does the planning;{' '}
          <Term id="encode">encoding</Term> it does none. Kernel density, compiled once, runs every
          frame, and a new bandwidth is a write into its <Term id="parameter-buffer" />.
        </Prose>
        <Figure
          notice="Compile happens once; a parameter write does not trigger it."
          caption={`Every frame dispatches the same ${computeCount} kernels in the same order. Only the ${parameterBytes ? `${parameterBytes}-byte ` : ''}parameter buffer changed between frame 2 and frame 3.`}
        >
          <TimelineFigure model={density} active={active} />
        </Figure>
        {example ? (
          <CodeWalk
            example={example}
            steps={steps}
            active={active}
            onActive={id => setActive(id as TimelineStep)}
            label="compile once, encode every frame"
          />
        ) : null}
        <Prose>
          The rule behind this: a prop that changes a buffer size or a loop bound is fixed at
          compile time; any other number is read from a parameter buffer every frame.
        </Prose>
        <div
          className="scheduling-props"
          role="table"
          aria-label="Compile-time and per-frame props"
        >
          <div role="row" className="scheduling-props__head">
            <span role="columnheader">operation</span>
            <span role="columnheader">per frame, no recompile</span>
            <span role="columnheader">compile time, new graph</span>
          </div>
          {LIFETIMES_OF_PROPS.map(row => (
            <div role="row" key={row.operation}>
              <code role="cell">{row.operation}</code>
              <span role="cell" data-kind="frame">
                {row.perFrame}
              </span>
              <span role="cell" data-kind="compile">
                {row.compileTime}
              </span>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Walk the order">
        <Prose>
          Compile turned the hazards into one order. Step through the recorded kernel density run: a
          kernel runs only after every kernel that writes one of its inputs.
        </Prose>
        <Figure
          notice={`Step ${busiest.index + 1}: ${busiestNode.label} runs.`}
          caption={`The kernels that wrote its inputs ran at steps ${busiest.writers.map(writer => writer + 1).join(', ')}, all earlier. Drag the slider to check any other step.`}
        >
          <LensGraphPanel scenario={density} showHazards defaultStep={busiest.index + 1} />
        </Figure>
      </Section>

      <Section title="Buffers live and die">
        <Prose>
          A <Term id="buffer" /> is needed from its first use to its last read. The bars show that
          span for every buffer of the run.
        </Prose>
        <Figure
          notice={`At step ${peakStep}, ${formatBytes(peakBytes)} of ${formatBytes(density.resources.totalBytes)} are in use.`}
          caption={`The bars are logical lifetimes. The record lists all ${density.resources.count} buffers as imported, so here nothing shares memory; luma.gl can let transient buffers whose lifetimes do not overlap share one allocation.`}
        >
          <LensGraphPanel scenario={density} defaultView="buffers" defaultStep={peakStep} />
        </Figure>
        <StatStrip
          stats={[
            {value: formatBytes(peakBytes), label: 'peak logical live bytes'},
            {value: formatBytes(density.resources.totalBytes), label: 'all buffers together'},
            {value: formatCount(density.resources.count), label: 'buffers'},
            {value: formatCount(transientCount), label: 'recorded as transient'}
          ]}
        />
      </Section>

      <Section title="One shared pass">
        <Prose>
          Encoding opens a compute pass and keeps it open while the next kernel is also a compute
          kernel; a copy or render node closes it. Timing queries, or{' '}
          <code>coalesceComputePasses: false</code>, give each kernel its own pass.
        </Prose>
        <Figure
          notice={`${computeCount} compute kernels, ${computeRuns.length} compute ${computeRuns.length === 1 ? 'pass' : 'passes'}.`}
          caption="The recorded order of kernel density, encoded without timing queries: the snapshot copy at the end closes the pass."
        >
          <PassStrip model={density} runs={computeRuns} />
        </Figure>
      </Section>
    </TopicShell>
  );
}

function PassStrip({model, runs}: {model: LensViewModel; runs: readonly [number, number][]}) {
  const cell = 46;
  const width = model.nodes.length * cell + 16;
  return (
    <div className="meta-panel">
      <div className="meta-scroll">
        <svg
          className="scheduling-passes"
          viewBox={`0 0 ${width} 84`}
          style={{minWidth: Math.min(width, 560), maxWidth: 760}}
          role="img"
          aria-label={`${model.nodes.length} recorded steps; consecutive compute kernels share ${runs.length} compute ${runs.length === 1 ? 'pass' : 'passes'}.`}
        >
          {runs.map(([first, last]) => (
            <g key={first} className="scheduling-passes__pass">
              <rect
                x={8 + first * cell - 3}
                y={8}
                width={(last - first + 1) * cell + 2}
                height={46}
                rx={8}
              />
              <text x={8 + first * cell + 4} y={72}>
                compute pass: steps {first + 1} to {last + 1}
              </text>
            </g>
          ))}
          {model.nodes.map((node, index) => (
            <g key={node.id} className="scheduling-passes__node" data-kind={node.kind}>
              <title>{`${index + 1}. ${node.label} (${node.kind})`}</title>
              <rect x={8 + index * cell + 2} y={14} width={cell - 8} height={34} rx={5} />
              <text x={8 + index * cell + cell / 2 - 2} y={35} textAnchor="middle">
                {index + 1}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
