// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Anatomy of a run: the kernel-density call, line by line, beside who it talks to; then the kernels
// that one `graph.add` declared in the recorded run, and the raster it produced.

import {useMemo, useState} from 'react';
import {formatResourceBytes, formatCount} from '../../../../lens/format';
import {FEATURED_LENS_SCENARIOS} from '../../../../lens/featured';
import {Section} from '../../../ui/section';
import {findExampleLines} from '../../examples';
import {LensGraphPanel} from '../../lens-graph-panel';
import {
  CodeWalk,
  type CodeWalkStep,
  Figure,
  InShort,
  MissingRun,
  Prose,
  StatStrip,
  Term,
  TopicShell
} from '../../page-kit';
import {SnapshotMap} from '../../snapshot-map';
import type {TopicPageProps} from '../../topic-types';
import {getPhaseOfNode, type PhaseId, PhaseStrip} from './phases';
import {RunSequence, type RunStepId} from './run-sequence';
import './anatomy.css';

const STEP_TEXT: Record<RunStepId, {title: string; text: string; from: string; to?: string}> = {
  buffers: {
    title: 'Buffers',
    text: 'Your points and your output raster live in GPU buffers you create and keep.',
    from: '// 1.',
    to: '});'
  },
  views: {
    title: 'Views',
    text: 'A view tells the graph what the bytes are: a format and a row count. Nothing is copied.',
    from: '// 2.',
    to: "{format: 'float32', length"
  },
  build: {
    title: 'Build',
    text: 'The constructor checks every prop. graph.add declares all the kernels of the operation.',
    from: '// 3.',
    to: '// 4.'
  },
  compile: {
    title: 'Compile',
    text: 'Once: the graph orders the kernels from what they read and write, and creates pipelines.',
    from: '// 4.',
    to: 'compileAsync'
  },
  encode: {
    title: 'Encode and submit',
    text: 'Every frame: record the passes, hand them to the GPU. No analysis runs on the CPU.',
    from: '// 5.',
    to: 'device.submit'
  },
  readback: {
    title: 'Read back',
    text: 'Optional. The result is already in your buffer, where a layer can draw it.',
    from: '// 6.',
    to: 'new Float32Array'
  }
};

const STEP_ORDER: readonly RunStepId[] = [
  'buffers',
  'views',
  'build',
  'compile',
  'encode',
  'readback'
];

export function Page({topic, models, code}: TopicPageProps) {
  const density = models[FEATURED_LENS_SCENARIOS.kernelDensity];
  const example = code.kernelDensity;
  const [step, setStep] = useState<RunStepId>('build');
  const [hoveredPhase, setHoveredPhase] = useState<PhaseId | null>(null);
  const [pinnedPhase, setPinnedPhase] = useState<PhaseId | null>(null);
  const phase = hoveredPhase ?? pinnedPhase;

  const steps = useMemo<CodeWalkStep[]>(
    () =>
      STEP_ORDER.map(id => {
        const [first, last] = findExampleLines(
          example?.code ?? '',
          STEP_TEXT[id].from,
          STEP_TEXT[id].to
        );
        // "Build" runs up to the line before the next comment.
        return {
          id,
          title: STEP_TEXT[id].title,
          text: STEP_TEXT[id].text,
          lines: [first, id === 'build' ? last - 2 : last] as const
        };
      }),
    [example]
  );

  const kernelCount = density
    ? density.nodes
        .filter(node => node.kind === 'compute')
        .reduce((sum, node) => sum + node.count, 0)
    : undefined;
  const highlightNodeIds = useMemo(
    () =>
      density && phase
        ? density.nodes.filter(node => getPhaseOfNode(node) === phase).map(node => node.id)
        : undefined,
    [density, phase]
  );
  const gather = density?.nodes.find(node => node.label === 'gather');
  const raster = density?.map?.raster;

  return (
    <TopicShell
      topic={topic}
      tryIt={[
        {kind: 'function', id: 'kernel-density', note: 'Props, outputs and a live run.'},
        {
          kind: 'reference',
          slug: 'api/gpu-spatial-analysis/operators-density',
          label: 'Density operators',
          note: 'Every option of the density family.'
        },
        {kind: 'topic', slug: 'data-on-the-gpu', note: 'Next: what a view is.'}
      ]}
    >
      <InShort
        points={[
          <>
            You own the <Term id="buffer">buffers</Term>. An operation is a list of{' '}
            <Term id="kernel">kernels</Term> added to a <Term id="command-graph" />.
          </>,
          <>
            The graph <Term id="compile">compiles</Term> once. A frame only{' '}
            <Term id="encode">encodes</Term> and submits.
          </>,
          <>
            Results stay on the GPU. <Term id="readback">Reading back</Term> is your choice, never a
            step of the analysis.
          </>
        ]}
      />

      <Section title="The whole call">
        <Prose>
          This is kernel density over{' '}
          {density ? formatCount((density.bufferBytes?.observations ?? 0) / 8) : 'a few thousand'}{' '}
          wildlife observations, start to finish. Six steps; four of them happen once.
        </Prose>
        {example ? (
          <CodeWalk
            example={example}
            steps={steps}
            active={step}
            onActive={id => setStep(id as RunStepId)}
            label="kernel-density.example.ts, compiled against the public API"
          />
        ) : null}
        <Figure
          notice="Only two arrows reach your code's side again."
          caption="Buffers go up once; numbers come back only when you ask. Everything between is the graph talking to the GPU."
        >
          <RunSequence active={step} onActive={setStep} kernelCount={kernelCount} />
        </Figure>
      </Section>

      <Section title="What one add became">
        {density ? (
          <>
            <Prose>
              That single <code>graph.add</code> declared {kernelCount} kernels. This is the run
              recorded on a real GPU, grouped by the job each kernel does.
            </Prose>
            <PhaseStrip
              model={density}
              active={phase}
              onActive={setHoveredPhase}
              pinned={pinnedPhase}
              onPin={setPinnedPhase}
            />
            <Figure
              notice="Most kernels build the index; one kernel does the estimate. Press a job above to find its kernels."
              caption={`${formatCount(density.recordedNodeCount)} recorded nodes and ${formatCount(density.recordedEdgeCount)} dependencies, none written by hand. An edge is a buffer one kernel writes and a later one reads.`}
            >
              <LensGraphPanel scenario={density} highlightNodeIds={highlightNodeIds} />
            </Figure>
          </>
        ) : (
          <MissingRun scenarioId={FEATURED_LENS_SCENARIOS.kernelDensity} />
        )}
      </Section>

      {density?.map && raster ? (
        <Section title="What comes back">
          <div className="anatomy-result">
            <div className="anatomy-result__map">
              <SnapshotMap map={density.map} label="Kernel density raster with its observations" />
            </div>
            <div>
              <StatStrip
                stats={[
                  {
                    value: formatCount((density.bufferBytes?.density ?? 0) / 4),
                    label: 'raster cells, one float each'
                  },
                  {
                    value: formatResourceBytes(
                      density.bufferBytes?.density ?? 0,
                      density.bufferBytesKind?.density
                    ),
                    label: 'the density buffer, yours to draw or read'
                  },
                  {
                    value: gather ? formatCount(gather.workgroups) : '',
                    label: 'workgroups in the gather dispatch'
                  },
                  {
                    value: formatResourceBytes(
                      density.bufferBytes?.['kde-parameters'] ?? 0,
                      density.bufferBytesKind?.['kde-parameters']
                    ),
                    label: 'to write for a new bandwidth'
                  }
                ]}
              />
              <Prose>
                The raster and the points, captured from the recorded run and drawn here downsampled
                to {raster.width} by {raster.height}. Values are observations per square metre, from{' '}
                {raster.min} to {raster.max}.
              </Prose>
              <Prose>
                Change the bandwidth and only the parameter buffer changes. The next frame encodes
                the same compiled graph and the raster is new.
              </Prose>
            </div>
          </div>
        </Section>
      ) : null}
    </TopicShell>
  );
}
