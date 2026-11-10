// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatCount} from '../../../../lens/format';
import {Section} from '../../../ui/section';
import {KernelDemo} from '../../kernel-demo';
import {getKernelLines} from '../../kernel-source';
import {
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
import {WorkgroupDemo} from '../../workgroup-demo';
import {getRecordedDispatch, getRecordedGrid} from './dispatch-numbers';
import {KernelDeclares} from './kernel-declares';
import {SizingTable} from './sizing-table';
import {Link} from '@tanstack/react-router';

const SCENARIO_ID = 'gpu-spatial-analysis.kernel-density';
const INDEX_TAGS = new Set(['index', 'guard']);

export function Page({topic, models}: TopicPageProps) {
  const model = models[SCENARIO_ID];
  const gather = getRecordedDispatch(model, 'gather');
  const pointWeights = getRecordedDispatch(model, 'point-weights');
  const gatherGrid = gather ? getRecordedGrid(gather) : null;
  const indexLines = getKernelLines(64).filter(line => INDEX_TAGS.has(line.tag));
  return (
    <TopicShell
      topic={topic}
      tryIt={[
        {kind: 'function', id: 'terrain-rgb-decode', note: 'The kernel on this page.'},
        {kind: 'function', id: 'kernel-density', note: 'The recorded run behind the numbers.'},
        {
          kind: 'topic',
          slug: 'command-graph',
          note: 'How declared reads and writes become an order.'
        },
        {
          kind: 'topic',
          slug: 'building-blocks',
          note: 'How totals and sorting avoid a loop over rows.'
        }
      ]}
    >
      <RunLocator stages={['dispatch']} />
      <InShort
        points={[
          <>
            A <Term id="kernel">kernel</Term> is one small program; the GPU runs it once per
            element, each <Term id="invocation">invocation</Term> owning one.
          </>,
          <>
            A <Term id="dispatch">dispatch</Term> launches whole{' '}
            <Term id="workgroup">workgroups</Term>, so the last one usually overshoots.
          </>,
          <>
            The <Term id="index-guard">index guard</Term> returns the surplus invocations before
            they touch memory.
          </>
        ]}
      />
      <Section title="One kernel, one element">
        <Prose>
          This is the WGSL the library emits for <code>GPUTerrainRGBDecode</code>, one pixel per
          invocation. Every kernel starts the same way: work out <code>index</code>, then return if
          it is past the element count (amber).
        </Prose>
        <Figure
          notice="Each tile is one invocation and the pixel it owns."
          caption="The library picks the workgroup size itself; the slider is a teaching knob. Hover a workgroup to light the lines it runs."
        >
          <KernelDemo />
        </Figure>
      </Section>
      <Section title="A dispatch covers the data">
        <Prose>
          Invocations launch in whole workgroups, so the slots rarely equal the elements. The
          kernel-density run shows it on a real kernel: one invocation per output cell.
        </Prose>
        {gather && gatherGrid ? (
          <>
            <StatStrip
              stats={[
                {value: formatCount(gather.invocations), label: 'invocations (one per cell)'},
                {value: formatCount(gather.workgroups), label: 'workgroups'},
                {value: gather.workgroupSize, label: 'invocations per workgroup'},
                {value: formatCount(gather.slots), label: 'slots'},
                {value: formatCount(gather.guarded), label: 'guarded out'}
              ]}
            />
            <Figure
              notice="Only the last workgroup has lanes that return."
              caption={
                <>
                  The recorded <code>gather</code> dispatch, drawn to scale. Real GPUs run
                  workgroups in any order, as many at once as the hardware allows.
                </>
              }
            >
              <WorkgroupDemo grid={gatherGrid} />
            </Figure>
          </>
        ) : (
          <MissingRun scenarioId={SCENARIO_ID} />
        )}
      </Section>
      <Section title="How the library sizes it">
        <Prose>
          A kernel gets a power-of-two workgroup size from 32 to 256, or sets its own. The 3-D
          dispatch keeps every dimension under the device limit, and the invocation count is a
          pipeline constant, so kernels of the same shape share one shader module.
        </Prose>
        <SizingTable pointWeights={pointWeights} gather={gather} />
        <div className="meta-panel" style={{marginTop: 'var(--space-4)'}}>
          <pre className="kernels-index-lines" aria-label="How a kernel computes its index">
            {indexLines.map((line, lineIndex) => (
              <span key={lineIndex} data-tag={line.tag}>
                {line.text}
              </span>
            ))}
          </pre>
        </div>
        <p className="meta-figure__caption">
          Rows are 3-D dispatch coordinates folded into one linear <code>index</code>; the dispatch
          sizes and the element count enter as pipeline constants, not as text in the shader.
        </p>
      </Section>
      <Section title="What a kernel declares">
        <Prose>
          A kernel lists the buffers it reads and the buffers it writes, and nothing else. That list
          is what the{' '}
          <Link to="/how-it-works/$topic" params={{topic: 'command-graph'}}>
            command graph
          </Link>{' '}
          orders. No kernel loops over all rows;{' '}
          <Link to="/how-it-works/$topic" params={{topic: 'building-blocks'}}>
            building blocks
          </Link>{' '}
          explains what does that work instead.
        </Prose>
        {model ? (
          <Figure
            notice="A read_write buffer appears on both sides."
            caption="The recorded gather kernel: its declared reads and writes, nothing more."
          >
            <KernelDeclares model={model} label="gather" />
          </Figure>
        ) : null}
      </Section>
    </TopicShell>
  );
}
