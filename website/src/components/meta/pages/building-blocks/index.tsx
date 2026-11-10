// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Parallel building blocks: reduce, scan and scatter, sort and the grid index, each stepped on
// small data beside the recorded kernels that do the same at scale.

import {formatCount} from '../../../../lens/format';
import type {LensViewModel} from '../../../../lens/types';
import {Section} from '../../../ui/section';
import {Figure, InShort, MissingRun, Prose, StatStrip, Term, TopicShell} from '../../page-kit';
import type {TopicPageProps} from '../../topic-types';
import {BlocksTable} from './blocks-table';
import {CompactionFigure} from './compaction-figure';
import {SCENARIOS} from './data';
import {GridIndexFigure} from './grid-index-figure';
import {RadixFigure} from './radix-figure';
import {ReductionFigure} from './reduction-figure';
import {SerialParallelFigure} from './serial-parallel-figure';
import {RECORDED} from './toy-data';
import './building-blocks.css';

function findNode(model: LensViewModel, suffix: string) {
  return model.nodes.find(node => node.id.endsWith(suffix));
}

function KernelDensityStats({model}: {model: LensViewModel}) {
  const pointWeights = findNode(model, '-point-weights');
  const gather = findNode(model, '-gather');
  const countsKey = Object.keys(model.bufferBytes ?? {}).find(key =>
    key.endsWith('-index-counts-chunk-0')
  );
  // One uint32 count per index cell.
  const indexCells = countsKey ? (model.bufferBytes?.[countsKey] ?? 0) / 4 : 0;
  return (
    <StatStrip
      stats={[
        {value: formatCount(pointWeights?.invocations ?? 0), label: 'points indexed'},
        {value: formatCount(indexCells), label: 'index cells counted and scanned'},
        {
          value: formatCount(gather?.invocations ?? 0),
          label: 'output cells, one gather invocation each'
        },
        {value: formatCount(gather?.workgroups ?? 0), label: 'gather workgroups'}
      ]}
    />
  );
}

export function Page({topic, models}: TopicPageProps) {
  const timeWindow = models[SCENARIOS.timeWindow];
  const kernelDensity = models[SCENARIOS.kernelDensity];
  const {rowCount, keptCount} = RECORDED.timeWindow;
  return (
    <TopicShell
      topic={topic}
      tryIt={[
        {
          kind: 'function',
          id: 'time-window-filter',
          note: 'Flags, scan and scatter on timestamps.'
        },
        {
          kind: 'function',
          id: 'cell-aggregation',
          note: 'Radix sort by cell key, then per-cell sums.'
        },
        {
          kind: 'function',
          id: 'kernel-density',
          note: 'A grid index and one gather per output cell.'
        },
        {kind: 'topic', slug: 'iteration', note: 'The same blocks inside rounds gated on the GPU.'},
        {kind: 'topic', slug: 'contract', note: 'The rules that require these blocks.'}
      ]}
    >
      <InShort
        points={[
          <>
            A <Term id="kernel" /> runs one <Term id="invocation" /> per row, all at once, so no
            invocation can keep a running count.
          </>,
          <>
            Totals come from a <Term id="reduction" />, write positions from a <Term id="scan" />,
            packed outputs from <Term id="compaction">compaction</Term>, groups from a{' '}
            <Term id="sort" />, neighbours from a <Term id="grid-index" />.
          </>,
          <>
            Each is a few kernels you can see in the recorded graphs; the contributor contract
            forbids the loop.
          </>
        ]}
      />

      <Section title="One row each">
        <Prose>
          A loop over rows carries state: to pack the kept rows it keeps a counter <code>n</code>{' '}
          and writes each kept row at <code>n++</code>. A kernel has no such counter. Its
          invocations run together and each sees only its own row, so a kept row cannot know how
          many kept rows came before it.
        </Prose>
        <Figure notice="The loop's slot depends on every earlier row; an invocation sees one row.">
          <SerialParallelFigure />
        </Figure>
        <Prose>
          The library's rule is that no single invocation loops over all rows (contributor contract,
          rule 2). Counting, totals and packing go through four published luma.gl blocks instead.
        </Prose>
      </Section>

      {/* Fixed-point sums: modules/analysis/src/analysis-core/kernels/fixed-point-accumulator.ts:14-20.
          Contract rules 2 and 3: dev-docs/contributor-contract.md:13-21. Atomic scatter and the
          in-cell sort: gpu-spatial-analysis/point-density/gpu-kernel-density.ts:751. */}
      <Section title="Reduce to a total">
        <Prose>
          A <Term id="reduction" /> adds values in pairs, then pairs of pairs. Sixteen values need
          four steps, a million need twenty, and every step runs in parallel.
        </Prose>
        <Figure
          notice="Four steps reduce sixteen values to one."
          caption={
            <>
              Leaves are kept rows per slice of the recorded {formatCount(rowCount)}-row run.
              Integer sums are exact in any order; float sums use a 96-bit fixed-point accumulator
              so the bits do not depend on the order either.
            </>
          }
        >
          <ReductionFigure />
        </Figure>
      </Section>

      <Section title="Scan and scatter">
        <Prose>
          A <Term id="scan" /> (prefix sum) gives each row the sum of everything before it. Scan the
          0 or 1 flags and every kept row learns its slot; then each writes itself there in one
          parallel scatter. That is <Term id="compaction">compaction</Term>.
        </Prose>
        {timeWindow ? (
          <Figure
            notice="The scan column is each kept row's slot in the packed output."
            caption={
              <>
                The recorded time-window filter does exactly this to {formatCount(rowCount)} rows
                and keeps {formatCount(keptCount)}. Its timestamps are sorted, so the kept rows form
                one run; the scan works the same for any mask.
              </>
            }
          >
            <CompactionFigure model={timeWindow} />
          </Figure>
        ) : (
          <MissingRun scenarioId={SCENARIOS.timeWindow} />
        )}
      </Section>

      <Section title="Sort by digits">
        <Prose>
          A radix <Term id="sort" /> is count, scan and scatter repeated, once per digit from the
          lowest. The scatter is stable, so each pass keeps the order the earlier passes made.
        </Prose>
        <Figure
          notice="Same three steps per digit; after the last pass the keys are in order."
          caption="Toy keys use 2-bit digits so they fit; the recorded sort uses 4-bit digits over 64-bit cell keys stored as two words."
        >
          <RadixFigure kernelIds={RECORDED.cellAggregationKernels} />
        </Figure>
      </Section>

      <Section title="A grid index">
        <Prose>
          Testing every pair of points costs rows squared. A <Term id="grid-index" /> counts points
          per cell, scans the counts into cell offsets and scatters the point ids into cell order. A
          query then reads only the cells near it, and each grid row of that neighbourhood is one
          contiguous range.
        </Prose>
        {kernelDensity ? (
          <>
            <Figure
              notice="Move the query cell: it visits a few cells, never all sixty points."
              caption="Kernel density's index scatters with atomics, so a stable sort (sort-cells) restores row order inside each cell before any float is summed."
            >
              <GridIndexFigure model={kernelDensity} />
            </Figure>
            <KernelDensityStats model={kernelDensity} />
          </>
        ) : (
          <MissingRun scenarioId={SCENARIOS.kernelDensity} />
        )}
      </Section>

      <Section title="Which block when">
        <Prose>
          Most operators combine several blocks. Rows of uneven length, such as neighbour lists, are
          split evenly over invocations with a merge-path loop built on a scan.
        </Prose>
        <Figure caption="Records counts the recorded runs whose graph contains the block's kernels, recognised by the node ids luma.gl gives them. Source files counts library files that construct the class or import the loop.">
          <BlocksTable />
        </Figure>
      </Section>
    </TopicShell>
  );
}
