// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {FEATURED_LENS_SCENARIOS} from '../../../../lens/featured';
import {formatCount} from '../../../../lens/format';
import {Section} from '../../../ui/section';
import {ExampleCode, Figure, InShort, Prose, RunLocator, Term, TopicShell} from '../../page-kit';
import type {TopicPageProps} from '../../topic-types';
import {
  BufferViewsFigure,
  CapacityFigure,
  CsrFigure,
  MaskValidityFigure,
  PolygonOffsetsFigure,
  PrecisionFigure
} from './figures';

const UINT32_BYTES = 4;

export function Page({topic, models, code}: TopicPageProps) {
  const density = models[FEATURED_LENS_SCENARIOS.kernelDensity];
  const moran = models[FEATURED_LENS_SCENARIOS.localMoran];
  const neighborSlots = moran?.resources.largest.find(
    resource => resource.id === 'weightsNeighbors'
  );
  const example = code['buffer-views'];
  return (
    <TopicShell
      topic={topic}
      tryIt={[
        {kind: 'function', id: 'contiguity-weights', note: 'Polygons in, CSR weights out.'},
        {kind: 'function', id: 'kernel-density', note: 'Points in, a raster band out.'},
        {kind: 'reference', slug: 'api/gpu-spatial-analysis/concepts', label: 'Concepts'},
        {kind: 'topic', slug: 'command-graph'}
      ]}
    >
      <RunLocator stages={['buffers']} />
      <InShort
        points={[
          <>
            An operation never owns your data. It reads a typed <Term id="view" /> of a{' '}
            <Term id="buffer" /> you imported.
          </>,
          <>
            Variable-length rows are flat arrays plus <Term id="offsets" />: where each row starts.
          </>,
          <>
            Output cannot grow on the GPU, so a <Term id="compact-output" /> reports how many rows
            it needed.
          </>
        ]}
      />
      <Section title="Buffer, view, format">
        <Prose>
          You create the buffer and import it into a command graph. The graph returns a view: a{' '}
          <Term id="format" /> laid over the bytes and a row count. The figure shows three buffers
          of the recorded kernel-density run.
        </Prose>
        <div className="dotg-pair">
          <Figure
            notice="A format groups bytes into rows."
            caption="float32x2 puts two floats in a row, so the point buffer is its size divided by 8."
          >
            <BufferViewsFigure model={density} />
          </Figure>
          {example ? <ExampleCode example={example} label="buffer-views.example.ts" /> : null}
        </div>
      </Section>
      <Section title="Shape from offsets">
        <Prose>
          Polygons differ in size, so a table of them cannot be a fixed-width array. Four flat
          arrays hold them instead. Each offsets array says where a row starts in the next one down.
        </Prose>
        <Figure
          notice="A feature is a slice of polygons, a slice of rings, a slice of vertices."
          caption="Each offsets array has one terminal entry, drawn dashed, that closes the last row."
        >
          <PolygonOffsetsFigure />
        </Figure>
      </Section>
      <Section title="Neighbours are offsets too">
        <Prose>
          Spatial weights and networks use the same trick, called CSR. Row i's neighbours are{' '}
          <code>neighbors[offsets[i] .. offsets[i+1])</code>, and <code>weights</code> lines up with{' '}
          <code>neighbors</code>.
        </Prose>
        <Figure
          notice="Choosing a node lights one slice of neighbors and weights."
          caption="Each edge is stored once in each direction, so a graph of nine edges has eighteen slots."
        >
          <CsrFigure />
        </Figure>
        {neighborSlots ? (
          <Prose>
            The recorded local Moran run reserves a neighbours buffer of{' '}
            {formatCount(neighborSlots.bytes)} bytes:{' '}
            {formatCount(neighborSlots.bytes / UINT32_BYTES)} uint32 slots, fixed before the weights
            are built.
          </Prose>
        ) : null}
      </Section>
      <Section title="Fixed capacity, honest count">
        <Prose>
          A kernel cannot allocate. A result of unknown length goes into a buffer whose capacity you
          chose, and the status words say what happened. On overflow you read{' '}
          <code>requiredCount</code>, allocate that many rows and rebuild the graph.
        </Prose>
        <Figure
          notice="count stops at capacity; requiredCount keeps rising."
          caption="Overflow turns on as soon as one row is dropped. The rebuild button is the grow loop."
        >
          <CapacityFigure />
        </Figure>
      </Section>
      <Section title="No data and precision">
        <Prose>
          One rule for no-data: an input <Term id="mask" /> keeps a row when it is nonzero, and an
          output <code>validity</code> is nonzero when the row is valid. Raster no-data lives in the
          band's validity, and a missing key is <code>0xffffffff</code>.
        </Prose>
        <Figure
          notice="Masked rows are skipped, and their validity is zero."
          caption="Click a mask cell to flip a row."
        >
          <MaskValidityFigure />
        </Figure>
        <Prose>
          Longitude and latitude are float32 degrees, which cannot place a point finer than the
          ground quantum. Operators warn when a distance nears ten quanta and point you to{' '}
          <code>createGPULocalFrame</code>, which works in metres. Kernel density already measures
          from a grid-start origin, so large absolute coordinates cancel.
        </Prose>
        <Figure
          notice="Near 17 m, a degree grid has ten steps; a local frame has no visible grid."
          caption="Quanta by place are from the precision section of the concepts reference."
        >
          <PrecisionFigure />
        </Figure>
      </Section>
    </TopicShell>
  );
}
