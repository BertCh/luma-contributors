// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  buildKernelSourceModel,
  renderKernelSource
} from '../../../src/computation-views/views/kernel-source';
import {viewNodeToHtml} from '../../../src/computation-views/view-node';
import {makeRecord} from '../fixtures/v5-record';

describe('kernel source', () => {
  test('marks declarations and entry point, escapes source, and links resources', () => {
    const record = makeRecord();
    const model = buildKernelSourceModel(record, record.graphs[0].nodes[0]);
    expect(model).toMatchObject({
      lines: [{marks: ['binding']}, {marks: ['workgroup-size']}, {marks: ['entry-point']}],
      truncatedLineCount: 0
    });
    const html = viewNodeToHtml(renderKernelSource(model));
    expect(html).toContain('data-line="1"');
    expect(html).toContain('var&lt;storage');
    expect(html).toContain('data-axis="buffer" data-key="state"');
    expect(html).toContain('lens-provenance--metadata');
    expect(html).not.toContain('<td></td>');
    expect(viewNodeToHtml(renderKernelSource(model))).toBe(html);
  });
  test('reads the node entry point and constants before the shared source ref', () => {
    // Two pipelines share one WGSL module: the ref records the first node's entry point and
    // constants, while this node was created with `other` and FACTOR 3.
    const record = makeRecord();
    const source = record.refs.sources[0];
    record.refs = {
      ...record.refs,
      sources: [
        {
          ...source,
          entryPoint: 'main',
          constants: {FACTOR: 2},
          text: `override FACTOR: f32 = 1.0;\n${source.text!.replace('fn main() {}', 'fn main(@builtin(global_invocation_id) position: vec3<u32>) { values[position.x] *= FACTOR; }')}\n@compute @workgroup_size(4)\nfn other(@builtin(global_invocation_id) position: vec3<u32>) { values[position.x] *= FACTOR; }`
        }
      ]
    };
    const node = {...record.graphs[0].nodes[0], entryPoint: 'other', constants: {FACTOR: 3}};
    expect(record.refs.sources[0].text).toContain('override FACTOR: f32');
    const model = buildKernelSourceModel(record, node);
    expect(model).toMatchObject({entryPoint: 'other', constants: {FACTOR: 3}});
    if (!('lines' in model)) throw new Error('expected retained source');
    expect(
      model.lines.filter(line => line.marks.includes('entry-point')).map(line => line.text)
    ).toEqual([
      'fn other(@builtin(global_invocation_id) position: vec3<u32>) { values[position.x] *= FACTOR; }'
    ]);
    // Entry-point fallback remains available, but shared constants are not per-node evidence.
    const shared = buildKernelSourceModel(record, record.graphs[0].nodes[0]);
    expect(shared).toMatchObject({entryPoint: 'main'});
    expect('constants' in shared && shared.constants).toBeUndefined();
  });
  test('DPC-102: a default pipeline never inherits another pipeline override', () => {
    const record = makeRecord();
    Object.freeze(record.refs.sources);
    record.refs = {
      ...record.refs,
      sources: [
        {
          ...record.refs.sources[0],
          constants: {WIDTH: 4},
          text: `override WIDTH: u32 = 2u;
@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) position: vec3<u32>) {
  if (position.x < WIDTH) { values[position.x] = f32(WIDTH); }
}`
        }
      ]
    };
    const first = {...record.graphs[0].nodes[0], constants: {WIDTH: 4}};
    const second = {...record.graphs[0].nodes[0], id: 'default-width', constants: undefined};
    expect(buildKernelSourceModel(record, first)).toMatchObject({constants: {WIDTH: 4}});
    const model = buildKernelSourceModel(record, second);
    expect('constants' in model && model.constants).toBeUndefined();
    const empty = buildKernelSourceModel(record, {...second, constants: {}});
    expect(empty).toMatchObject({constants: {}});
  });
  test('reports hash-only source and caps retained lines without substring binding matches', () => {
    const record = makeRecord();
    const source = record.refs.sources[0];
    record.refs = {...record.refs, sources: [{...source, text: undefined}]};
    expect(buildKernelSourceModel(record, record.graphs[0].nodes[0])).toEqual({
      status: 'unavailable',
      reason: 'source text not retained (only hash abc123)'
    });
    record.refs = {
      ...record.refs,
      sources: [
        {
          ...source,
          text: Array.from({length: 203}, () => 'var<uniform> valuesExtra: f32;').join('\n')
        }
      ]
    };
    const model = buildKernelSourceModel(record, record.graphs[0].nodes[0]);
    expect(model).toMatchObject({truncatedLineCount: 3});
    if ('lines' in model) {
      expect(model.lines).toHaveLength(200);
      expect(model.lines[0].marks).toEqual([]);
    }
    expect(
      buildKernelSourceModel(
        {...record, refs: {sources: [], formulas: []}},
        record.graphs[0].nodes[0]
      )
    ).toMatchObject({status: 'unavailable'});
  });
});
