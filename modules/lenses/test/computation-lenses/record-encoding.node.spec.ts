// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  compactLensRecordEnvelope,
  expandLensRecordEnvelope
} from '../../src/computation-lenses/record-encoding';

function base64Words(words: number[]): string {
  const bytes = new Uint8Array(words.length * 4);
  const view = new DataView(bytes.buffer);
  words.forEach((word, index) => view.setUint32(index * 4, word >>> 0, true));
  return `b64:${Buffer.from(bytes).toString('base64')}`;
}

/** The parts of a compacted or expanded envelope these specs read. */
type EnvelopeView = {
  encoding: {sourceLines: string[]; strings: string[]};
  record: {
    graphs: unknown[];
    refs: {sources: {textLines?: unknown}[]};
    captures: {snapshots: {values: string}[]; stamps: {values: string}[]};
  };
};

const sentinelTail = [7, 8, 9, ...new Array(40).fill(0xffffffff)];
const envelope = {
  schema: 'luma.lens-record/1',
  scenarioId: 'gpu-core.scan',
  record: {
    graphs: [
      {
        id: 'graph',
        nodes: [
          {id: 'scan-level-0', label: 'scan-level-0', reads: ['~tilde', '~~double'], attrs: {}},
          {id: 'scan-level-1', label: 'scan-level-1', reads: ['~tilde', 'values'], attrs: {}}
        ],
        edges: [{from: 'scan-level-0', to: 'scan-level-1', resource: 'values'}],
        list: [['~:not-a-shape', 1], []]
      }
    ],
    refs: {
      formulas: [],
      sources: [
        {id: 'wgsl:a', text: 'const a = 0u;\nfn main() {\n  body();\n}'},
        {id: 'wgsl:b', text: 'const a = 4u;\nfn main() {\n  body();\n}'},
        {id: 'wgsl:c'}
      ]
    },
    captures: {
      snapshots: [
        {id: 'first', elementType: 'u32', values: base64Words(sentinelTail)},
        {id: 'copy', elementType: 'u32', values: base64Words(sentinelTail)},
        {id: 'short', elementType: 'f32', values: base64Words([1, 2, 3])},
        {id: 'numbers', values: [1, 'NaN']}
      ],
      stamps: [{id: 'stamp', elementType: 'u32', values: base64Words(sentinelTail)}],
      replays: []
    }
  }
};

test('lens record compaction round-trips exactly', () => {
  const compacted = JSON.parse(JSON.stringify(compactLensRecordEnvelope(envelope)));
  expect(expandLensRecordEnvelope(compacted)).toEqual(envelope);
});

test('lens record compaction encodes runs, repeats, lines and graph strings', () => {
  const compacted = compactLensRecordEnvelope(envelope) as unknown as EnvelopeView;
  const [first, copy, short] = compacted.record.captures.snapshots;
  expect(first.values).toMatch(/^rle:[^,]+,\*40:ffffffff$/);
  expect(copy.values).toBe('same:snapshots.0');
  expect(short.values).toBe(envelope.record.captures.snapshots[2].values);
  expect(compacted.record.captures.stamps[0].values).toBe('same:snapshots.0');
  expect(compacted.encoding.sourceLines).toEqual([
    'const a = 0u;',
    'fn main() {',
    '  body();',
    '}',
    'const a = 4u;'
  ]);
  expect(compacted.record.refs.sources[1].textLines).toEqual([4, [1, 3]]);
  expect(compacted.record.refs.sources[2]).toEqual({id: 'wgsl:c'});
  expect(compacted.encoding.strings).toContain('~tilde');
  expect(JSON.stringify(compacted.record.graphs)).not.toContain('scan-level-0');
});

test('undefined graph members stay absent, as in plain JSON', () => {
  const withUndefined = {
    ...envelope,
    record: {...envelope.record, graphs: [{id: 'graph', allocation: undefined, nodes: []}]}
  };
  const json = JSON.stringify(compactLensRecordEnvelope(withUndefined));
  expect(json).not.toContain('null');
  expect(
    (expandLensRecordEnvelope(JSON.parse(json)) as unknown as EnvelopeView).record.graphs
  ).toEqual([{id: 'graph', nodes: []}]);
});

test('lens records without an encoding table are read unchanged', () => {
  const {record, ...rest} = envelope;
  expect(expandLensRecordEnvelope({...rest, record})).toEqual(envelope);
});

test('interned strings inside plain graph objects are restored', () => {
  const expanded = expandLensRecordEnvelope({
    schema: 'luma.lens-record/1',
    encoding: {strings: ['scan-level-0']},
    record: {graphs: [{id: '~0', nodes: [{id: '~0', label: '~~literal'}]}]}
  }) as unknown as EnvelopeView;
  expect(expanded.record.graphs[0]).toEqual({
    id: 'scan-level-0',
    nodes: [{id: 'scan-level-0', label: '~literal'}]
  });
});

test('UDF-04 delegates capture codec for every category and reads an existing archive', async () => {
  const codec = await import('@luma-contributors/analysis/computation-record');
  const {toRecordedLensFile, fromRecordedLensFile} = await import(
    '../../src/computation-lenses/registry'
  );
  const {makeY2Record, makeY2Capture} = await import(
    '../computation-views/fixtures/y2-card-records'
  );
  const {readFileSync} = await import('node:fs');
  const {vi} = await import('vitest');
  const serialize = vi.spyOn(codec, 'serializeGPUComputationCaptures');
  const deserialize = vi.spyOn(codec, 'deserializeGPUComputationCaptures');
  try {
    const record = makeY2Record([]);
    // Literal IEEE-754 words: 2, quiet NaN, +infinity, -infinity, negative zero.
    const words = new Uint32Array([0x40000000, 0x7fc12345, 0x7f800000, 0xff800000, 0x80000000]);
    for (const category of ['snapshots', 'stamps', 'replays'] as const) {
      record.captures![category] = [
        makeY2Capture(category, [], {
          values: new Float32Array(words.buffer),
          ...(category === 'replays' ? {provenance: 'gpu-replay'} : {})
        })
      ];
    }
    const file = toRecordedLensFile(record, {
      scenarioId: 'gpu-core.scan',
      recordedAt: '2026-11-20T00:00:00Z',
      adapterLabel: 'test'
    });
    const restored = fromRecordedLensFile(JSON.parse(JSON.stringify(file))).record;
    expect(serialize).toHaveBeenCalled();
    expect(deserialize).toHaveBeenCalled();
    for (const category of ['snapshots', 'stamps', 'replays'] as const) {
      const capture = restored.captures![category][0];
      const values = capture.values as Float32Array;
      expect(Array.from(new Uint32Array(values.buffer, values.byteOffset, values.length))).toEqual(
        Array.from(words)
      );
      expect(capture).not.toHaveProperty('role');
    }
    const archive = JSON.parse(
      readFileSync(
        new URL(
          '../../../../docs/lens-records/gpu-spatial-analysis.pair-gather.json',
          import.meta.url
        ),
        'utf8'
      )
    );
    expect(fromRecordedLensFile(archive).record.captures!.snapshots.length).toBeGreaterThan(0);
  } finally {
    serialize.mockRestore();
    deserialize.mockRestore();
  }
});

test('UDF-04 legacy arrays retain signed zero and tokens while malformed lanes reject', async () => {
  const {toRecordedLensFile, fromRecordedLensFile} = await import(
    '../../src/computation-lenses/registry'
  );
  const {makeY2Record, makeY2Capture} = await import(
    '../computation-views/fixtures/y2-card-records'
  );
  const record = makeY2Record([makeY2Capture('legacy', [1, 2, 3, 4, 5])]);
  const file = expandLensRecordEnvelope(
    toRecordedLensFile(record, {
      scenarioId: 'gpu-core.scan',
      recordedAt: '2026-11-20T00:00:00Z',
      adapterLabel: 'test'
    })
  ) as {
    record: {captures: {snapshots: {values: unknown; valueCount?: number}[]}};
  };
  const capture = file.record.captures.snapshots[0];
  delete capture.valueCount;
  capture.values = [2, 'NaN', 'Infinity', '-Infinity', -0];
  const values = fromRecordedLensFile(file).record.captures!.snapshots[0].values;
  expect(values[0]).toBe(2);
  expect(Number.isNaN(values[1])).toBe(true);
  expect(values[2]).toBe(Infinity);
  expect(values[3]).toBe(-Infinity);
  expect(Object.is(values[4], -0)).toBe(true);
  capture.values = [2, null];
  expect(() => fromRecordedLensFile(file)).toThrow();
  capture.values = [2, 3];
  capture.valueCount = 3;
  expect(() => fromRecordedLensFile(file)).toThrow('payload length does not match');
});

test('B10 unrelated schema-valued graph metadata is not a capture sidecar', async () => {
  const {toRecordedLensFile, fromRecordedLensFile} = await import(
    '../../src/computation-lenses/registry'
  );
  const {makeRecord} = await import('../computation-views/fixtures/v4-record');
  const record = makeRecord();
  record.graphs[0].attrs = {schema: 'luma.computation-captures/1', source: 'historical import'};
  const restored = fromRecordedLensFile(
    toRecordedLensFile(record, {
      scenarioId: 'gpu-core.scan',
      recordedAt: '2026-11-20T00:00:00Z',
      adapterLabel: 'test'
    })
  ).record;
  expect(restored.graphs[0].attrs).toEqual(record.graphs[0].attrs);
});

test('S2 shared capture save boundary rejects invalid lanes, component lengths and metadata', async () => {
  const {toRecordedLensFile} = await import('../../src/computation-lenses/registry');
  const {makeY2Record, makeY2Capture} = await import(
    '../computation-views/fixtures/y2-card-records'
  );
  const save = (metadata: Parameters<typeof makeY2Capture>[2]) =>
    toRecordedLensFile(makeY2Record([makeY2Capture('invalid', [1], metadata)]), {
      scenarioId: 'gpu-core.scan',
      recordedAt: '2026-11-20T00:00:00Z',
      adapterLabel: 'test'
    });
  for (const values of [[1.5], [-1], [4294967296]])
    expect(() => save({elementType: 'u32', values})).toThrow('invalid u32 lane');
  for (const values of [[1.5], [-2147483649], [2147483648]])
    expect(() => save({elementType: 'i32', values})).toThrow('invalid i32 lane');
  expect(() => save({components: 2, values: [1, 2, 3]})).toThrow('whole components');
  expect(() => save({name: ''})).toThrow('nonempty string');
  expect(() => save({id: ''})).toThrow('nonempty string');
  expect(() => save({components: 0})).toThrow('positive safe integer');
});

test('W5-22 file loading validates capture IDs across every sidecar category', async () => {
  const {toRecordedLensFile, fromRecordedLensFile} = await import(
    '../../src/computation-lenses/registry'
  );
  const {makeY2Record, makeY2Capture} = await import(
    '../computation-views/fixtures/y2-card-records'
  );
  for (const category of ['snapshots', 'stamps', 'replays'] as const) {
    const record = makeY2Record([makeY2Capture('first', [1], {id: 'duplicate'})]);
    record.captures![category] = [
      ...record.captures![category],
      makeY2Capture('second', [2], {
        id: 'duplicate',
        provenance: category === 'replays' ? 'gpu-replay' : 'gpu-readback'
      })
    ];
    const file = toRecordedLensFile(record, {
      scenarioId: 'gpu-core.scan',
      recordedAt: record.createdAt,
      adapterLabel: 'fixture'
    });
    expect(() => fromRecordedLensFile(JSON.parse(JSON.stringify(file)))).toThrow(
      'Duplicate capture id'
    );
  }
});
