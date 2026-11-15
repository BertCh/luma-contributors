// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {WorkflowHistory} from '../../src/workflows/workflow-history';
import {WorkflowError} from '../../src/workflows/workflow-types';
import {makeFixtureDocument} from './workflow-fixtures';

describe('WorkflowHistory', () => {
  test('undoes and redoes snapshots by reference without changing them', () => {
    const initial = makeFixtureDocument();
    const first = {...initial, title: 'First'};
    const second = {...initial, title: 'Second'};
    const snapshots = [initial, first, second].map(document => structuredClone(document));
    const history = new WorkflowHistory(initial);
    expect(history.document).toBe(initial);
    expect(history.canUndo).toBe(false);
    expect(history.canRedo).toBe(false);
    expect(history.undo()).toBe(false);
    expect(history.redo()).toBe(false);
    history.apply(first);
    history.apply(second);
    expect(history.document).toBe(second);
    expect(history.canUndo).toBe(true);
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(first);
    expect(history.canRedo).toBe(true);
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(initial);
    expect(history.undo()).toBe(false);
    expect(history.redo()).toBe(true);
    expect(history.document).toBe(first);
    expect(history.redo()).toBe(true);
    expect(history.document).toBe(second);
    expect(history.redo()).toBe(false);
    expect(history.canRedo).toBe(false);
    expect([initial, first, second]).toEqual(snapshots);
    expect(first.steps).toBe(initial.steps);
  });

  test('coalesces a sliding window of matching non-empty tags using an injectable clock', () => {
    let time = 0;
    const initial = makeFixtureDocument();
    const history = new WorkflowHistory(initial, {now: () => time, coalesceMilliseconds: 100});
    const first = {...initial, title: 'First'};
    const second = {...initial, title: 'Second'};
    const third = {...initial, title: 'Third'};
    history.apply(first, 'move');
    time = 100;
    history.apply(second, 'move');
    time = 200;
    history.apply(third, 'move');
    expect(history.document).toBe(third);
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(initial);
    expect(history.canUndo).toBe(false);
    expect(history.redo()).toBe(true);
    expect(history.document).toBe(third);
  });

  test.each([undefined, ''])('does not coalesce empty tag %s', tag => {
    const initial = makeFixtureDocument();
    const history = new WorkflowHistory(initial, {now: () => 0});
    const first = {...initial, title: 'First'};
    history.apply(first, tag);
    history.apply({...initial, title: 'Second'}, tag);
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(first);
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(initial);
  });

  test('separates tags, expired windows and clocks moving backwards', () => {
    let time = 0;
    const initial = makeFixtureDocument();
    const history = new WorkflowHistory(initial, {now: () => time});
    const documents = Array.from({length: 4}, (_, index) => ({...initial, title: `Edit ${index}`}));
    history.apply(documents[0], 'move');
    history.apply(documents[1], 'parameter');
    time = 501;
    history.apply(documents[2], 'parameter');
    time = 499;
    history.apply(documents[3], 'parameter');
    for (const document of [documents[2], documents[1], documents[0], initial]) {
      expect(history.undo()).toBe(true);
      expect(history.document).toBe(document);
    }
    expect(history.undo()).toBe(false);
  });

  test('same-object apply is a complete no-op, preserving redo and the coalescing clock', () => {
    let time = 0;
    let calls = 0;
    const initial = makeFixtureDocument();
    const history = new WorkflowHistory(initial, {
      now: () => {
        calls++;
        return time;
      }
    });
    const first = {...initial, title: 'First'};
    history.apply(first, 'move');
    time = 400;
    history.apply(first, 'other');
    expect(calls).toBe(1);
    time = 501;
    history.apply({...initial, title: 'Second'}, 'move');
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(first);
    history.apply(first, 'noop');
    expect(history.canRedo).toBe(true);
    expect(calls).toBe(2);
  });

  test('a new edit after undo clears redo and starts a fresh coalescing group', () => {
    const initial = makeFixtureDocument();
    const history = new WorkflowHistory(initial, {now: () => 0});
    const first = {...initial, title: 'First'};
    const second = {...initial, title: 'Second'};
    history.apply(first, 'move');
    history.apply(second, 'other');
    history.undo();
    const branched = {...initial, title: 'Branched'};
    history.apply(branched, 'move');
    expect(history.canRedo).toBe(false);
    expect(history.redo()).toBe(false);
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(first);
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(initial);
  });

  test('a new edit after redo does not coalesce with the redone edit', () => {
    const initial = makeFixtureDocument();
    const history = new WorkflowHistory(initial, {now: () => 0});
    const first = {...initial, title: 'First'};
    history.apply(first, 'move');
    history.undo();
    history.redo();
    history.apply({...initial, title: 'Second'}, 'move');
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(first);
  });

  test('bounds the undo stack and preserves the bound after redo', () => {
    const initial = makeFixtureDocument();
    const history = new WorkflowHistory(initial, {maximumEntries: 2});
    const documents = Array.from({length: 4}, (_, index) => ({...initial, title: `Edit ${index}`}));
    documents.forEach(document => history.apply(document));
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(documents[2]);
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(documents[1]);
    expect(history.undo()).toBe(false);
    history.redo();
    history.redo();
    history.apply({...initial, title: 'Last'});
    history.undo();
    expect(history.document).toBe(documents[3]);
    history.undo();
    expect(history.document).toBe(documents[2]);
    expect(history.undo()).toBe(false);
  });

  test('defaults to 100 undo entries and allows zero entries', () => {
    const initial = makeFixtureDocument();
    const history = new WorkflowHistory(initial);
    for (let index = 0; index < 101; index++) history.apply({...initial, title: `Edit ${index}`});
    for (let index = 0; index < 100; index++) expect(history.undo()).toBe(true);
    expect(history.document.title).toBe('Edit 0');
    expect(history.undo()).toBe(false);
    const disabled = new WorkflowHistory(initial, {maximumEntries: 0, now: () => 0});
    const updated = {...initial, title: 'Updated'};
    disabled.apply(updated, 'move');
    expect(disabled.document).toBe(updated);
    expect(disabled.canUndo).toBe(false);
    expect(disabled.undo()).toBe(false);
  });

  test('reset replaces the current snapshot and clears both stacks and coalescing', () => {
    const initial = makeFixtureDocument();
    const history = new WorkflowHistory(initial, {now: () => 0});
    history.apply({...initial, title: 'First'}, 'move');
    history.apply({...initial, title: 'Second'}, 'other');
    history.undo();
    const replacement = {...initial, title: 'Replacement'};
    history.reset(replacement);
    expect(history.document).toBe(replacement);
    expect(history.canUndo).toBe(false);
    expect(history.canRedo).toBe(false);
    history.apply({...initial, title: 'New edit'}, 'move');
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(replacement);
    expect(history.undo()).toBe(false);
  });
});

describe('review regressions: finite history configuration (F4)', () => {
  test.each([NaN, Infinity, -Infinity])('rejects non-finite options (%s)', value => {
    const document = makeFixtureDocument();
    for (const name of ['maximumEntries', 'coalesceMilliseconds'] as const) {
      expect(() => new WorkflowHistory(document, {[name]: value})).toThrow(WorkflowError);
      try {
        new WorkflowHistory(document, {[name]: value});
      } catch (error) {
        expect(error).toMatchObject({code: 'parameter-out-of-range'});
        expect((error as WorkflowError).message).toContain(`${name} must be a finite number`);
      }
    }
  });

  test.each([
    NaN,
    Infinity,
    -Infinity
  ])('separates edits around invalid clock readings (%s)', invalid => {
    let time = 0;
    const initial = makeFixtureDocument();
    const history = new WorkflowHistory(initial, {now: () => time});
    const documents = ['First', 'Invalid clock', 'Still invalid', 'Recovered'].map(title => ({
      ...initial,
      title
    }));
    history.apply(documents[0], 'move');
    time = invalid;
    history.apply(documents[1], 'move');
    history.apply(documents[2], 'move');
    time = 1;
    history.apply(documents[3], 'move');
    for (const document of [documents[2], documents[1], documents[0], initial]) {
      expect(history.undo()).toBe(true);
      expect(history.document).toBe(document);
    }
    expect(history.undo()).toBe(false);
  });
});
