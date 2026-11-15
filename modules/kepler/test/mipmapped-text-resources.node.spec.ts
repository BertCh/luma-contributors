// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {FontAtlas} from '@luma.gl/engine/text';
import {snapshotKeplerFontAtlas} from '../../../examples/deck/kepler-compute/src/luma-gaps/mipmapped-text-resources';

function createStubCanvas(width: number, height: number) {
  const draws: unknown[] = [];
  return {
    width,
    height,
    draws,
    getContext: () => ({drawImage: (source: unknown) => draws.push(source)})
  };
}

test('snapshotKeplerFontAtlas copies pages at the validated size so shared canvas growth is not uploaded', () => {
  const shared = createStubCanvas(1024, 128);
  const fontAtlas = {
    width: 1024,
    height: 128,
    pages: [shared],
    mapping: {}
  } as unknown as FontAtlas;
  const snapshot = snapshotKeplerFontAtlas(fontAtlas, (width, height) =>
    createStubCanvas(width, height)
  );
  // Another runtime enlarges the shared cached canvas after admission.
  shared.height = 4096;
  const page = snapshot.pages[0] as unknown as ReturnType<typeof createStubCanvas>;
  expect(page).not.toBe(shared);
  expect([page.width, page.height]).toEqual([1024, 128]);
  expect(page.draws).toEqual([shared]);
  expect(snapshot.width).toBe(1024);
  expect(snapshot.mapping).toBe(fontAtlas.mapping);
});

test('snapshotKeplerFontAtlas keeps pages it cannot copy', () => {
  const page = {width: 4, height: 4};
  const fontAtlas = {width: 4, height: 4, pages: [page]} as unknown as FontAtlas;
  expect(snapshotKeplerFontAtlas(fontAtlas, () => null).pages[0]).toBe(page);
});
