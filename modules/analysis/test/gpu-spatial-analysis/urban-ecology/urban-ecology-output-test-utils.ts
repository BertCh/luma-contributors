// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {RecipeTestFixture} from '../recipes/recipe-harness';
export type Mirrored = {
  view: GraphDataView;
  buffer: ReturnType<RecipeTestFixture['output']>['buffer'];
};

/** The same nested shape with a caller-owned fixture output in place of each view. */
export function mirrorViews(
  fixture: RecipeTestFixture,
  value: unknown,
  path: string,
  into: Map<string, Mirrored>,
  bySource: Map<GraphDataView, Mirrored> = new Map()
): unknown {
  if (value instanceof GraphDataView) {
    const output =
      bySource.get(value) ??
      fixture.output(`caller${path}`, value.format as 'float32', value.length);
    bySource.set(value, output as Mirrored);
    into.set(path, output as Mirrored);
    return output.view;
  }
  if (Array.isArray(value)) {
    return value.map((entry, index) =>
      mirrorViews(fixture, entry, `${path}[${index}]`, into, bySource)
    );
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [
        key,
        mirrorViews(fixture, entry, `${path}.${key}`, into, bySource)
      ])
    );
  }
  return value;
}
