// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Adds a validated external MapLibre style to Kepler's native map-style picker.
 *
 * Kepler's own custom-style dialog fetches a style URL and then dispatches `inputMapStyle`,
 * `loadCustomMapStyle` and `addCustomMapStyle`. This module runs the same sequence from the host
 * after `basemap-context.ts` validated the style, so the style appears in the picker as a normal
 * custom entry (selected immediately) and Kepler persists it in the saved map config. Failed or
 * invalid styles never reach Kepler: the context reports the error and the local background stays.
 */

import {addCustomMapStyle, inputMapStyle, loadCustomMapStyle, wrapTo} from '@kepler.gl/actions';
import {
  type BasemapContext,
  type BasemapContextState,
  type BasemapStyleSpecification
} from './basemap-context';

export type BasemapPickerOptions = {
  dispatch: (action: unknown) => unknown;
  instanceId: string;
};

export type BasemapPicker = {
  /** Fetch hook for `makeBasemapContext` that remembers the validated style JSON per URL. */
  fetchStyle(url: string, signal: AbortSignal): Promise<BasemapStyleSpecification>;
  /** Validate the URL and, on success, add and select it in the Kepler picker. */
  addStyle(
    context: BasemapContext,
    url: string,
    label?: string
  ): Promise<{state: BasemapContextState; styleId: string | null}>;
};

/** Stable style id for a URL so adding the same URL twice updates one picker entry. */
export function makeExternalStyleId(url: string): string {
  let hash = 5381;
  for (let index = 0; index < url.length; index++)
    hash = ((hash * 33) ^ url.charCodeAt(index)) >>> 0;
  return `external-${hash.toString(36)}`;
}

export function makeBasemapPicker({dispatch, instanceId}: BasemapPickerOptions): BasemapPicker {
  const validatedStyles = new Map<string, BasemapStyleSpecification>();
  return {
    async fetchStyle(url, signal) {
      const response = await fetch(url, {signal});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const style = (await response.json()) as BasemapStyleSpecification;
      validatedStyles.set(url, style);
      return style;
    },
    async addStyle(context, url, label) {
      const state = await context.setStyleUrl(url);
      const style = validatedStyles.get(url);
      if (state.status !== 'external' || !style) return {state, styleId: null};
      const styleId = makeExternalStyleId(url);
      const styleLabel = label?.trim() || (style as {name?: string}).name || new URL(url).hostname;
      dispatch(wrapTo(instanceId, inputMapStyle({url, label: styleLabel, id: styleId})));
      dispatch(wrapTo(instanceId, loadCustomMapStyle({style: {...style, id: styleId}})));
      dispatch(wrapTo(instanceId, addCustomMapStyle()));
      return {state, styleId};
    }
  };
}
