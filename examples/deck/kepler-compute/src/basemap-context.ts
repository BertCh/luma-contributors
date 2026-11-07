// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Optional geographic context for the host map.
 *
 * The default is the token-free local background: no network access and no attribution
 * requirement. A configured external MapLibre style URL is used only after it has been fetched and
 * validated. When it fails, the local background stays in effect, the error is surfaced as a
 * message, and no dataset, layer or filter state is touched. Attribution strings declared by an
 * external style are always preserved and exposed so the host can render them.
 */

export type BasemapStyleSpecification = {
  version?: number;
  sources?: Record<string, {attribution?: string} | undefined>;
  layers?: unknown[];
};

export type BasemapContextState = {
  /** The style currently applied: the external URL, or null for the local background. */
  activeStyleUrl: string | null;
  /** Configured external URL, even when it failed to load. */
  configuredStyleUrl: string | null;
  /** Message to show when the configured style failed; null otherwise. */
  errorMessage: string | null;
  /** Distinct attribution text of the active style, in source order. */
  attributions: string[];
  status: 'local' | 'loading' | 'external' | 'failed';
};

export type BasemapContextOptions = {
  fetchStyle?: (url: string, signal: AbortSignal) => Promise<BasemapStyleSpecification>;
  onChange?: (state: BasemapContextState) => void;
  /** Attribution kept for the local background; empty by default. */
  localAttribution?: string;
};

export type BasemapContext = {
  getState(): BasemapContextState;
  /** Configure an external style URL, or pass null/empty to return to the local background. */
  setStyleUrl(url: string | null): Promise<BasemapContextState>;
  destroy(): void;
};

async function fetchStyleJson(
  url: string,
  signal: AbortSignal
): Promise<BasemapStyleSpecification> {
  const response = await fetch(url, {signal});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()) as BasemapStyleSpecification;
}

/** Collect the distinct, trimmed attribution text declared by a style's sources. */
export function readStyleAttributions(style: BasemapStyleSpecification): string[] {
  const attributions: string[] = [];
  for (const source of Object.values(style.sources ?? {})) {
    const text = source?.attribution?.trim();
    if (text && !attributions.includes(text)) attributions.push(text);
  }
  return attributions;
}

export function makeBasemapContext(options: BasemapContextOptions = {}): BasemapContext {
  const fetchStyle = options.fetchStyle ?? fetchStyleJson;
  const localAttributions = options.localAttribution ? [options.localAttribution] : [];
  let state: BasemapContextState = {
    activeStyleUrl: null,
    configuredStyleUrl: null,
    errorMessage: null,
    attributions: localAttributions,
    status: 'local'
  };
  let controller: AbortController | null = null;
  let destroyed = false;

  const update = (next: Partial<BasemapContextState>): BasemapContextState => {
    state = {...state, ...next};
    if (!destroyed) options.onChange?.(state);
    return state;
  };

  return {
    getState: () => state,
    async setStyleUrl(url) {
      controller?.abort();
      const configured = url?.trim() || null;
      if (!configured) {
        controller = null;
        return update({
          activeStyleUrl: null,
          configuredStyleUrl: null,
          errorMessage: null,
          attributions: localAttributions,
          status: 'local'
        });
      }
      const current = new AbortController();
      controller = current;
      update({configuredStyleUrl: configured, errorMessage: null, status: 'loading'});
      try {
        const style = await fetchStyle(configured, current.signal);
        // A newer request owns the state; a stale result must not overwrite it.
        if (current !== controller || destroyed) return state;
        if (!Array.isArray(style?.layers) || typeof style.sources !== 'object') {
          throw new Error('not a MapLibre style (missing layers or sources)');
        }
        return update({
          activeStyleUrl: configured,
          errorMessage: null,
          attributions: readStyleAttributions(style),
          status: 'external'
        });
      } catch (error) {
        if (current !== controller || destroyed) return state;
        const reason = error instanceof Error ? error.message : String(error);
        // Keep the previous local background and all data; only explain the failure.
        return update({
          activeStyleUrl: null,
          errorMessage: `Basemap style could not be loaded (${reason}). Showing the local background; your data is unchanged.`,
          attributions: localAttributions,
          status: 'failed'
        });
      }
    },
    destroy() {
      destroyed = true;
      controller?.abort();
      controller = null;
    }
  };
}
