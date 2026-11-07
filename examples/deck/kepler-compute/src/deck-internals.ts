// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Deck} from '@deck.gl/core';
import type {MeasurableDeckLayer} from './memory-layer-buffers';

/** Layer view used by dev probes: the layer id plus the models it renders with. */
export type DeckProbeLayer = MeasurableDeckLayer & {getModels?: () => unknown[]};

/** The parts of a Deck instance's internal state that diagnostics and memory accounting read. */
export type DeckInternals = {
  readonly layerManager?: {getLayers(): DeckProbeLayer[]};
  readonly effectManager?: {
    readonly effects?: readonly {id?: string; constructor: {name: string}}[];
  };
  readonly props?: {readonly layers?: unknown};
};

/**
 * Typed read-only view of Deck internals that its public typings do not declare. This is the one
 * place that depends on deck.gl private state, so a Deck upgrade has a single spot to revisit.
 */
export function getDeckInternals(deck: Deck | null | undefined): DeckInternals | undefined {
  // Deck declares these members privately; the structural view above is what the example reads.
  return deck as unknown as DeckInternals | undefined;
}

/** Layers Deck currently has instantiated, or an empty list before the first render. */
export function getDeckLayers(deck: Deck | null | undefined): DeckProbeLayer[] {
  return getDeckInternals(deck)?.layerManager?.getLayers() ?? [];
}
