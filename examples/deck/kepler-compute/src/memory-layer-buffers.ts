// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Structural view of the deck.gl layer internals needed to count owned GPU buffers. */
type MeasurableBuffer = {readonly byteLength?: number};
type MeasurableAttribute = {
  readonly buffer?: MeasurableBuffer | null;
  readonly state?: {readonly externalBuffer?: unknown};
};
export type MeasurableDeckLayer = {
  readonly id: string;
  readonly parent?: MeasurableDeckLayer | null;
  readonly state?: {
    readonly attributeManager?: {getAttributes?(): Record<string, MeasurableAttribute>};
  } | null;
  readonly internalState?: {
    readonly attributeManager?: {getAttributes?(): Record<string, MeasurableAttribute>};
  } | null;
};

export type LayerRenderBuffers = Readonly<{bytes: number; bufferCount: number}>;

function getAttributes(layer: MeasurableDeckLayer): MeasurableAttribute[] {
  try {
    const manager = layer.state?.attributeManager ?? layer.internalState?.attributeManager;
    return Object.values(manager?.getAttributes?.() ?? {});
  } catch {
    // A layer that is mid-finalization has no readable attributes; count it as zero.
    return [];
  }
}

/**
 * Bytes of GPU buffers each Kepler layer owns through its deck.gl layers (attributes the layer
 * allocated itself, including composite sublayers). Buffers supplied from outside, such as registry
 * GPUData borrowed by resident layers, are excluded because the dataset rows already count them.
 * A buffer shared by several attributes (interleaved) is counted once.
 */
export function measureLayerRenderBuffers(
  deckLayers: readonly MeasurableDeckLayer[],
  keplerLayerIds: readonly string[]
): ReadonlyMap<string, LayerRenderBuffers> {
  const owners = new Set(keplerLayerIds);
  const seen = new Set<MeasurableBuffer>();
  const totals = new Map<string, {bytes: number; bufferCount: number}>(
    keplerLayerIds.map(id => [id, {bytes: 0, bufferCount: 0}])
  );
  for (const deckLayer of deckLayers) {
    let ownerId: string | undefined;
    for (
      let layer: MeasurableDeckLayer | null | undefined = deckLayer;
      layer;
      layer = layer.parent
    ) {
      if (owners.has(layer.id)) {
        ownerId = layer.id;
        break;
      }
    }
    // Deck sublayer ids extend their parent's id when the parent link is not available yet.
    ownerId ??= keplerLayerIds.find(id => deckLayer.id.startsWith(`${id}-`));
    const total = ownerId ? totals.get(ownerId) : undefined;
    if (!total) continue;
    for (const attribute of getAttributes(deckLayer)) {
      const buffer = attribute.buffer;
      if (!buffer || attribute.state?.externalBuffer || seen.has(buffer)) continue;
      seen.add(buffer);
      total.bytes += buffer.byteLength ?? 0;
      total.bufferCount++;
    }
  }
  return totals;
}
