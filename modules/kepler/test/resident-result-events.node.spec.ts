// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import {Layer, type PickingInfo} from '@deck.gl/core';
import {withResidentResultEvents} from '../../../examples/deck/kepler-compute/src/resident-result-events';

class ResultLayer extends Layer {
  static layerName = 'ResultEventTestLayer';
  initializeState(): void {}
}

describe('independent analysis result events', () => {
  it('preserves original callbacks and clears native hover only when the pick is consumed', () => {
    const hover = vi.fn(() => true);
    const click = vi.fn(() => false);
    const clearNativeHover = vi.fn();
    const original = new ResultLayer({id: 'cartogram-result', onHover: hover, onClick: click});
    const layer = withResidentResultEvents(original, {isCurrent: () => true, clearNativeHover});
    const information = {index: 2} as PickingInfo;
    expect(layer.props.onHover(information, undefined!)).toBe(true);
    expect(hover).toHaveBeenCalledWith(information, undefined);
    expect(clearNativeHover).toHaveBeenCalledTimes(1);
    expect(layer.props.onClick(information, undefined!)).toBe(false);
    expect(click).toHaveBeenCalledWith(information, undefined);
    expect(clearNativeHover).toHaveBeenCalledTimes(1);
    expect(layer.id).toBe(original.id);
    expect('nativeLayerId' in layer.props).toBe(false);
  });

  it('consumes stale-owner events without inspecting old results or clearing new native hover', () => {
    const hover = vi.fn(() => true);
    const clearNativeHover = vi.fn();
    const layer = withResidentResultEvents(new ResultLayer({id: 'retired', onHover: hover}), {
      isCurrent: () => false,
      clearNativeHover
    });
    expect(layer.props.onHover({index: 0} as PickingInfo, undefined!)).toBe(true);
    expect(hover).not.toHaveBeenCalled();
    expect(clearNativeHover).not.toHaveBeenCalled();
  });
});
