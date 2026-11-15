// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {MapView, WebMercatorViewport, type MapViewState, type View} from '@deck.gl/core';
import {
  buildResidentSplitDeckProps,
  type ResidentSplitDeckProps
} from '../../../examples/deck/kepler-compute/src/split-deck-props';

const viewStates = {
  'view-0': {longitude: -80, latitude: 40, zoom: 11},
  'view-1': {longitude: -118, latitude: 34, zoom: 12}
};

function createSplitProps(): ResidentSplitDeckProps {
  return {
    views: [
      new MapView({id: 'view-0', x: 0, width: '50%'}),
      new MapView({id: 'view-1', x: '50%', width: '50%'})
    ],
    controller: {dragRotate: false},
    width: 800,
    height: 400,
    layers: []
  };
}

/** Mock only Deck's props merge boundary; use real public Views for cloning and projection. */
function expectUsableViews(props: ResidentSplitDeckProps, expectedIds: string[]): void {
  const views = (Array.isArray(props.views) ? props.views : [props.views]) as View[];
  expect(views.map(view => view.id)).toEqual(expectedIds);
  for (const view of views) {
    expect(view).toBeInstanceOf(MapView);
    // Deck resolves the top-level controller by cloning a view. An undefined entry fails here
    // just as it does in Deck.setProps, without constructing a device or using private APIs.
    const controlledView = view.clone({controller: props.controller});
    const state = props.viewState as unknown as Record<string, MapViewState>;
    const viewport = controlledView.makeViewport({
      width: 800,
      height: 400,
      viewState: state[view.id] ?? (props.viewState as MapViewState)
    });
    expect(viewport?.id).toBe(view.id);
    expect(viewport).toBeInstanceOf(WebMercatorViewport);
    expect((viewport as WebMercatorViewport).longitude).toBe(
      (state[view.id] ?? (props.viewState as MapViewState)).longitude
    );
  }
}

test('layers-only retirement retains valid split views, controllers and callbacks', () => {
  const primaryHover = vi.fn();
  const secondaryHover = vi.fn();
  const current = buildResidentSplitDeckProps(
    {...createSplitProps(), onHover: primaryHover},
    viewStates,
    () => ({onHover: secondaryHover})
  );
  const previousViews = current.views;
  const previousHover = current.onHover;
  for (const update of [{layers: []}, {}, {width: 900}, {layers: [], height: 500}]) {
    const prepared = buildResidentSplitDeckProps(update, viewStates, () => ({}));
    expect(prepared).not.toHaveProperty('views');
    expect(prepared).not.toHaveProperty('controller');
    expect(prepared).not.toHaveProperty('onHover');
    Object.assign(current, prepared);
    expect(current.views).toBe(previousViews);
    expect(current.onHover).toBe(previousHover);
    expectUsableViews(current, ['view-0', 'view-1']);
  }
  const views = current.views as MapView[];
  for (const view of views)
    expect(view.props.controller).toEqual({dragRotate: false, touchRotate: true});
  current.onHover!({viewport: {id: 'view-0'}} as never, {} as never);
  current.onHover!({viewport: {id: 'view-1'}} as never, {} as never);
  expect(primaryHover).toHaveBeenCalledTimes(1);
  expect(secondaryHover).toHaveBeenCalledTimes(1);
});

test('controller-only updates never replace omitted views with undefined', () => {
  for (const states of [null, viewStates]) {
    const prepared = buildResidentSplitDeckProps({controller: true}, states, () => null);
    expect(prepared.controller).toEqual({touchRotate: true});
    expect(prepared).not.toHaveProperty('views');
  }
});

test('split apply, single replacement, split rollback and retirement retain usable view states', () => {
  const singleProps: ResidentSplitDeckProps = {
    views: new MapView({id: 'view-0'}),
    viewState: viewStates['view-0'],
    controller: true,
    layerFilter: null
  };
  const current: ResidentSplitDeckProps = {};
  const steps = [
    {props: singleProps, states: null, ids: ['view-0']},
    {props: createSplitProps(), states: viewStates, ids: ['view-0', 'view-1']},
    {props: {layers: []}, states: viewStates, ids: ['view-0', 'view-1']},
    {props: singleProps, states: null, ids: ['view-0']},
    {props: {layers: []}, states: null, ids: ['view-0']},
    {props: createSplitProps(), states: viewStates, ids: ['view-0', 'view-1']},
    {props: {layers: []}, states: viewStates, ids: ['view-0', 'view-1']}
  ];
  for (const {props, states, ids} of steps) {
    Object.assign(
      current,
      buildResidentSplitDeckProps(props, states, () => null)
    );
    expectUsableViews(current, ids);
  }
});

test('incremental events keep the primary callback and resolve the current secondary callback', () => {
  const firstPrimary = vi.fn();
  const secondPrimary = vi.fn();
  const firstSecondary = vi.fn();
  const secondSecondary = vi.fn();
  let secondary: ResidentSplitDeckProps = {
    onClick: firstSecondary,
    onViewStateChange: firstSecondary
  };
  const current = buildResidentSplitDeckProps(
    {...createSplitProps(), onClick: firstPrimary, onViewStateChange: firstPrimary},
    viewStates,
    () => secondary
  );
  Object.assign(
    current,
    buildResidentSplitDeckProps({layers: []}, viewStates, () => secondary)
  );
  current.onClick!({viewport: {id: 'view-0'}} as never, {} as never);
  current.onViewStateChange!({viewId: 'view-1'} as never);
  expect(firstPrimary).toHaveBeenCalledTimes(1);
  expect(firstSecondary).toHaveBeenCalledTimes(1);
  secondary = {onClick: secondSecondary};
  current.onClick!({viewport: {id: 'view-1'}} as never, {} as never);
  expect(secondSecondary).toHaveBeenCalledTimes(1);
  Object.assign(
    current,
    buildResidentSplitDeckProps({onClick: secondPrimary}, viewStates, () => secondary)
  );
  current.onClick!({viewport: {id: 'view-0'}} as never, {} as never);
  expect(secondPrimary).toHaveBeenCalledTimes(1);
  expect(firstPrimary).toHaveBeenCalledTimes(1);
});

test('disabled view controllers and original input arrays remain unchanged', () => {
  const views = Object.freeze([
    new MapView({id: 'view-0', controller: false}),
    new MapView({id: 'view-1', controller: false})
  ]);
  const props = Object.freeze({views: [...views], controller: true});
  const prepared = buildResidentSplitDeckProps(props, viewStates, () => null);
  expect((prepared.views as View[]).map(view => view.props.controller)).toEqual([false, false]);
  expect(props.views).toEqual(views);
  expect(props.controller).toBe(true);
  expect(prepared.views).not.toBe(props.views);
});
