// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ShapeDescriptorsOperationProps} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors-operation';

type ShapeOutputSelection = ShapeDescriptorsOperationProps['output'];

/**
 * Every column, then for each of six terminal columns that column alone and every column except
 * that one. Written out and typed as the operation's own `output`, so each case compiles against
 * the contract it exercises and an omitted column is an absent key.
 */
export const SHAPE_OUTPUT_CASES: ShapeOutputSelection[] = [
  {
    areas: 'areas',
    perimeters: 'perimeters',
    polsbyPopper: 'polsbyPopper',
    schwartzberg: 'schwartzberg',
    elongation: 'elongation',
    orientation: 'orientation',
    convexity: 'convexity',
    clockwise: 'clockwise',
    sliver: 'sliver'
  },
  {areas: 'areas'},
  {
    perimeters: 'perimeters',
    polsbyPopper: 'polsbyPopper',
    schwartzberg: 'schwartzberg',
    elongation: 'elongation',
    orientation: 'orientation',
    convexity: 'convexity',
    clockwise: 'clockwise',
    sliver: 'sliver'
  },
  {perimeters: 'perimeters'},
  {
    areas: 'areas',
    polsbyPopper: 'polsbyPopper',
    schwartzberg: 'schwartzberg',
    elongation: 'elongation',
    orientation: 'orientation',
    convexity: 'convexity',
    clockwise: 'clockwise',
    sliver: 'sliver'
  },
  {polsbyPopper: 'polsbyPopper'},
  {
    areas: 'areas',
    perimeters: 'perimeters',
    schwartzberg: 'schwartzberg',
    elongation: 'elongation',
    orientation: 'orientation',
    convexity: 'convexity',
    clockwise: 'clockwise',
    sliver: 'sliver'
  },
  {schwartzberg: 'schwartzberg'},
  {
    areas: 'areas',
    perimeters: 'perimeters',
    polsbyPopper: 'polsbyPopper',
    elongation: 'elongation',
    orientation: 'orientation',
    convexity: 'convexity',
    clockwise: 'clockwise',
    sliver: 'sliver'
  },
  {clockwise: 'clockwise'},
  {
    areas: 'areas',
    perimeters: 'perimeters',
    polsbyPopper: 'polsbyPopper',
    schwartzberg: 'schwartzberg',
    elongation: 'elongation',
    orientation: 'orientation',
    convexity: 'convexity',
    sliver: 'sliver'
  },
  {sliver: 'sliver'},
  {
    areas: 'areas',
    perimeters: 'perimeters',
    polsbyPopper: 'polsbyPopper',
    schwartzberg: 'schwartzberg',
    elongation: 'elongation',
    orientation: 'orientation',
    convexity: 'convexity',
    clockwise: 'clockwise'
  }
];
