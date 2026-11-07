// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ExampleSupportDefinition} from '../../example-support';

const exampleSupport = {
  id: 'deck/kepler-compute',
  mobileMode: 'unsupported',
  mobileProfile: 'large-data',
  unsupportedReason: 'The Kepler compute sandbox requires a desktop layout and WebGPU.',
  requirements: {backends: ['webgpu']}
} satisfies ExampleSupportDefinition;

export default exampleSupport;
