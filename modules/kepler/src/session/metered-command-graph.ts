// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';

/**
 * GPUCommandGraph that remembers its compiled transient bytes. Published compiled queries keep
 * their graph private, so passing this graph to a query compile is how Kepler accounts for scratch.
 */
export class MeteredCommandGraph<Parameters = void> extends GPUCommandGraph<Parameters> {
  /** Physical transient bytes of the last compilation, 0 before compile. */
  transientByteLength = 0;

  override compile(): CompiledGPUCommandGraph<Parameters> {
    const compiled = super.compile();
    this.transientByteLength = compiled.stats.physicalTransientResourceBytes;
    return compiled;
  }
}
