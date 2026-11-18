// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GPUCommandGraphCompileOptions
} from '@luma.gl/graph/gpu-core';

/**
 * GPUCommandGraph that remembers its compiled transient bytes. Published compiled queries keep
 * their graph private, so passing this graph to a query compile is how Kepler accounts for scratch.
 * Both the synchronous and the asynchronous compile record the same statistic.
 */
export class MeteredCommandGraph<Parameters = void> extends GPUCommandGraph<Parameters> {
  /** Physical transient bytes of the last compilation, 0 before compile. */
  transientByteLength = 0;

  override compile(options?: GPUCommandGraphCompileOptions): CompiledGPUCommandGraph<Parameters> {
    const compiled = super.compile(options);
    this.transientByteLength = compiled.stats.physicalTransientResourceBytes;
    return compiled;
  }

  override async compileAsync(
    options?: GPUCommandGraphCompileOptions
  ): Promise<CompiledGPUCommandGraph<Parameters>> {
    const compiled = await super.compileAsync(options);
    this.transientByteLength = compiled.stats.physicalTransientResourceBytes;
    return compiled;
  }
}
