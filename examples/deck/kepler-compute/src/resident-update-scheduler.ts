// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Finish the current native filter transaction before notifying consumers or rendering. */
export class ResidentUpdateScheduler {
  private pending = false;
  private generation = 0;

  constructor(
    private readonly isCurrent: () => boolean,
    private readonly flush: () => void
  ) {}

  request(): void {
    if (this.pending || !this.isCurrent()) return;
    this.pending = true;
    const generation = this.generation;
    queueMicrotask(() => {
      if (generation !== this.generation) return;
      this.pending = false;
      if (this.isCurrent()) this.flush();
    });
  }

  cancel(): void {
    this.generation++;
    this.pending = false;
  }
}
