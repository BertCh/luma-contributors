// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {KeplerResidentStyle, type KeplerResidentStyleProps} from './resident-style';
import type {KeplerResidentStylePlan} from './style-plan';

export type KeplerResidentStyleCacheProps = Omit<
  KeplerResidentStyleProps,
  'plan' | 'selectionMask' | 'workingMemoryBudgetBytes'
> & {
  workingMemoryBudgetBytesPerStyle: number;
};

/** Per-channel cache. Topology replacements remain leased until their explicit GPU completion. */
export class KeplerResidentStyleCache {
  private readonly styles = new Map<string, KeplerResidentStyle>();
  private readonly retirements = new Set<Promise<void>>();
  private readonly retirementErrors: unknown[] = [];
  private destroyed = false;
  private destruction?: Promise<void>;

  constructor(private readonly props: KeplerResidentStyleCacheProps) {}

  get(id: string): KeplerResidentStyle | undefined {
    return this.styles.get(id);
  }

  set(
    plan: KeplerResidentStylePlan,
    options: {
      selectionMask?: GPUVector<'uint32'>;
      sourceOverride?: GPUVector<'float32'>;
      retireAfter?: PromiseLike<unknown>;
    } = {}
  ): KeplerResidentStyle {
    if (this.destroyed) throw new Error('Style cache is destroyed');
    const previous = this.styles.get(plan.id);
    if (previous?.canUpdate(plan, options.selectionMask, options.sourceOverride)) {
      previous.update(plan);
      return previous;
    }
    if (previous && !options.retireAfter)
      throw new Error('Style replacement requires a submission fence');
    const {workingMemoryBudgetBytesPerStyle, ...props} = this.props;
    // Allocate and validate before replacing a valid cached style; failure leaves it usable.
    const style = new KeplerResidentStyle({
      ...props,
      plan,
      selectionMask: options.selectionMask,
      sourceOverride: options.sourceOverride,
      workingMemoryBudgetBytes: workingMemoryBudgetBytesPerStyle
    });
    this.styles.set(plan.id, style);
    if (previous) this.trackRetirement(previous.destroy(options.retireAfter!));
    return style;
  }

  remove(id: string, completion: PromiseLike<unknown>): Promise<void> {
    const style = this.styles.get(id);
    if (!style) return Promise.resolve();
    this.styles.delete(id);
    return this.trackRetirement(style.destroy(completion));
  }

  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    const pending = [...this.styles.values()].map(style => style.destroy(completion));
    this.styles.clear();
    this.destruction = Promise.allSettled([...this.retirements, ...pending]).then(results => {
      const rejected = results.find(result => result.status === 'rejected');
      if (rejected?.status === 'rejected') throw rejected.reason;
      if (this.retirementErrors.length) throw this.retirementErrors[0];
    });
    return this.destruction;
  }

  private trackRetirement(completion: Promise<void>): Promise<void> {
    this.retirements.add(completion);
    void completion.then(
      () => this.retirements.delete(completion),
      error => {
        this.retirements.delete(completion);
        this.retirementErrors.push(error);
      }
    );
    return completion;
  }
}
