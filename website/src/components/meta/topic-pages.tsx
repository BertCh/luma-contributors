// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Routes a topic to its page. Each page lives in `pages/<slug>/index.tsx` and exports `Page`.

import {Page as AnatomyPage} from './pages/anatomy';
import {Page as BuildingBlocksPage} from './pages/building-blocks';
import {Page as CommandGraphPage} from './pages/command-graph';
import {Page as ContractPage} from './pages/contract';
import {Page as DataOnTheGpuPage} from './pages/data-on-the-gpu';
import {Page as GlossaryPage} from './pages/glossary';
import {Page as IterationPage} from './pages/iteration';
import {Page as KernelsPage} from './pages/kernels';
import {Page as LensesPage} from './pages/lenses';
import {Page as SchedulingPage} from './pages/scheduling';
import {Page as ThisPagePage} from './pages/this-page';
import {Page as UpstreamFirstPage} from './pages/upstream-first';
import {Page as WorkflowsPage} from './pages/workflows';
import type {TopicPageProps, TopicSlug} from './topic-types';

export type {TopicCode, TopicModels} from './topic-types';

const PAGES: Record<TopicSlug, (props: TopicPageProps) => React.ReactNode> = {
  anatomy: AnatomyPage,
  'data-on-the-gpu': DataOnTheGpuPage,
  'command-graph': CommandGraphPage,
  scheduling: SchedulingPage,
  kernels: KernelsPage,
  'building-blocks': BuildingBlocksPage,
  iteration: IterationPage,
  contract: ContractPage,
  workflows: WorkflowsPage,
  lenses: LensesPage,
  'this-page': ThisPagePage,
  'upstream-first': UpstreamFirstPage,
  glossary: GlossaryPage
};

export function TopicPage(props: TopicPageProps) {
  const Page = PAGES[props.topic.slug];
  return <Page {...props} />;
}
