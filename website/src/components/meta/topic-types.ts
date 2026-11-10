// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Types shared by the topic registry, the page kit and every page folder. Plain data only, so a
// page's data.ts can be imported by route loaders without pulling in components.

import type {LensViewModel} from '../../lens/types';
import type {TopicSlug} from './topic-slugs';

export type {TopicSlug} from './topic-slugs';

export type TopicPartId = 'run' | 'parallel' | 'practice' | 'system';

/** What a page needs loaded before it renders. Lives in `pages/<slug>/data.ts`. */
export type TopicPageData = {
  /** Lens scenario ids whose view models the page draws. */
  scenarios: readonly string[];
  /**
   * Code the page shows, by name. Each value is the raw text of a typechecked `*.example.ts` file
   * (import it with `?raw`); the loader trims and highlights it. See examples.ts.
   */
  examples: Readonly<Record<string, string>>;
};

export type Topic = {
  slug: TopicSlug;
  part: TopicPartId;
  title: string;
  /** One sentence, shown under the title and on the index card. */
  lead: string;
} & TopicPageData;

export type TopicModels = Record<string, LensViewModel | undefined>;

/** A loaded example: the trimmed source and its highlighted HTML (null when highlighting failed). */
export type TopicExample = {code: string; html: string | null};
export type TopicCode = Record<string, TopicExample>;

/** Props of every page component in `pages/<slug>/index.tsx`. */
export type TopicPageProps = {topic: Topic; models: TopicModels; code: TopicCode};
