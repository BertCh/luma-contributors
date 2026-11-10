// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

declare module 'virtual:docs-index' {
  const docsIndex: {slug: string; title: string}[];
  export default docsIndex;
}

declare module 'virtual:catalog-summaries' {
  import type {CatalogSummary} from './catalog/index';
  const catalogSummaries: CatalogSummary[];
  export default catalogSummaries;
}

declare module 'virtual:demo-data' {
  import type {DemoDataSource} from './gpu/data/provenance';
  const inventory: {
    sources: Record<string, DemoDataSource>;
    entries: Record<string, {sources: string[]; scenario?: string; caption?: string}>;
  };
  export default inventory;
}
