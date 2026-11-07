// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute, notFound} from '@tanstack/react-router';
import {getFacet} from '../../catalog';
import {FacetPage} from '../../components/catalog/facet-page';
import {SITE_NAME} from '../__root';

export const Route = createFileRoute('/explore/use/$id')({
  loader: ({params}) => {
    const facet = getFacet('use', params.id);
    if (!facet) {
      throw notFound();
    }
    return {label: facet.label, description: facet.description};
  },
  head: ({loaderData}) => ({
    meta: [
      {title: `${loaderData?.label ?? 'Use'} | ${SITE_NAME}`},
      {name: 'description', content: loaderData?.description ?? ''}
    ]
  }),
  component: Page
});

function Page() {
  const {id} = Route.useParams();
  return <FacetPage kind="use" id={id} />;
}
