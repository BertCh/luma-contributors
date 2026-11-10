// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Section} from '../../../ui/section';
import {Figure, TopicShell} from '../../page-kit';
import {ThisPageDemo} from '../../this-page-demo';
import type {TopicPageProps} from '../../topic-types';

export function Page({topic, models}: TopicPageProps) {
  return (
    <TopicShell topic={topic}>
      <Section title="Measured now">
        <Figure caption="Every number is read from the canvas on this page, in this tab. Reload to watch the steps light in order.">
          <ThisPageDemo />
        </Figure>
      </Section>
    </TopicShell>
  );
}
