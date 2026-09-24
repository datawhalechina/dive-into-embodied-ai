import React, {type ReactNode} from 'react';
import DocItemLayout from '@theme-original/DocItem/Layout';
import type {Props} from '@theme/DocItem/Layout';

export default function ReadingLayout(props: Props): ReactNode {
  return (
    <div className="site-doc-layout">
      <DocItemLayout {...props} />
    </div>
  );
}
