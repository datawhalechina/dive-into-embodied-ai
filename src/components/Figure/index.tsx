import React from 'react';
import useBrokenLinks from '@docusaurus/useBrokenLinks';

interface FigureProps {
  children: React.ReactNode;
  caption: string;
  width?: number | string;
  id?: string;
}

// Put a standard Markdown image between blank lines inside Figure. GitHub can
// render that image even though it strips this MDX wrapper from its preview.
export default function Figure({children, caption, width = 560, id}: FigureProps) {
  useBrokenLinks().collectAnchor(id);

  const imageStyle =
    typeof width === 'number'
      ? {width: '100%', maxWidth: `${width}px`}
      : {width: '100%', maxWidth: width};

  return (
    <figure className="doc-figure" id={id}>
      <div className="doc-figure-content" style={imageStyle}>
        {children}
      </div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
