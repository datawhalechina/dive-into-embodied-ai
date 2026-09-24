import React from 'react';
import Head from '@docusaurus/Head';
import {Redirect} from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';

// Preserve bookmarked course-overview URLs in both languages.
export default function TutorialsRedirect(): React.JSX.Element {
  const destination = useBaseUrl('/docs/practices/intro');
  return (
    <>
      <Head>
        <meta name="robots" content="noindex" />
        <meta httpEquiv="refresh" content={`0;url=${destination}`} />
      </Head>
      <Redirect to={destination} />
    </>
  );
}
