import React, {useEffect} from 'react';
import {useHistory, useLocation} from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import OriginalLayout from '@theme-original/Layout';
import type {Props} from '@theme/Layout';

export default function Layout(props: Props) {
  const {siteConfig: {baseUrl}} = useDocusaurusContext();
  const history = useHistory();
  const location = useLocation();

  useEffect(() => {
    // The preview server strips /en/'s trailing slash. Restore the locale root
    // in the router so Docusaurus can generate valid alternate-language URLs.
    if (location.pathname === baseUrl.replace(/\/$/, '')) {
      history.replace({...location, pathname: baseUrl});
    }
  }, [baseUrl, history, location]);

  return <OriginalLayout {...props} />;
}
