import React from 'react';
import Layout from '@theme/Layout';
import SimulationLab from '@site/src/components/SimulationLab';
import {messages} from '@site/src/components/SimulationLab/messages';

export default function SimulationPage() {
  const text = messages();
  return <Layout title={text.title} description={text.subtitle} noFooter><SimulationLab /></Layout>;
}
