import React from 'react';
import Layout from '@theme/Layout';
import PlaygroundHeader from '@site/src/components/PlaygroundHeader';
import FkPlayground from '@site/src/components/docs/projects/cs123/forward-kinematics/FkPlayground';

export default function FkPlaygroundPage() {
  return (
    <Layout
      title="FK Playground · 正运动学交互"
      description="3-DoF 平面臂 FK 交互 playground。拖动关节角,观察末端位置随关节角变化。"
      noFooter
    >
      <PlaygroundHeader currentKey="fk" />
      <div className="playground-shell">
        <FkPlayground />
      </div>
    </Layout>
  );
}
