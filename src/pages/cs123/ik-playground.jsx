import React from 'react';
import Layout from '@theme/Layout';
import PlaygroundHeader from '@site/src/components/PlaygroundHeader';
import IkPlayground from '@site/src/components/docs/projects/cs123/inverse-kinematics/IkPlayground';

export default function IkPlaygroundPage() {
  return (
    <Layout
      title="IK Playground · 逆运动学交互"
      description="3-DoF 平面臂 IK 交互 playground。拖动目标点,求解器会反解关节角并把臂拉过去。"
      noFooter
    >
      <PlaygroundHeader currentKey="ik" />
      <div className="playground-shell">
        <IkPlayground />
      </div>
    </Layout>
  );
}
