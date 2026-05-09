import React from 'react';
import Layout from '@theme/Layout';
import PlaygroundHeader from '@site/src/components/PlaygroundHeader';
import PdPlayground from '@site/src/components/docs/projects/cs123/pid-control/PdPlayground';

export default function PdPlaygroundPage() {
  return (
    <Layout
      title="PD Playground · PID 控制交互"
      description="无人机悬停 PID 交互 playground。拖动 Kp/Ki/Kd 滑块或点击画布改目标,实时看闭环如何追上目标。"
      noFooter
    >
      <PlaygroundHeader currentKey="pd" />
      <div className="playground-shell">
        <PdPlayground />
      </div>
    </Layout>
  );
}
