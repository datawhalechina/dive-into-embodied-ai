import React from 'react';
import Layout from '@theme/Layout';
import TranslationNotice from '@site/src/components/TranslationNotice';
import SensorCalibrationPlayground from '@site/src/components/docs/foundations/perception/SensorCalibrationPlayground';

export default function SensorCalibrationPlaygroundPage() {
  return (
    <Layout
      title="Sensor Calibration Playground · 在线标定小游戏"
      description="把传感器外参误差映射成机械臂、移动机器人和四足机器人任务误差的在线标定交互小游戏。"
      noFooter
    >
      <TranslationNotice />
      <div lang="zh-Hans">
        <SensorCalibrationPlayground />
      </div>
    </Layout>
  );
}
