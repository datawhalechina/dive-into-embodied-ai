import React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import { Play } from 'lucide-react';

function PlaceholderCard() {
  return (
    <div className="tw-my-6 tw-rounded-2xl tw-border tw-border-slate-700/60 tw-bg-slate-900 tw-p-4 tw-text-slate-300">
      <div className="tw-flex tw-items-center tw-justify-between tw-mb-3">
        <div className="tw-flex tw-items-center tw-gap-2 tw-text-sm tw-font-medium">
          <span>🕹 在线试试 · CartPole-v1</span>
          <span className="tw-text-xs tw-text-slate-400">(浏览器里跑)</span>
        </div>
      </div>
      <button
        type="button"
        disabled
        className="tw-inline-flex tw-items-center tw-gap-2 tw-rounded-lg tw-bg-slate-700/70 tw-px-4 tw-py-2 tw-text-sm tw-text-slate-300 tw-opacity-70 tw-cursor-not-allowed"
      >
        <Play size={16} />
        正在初始化交互组件…
      </button>
    </div>
  );
}

export default function CartPolePyodide(props) {
  return (
    <BrowserOnly fallback={<PlaceholderCard />}>
      {() => {
        const CartPolePyodideClient =
          require('./CartPolePyodideClient').default;
        return <CartPolePyodideClient {...props} />;
      }}
    </BrowserOnly>
  );
}
