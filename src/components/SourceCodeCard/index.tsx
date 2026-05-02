import {type ReactNode, useState} from 'react';
import CodeBlock from '@theme/CodeBlock';

interface SourceCodeCardProps {
  title: string;
  language: string;
  children: string;
  description?: string;
  filename?: string;
  defaultOpen?: boolean;
}

export default function SourceCodeCard({
  title,
  language,
  children,
  description,
  filename,
  defaultOpen = false,
}: SourceCodeCardProps): ReactNode {
  const [open, setOpen] = useState(defaultOpen);
  const lineCount = children.trim().split('\n').length;

  return (
    <div className="tw-rounded-xl tw-border tw-border-gray-200 dark:tw-border-gray-800 tw-overflow-hidden tw-my-6 tw-transition-all tw-duration-300 hover:tw-border-sky-500/40 hover:tw-shadow-md hover:tw-shadow-sky-500/5">
      {/* Card header */}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="tw-w-full tw-flex tw-items-center tw-justify-between tw-px-5 tw-py-4 tw-bg-gray-50 dark:tw-bg-gray-900/50 tw-border-b tw-border-gray-200 dark:tw-border-gray-800 tw-cursor-pointer hover:tw-bg-gray-100 dark:hover:tw-bg-gray-900/80 tw-transition-colors tw-duration-200"
      >
        <div className="tw-flex tw-items-center tw-gap-3">
          {/* File icon */}
          <div className="tw-flex tw-items-center tw-justify-center tw-w-9 tw-h-9 tw-rounded-lg tw-bg-sky-500/10">
            <svg className="tw-w-5 tw-h-5 tw-text-sky-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
            </svg>
          </div>
          <div className="tw-text-left">
            <div className="tw-font-semibold tw-text-sm">{title}</div>
            {description && (
              <div className="tw-text-xs tw-opacity-50 tw-mt-0.5">{description}</div>
            )}
          </div>
        </div>

        <div className="tw-flex tw-items-center tw-gap-3">
          {/* Meta badges */}
          {filename && (
            <span className="tw-hidden sm:tw-inline-flex tw-items-center tw-rounded-md tw-bg-gray-200/60 dark:tw-bg-gray-800 tw-px-2 tw-py-0.5 tw-text-xs tw-font-mono tw-opacity-60">
              {filename}
            </span>
          )}
          <span className="tw-hidden sm:tw-inline-flex tw-items-center tw-rounded-md tw-bg-gray-200/60 dark:tw-bg-gray-800 tw-px-2 tw-py-0.5 tw-text-xs tw-opacity-60">
            {lineCount} lines
          </span>

          {/* Chevron */}
          <svg
            className={`tw-w-5 tw-h-5 tw-opacity-40 tw-transition-transform tw-duration-300 ${open ? 'tw-rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Collapsible code block */}
      {open && (
        <div className="source-code-card-body">
          <CodeBlock language={language} showLineNumbers title={filename}>
            {children.trim()}
          </CodeBlock>
        </div>
      )}
    </div>
  );
}
