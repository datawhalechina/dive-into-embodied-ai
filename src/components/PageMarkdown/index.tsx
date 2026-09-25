import React, {useEffect, useId, useRef, useState} from 'react';
import {useLocation} from '@docusaurus/router';
import {translate} from '@docusaurus/Translate';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {Check, ChevronDown, Copy, FileText, X} from 'lucide-react';
import {pageToMarkdown} from './markdown.mjs';
import styles from './styles.module.css';

// Remount on navigation so previews and copy feedback always belong to this page.
export default function PageMarkdown({className}: {className?: string}) {
  const {pathname} = useLocation();
  return <PageMarkdownActions key={pathname} className={className} />;
}

function PageMarkdownActions({className}: {className?: string}) {
  const {i18n} = useDocusaurusContext();
  const [status, setStatus] = useState<'idle' | 'copying' | 'copied' | 'error'>('idle');
  const [markdown, setMarkdown] = useState('');
  const [previewError, setPreviewError] = useState(false);
  const details = useRef<HTMLDetailsElement>(null);
  const summary = useRef<HTMLElement>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const copyButton = useRef<HTMLButtonElement>(null);
  const mounted = useRef(true);
  const titleId = useId();
  const hintId = useId();
  const copyLabel = translate({id: 'pageMarkdown.copy', message: '复制为 Markdown'});
  const viewLabel = translate({id: 'pageMarkdown.view', message: '查看 Markdown'});
  const copiedLabel = translate({id: 'pageMarkdown.copied', message: '已复制'});
  const copyingLabel = translate({id: 'pageMarkdown.copying', message: '正在复制…'});
  const copyError = translate({id: 'pageMarkdown.copyError', message: '无法自动复制，请在下方选中文本后复制。'});
  const readError = translate({id: 'pageMarkdown.readError', message: '无法读取页面内容，请刷新后重试。'});

  useEffect(() => {
    mounted.current = true;
    const closeOutside = (event: PointerEvent) => {
      if (!details.current?.contains(event.target as Node) && details.current) details.current.open = false;
    };
    document.addEventListener('pointerdown', closeOutside);
    return () => {
      mounted.current = false;
      document.removeEventListener('pointerdown', closeOutside);
    };
  }, []);

  useEffect(() => {
    if (status !== 'copied') return undefined;
    const timer = window.setTimeout(() => setStatus('idle'), 2500);
    return () => window.clearTimeout(timer);
  }, [status]);

  function closeMenu() {
    if (details.current) details.current.open = false;
  }

  function readPage() {
    const content = document.querySelector('[data-markdown-content]') ?? document.querySelector('main');
    if (!content) throw new Error('No page content');
    // Exclude the current scroll anchor; in-page links are expanded to full URLs.
    const url = new URL(window.location.href);
    url.hash = '';
    url.search = '';
    return pageToMarkdown(content, url.href, document.title);
  }

  function showPreview(text: string, error = false) {
    closeMenu();
    setMarkdown(text);
    setPreviewError(error);
    dialog.current?.showModal();
    // Wait for React to put the text into the textarea before selecting it.
    requestAnimationFrame(() => {
      if (!mounted.current) return;
      textarea.current?.focus();
      if (error) textarea.current?.select();
      else textarea.current?.setSelectionRange(0, 0);
      if (textarea.current) textarea.current.scrollTop = 0;
    });
  }

  async function copy(text?: string) {
    if (status === 'copying') return;
    closeMenu();
    let value: string;
    try { value = text ?? readPage(); } catch { setStatus('error'); return; }
    setStatus('copying');
    try {
      // Keep conversion synchronous to preserve the browser's user activation.
      await navigator.clipboard.writeText(value);
      if (mounted.current) {
        setStatus('copied');
        setPreviewError(false);
        if (!dialog.current?.open) copyButton.current?.focus({preventScroll: true});
      }
    } catch {
      if (mounted.current) {
        setStatus('idle');
        showPreview(value, true);
      }
    }
  }

  function view() {
    try { showPreview(readPage()); setStatus('idle'); } catch { setStatus('error'); }
  }

  const label = status === 'copied' ? copiedLabel : status === 'copying' ? copyingLabel : copyLabel;
  const Icon = status === 'copied' ? Check : Copy;
  return (
    <div className={[styles.root, className].filter(Boolean).join(' ')} data-markdown-exclude lang={i18n.currentLocale}>
      <div className={styles.splitButton}>
        <button ref={copyButton} type="button" className={styles.copyButton} onClick={() => copy()} disabled={status === 'copying'}>
          <Icon size={16} aria-hidden="true" /><span>{label}</span>
        </button>
        <details ref={details} className={styles.dropdown}
          onBlur={event => {
            if (!event.currentTarget.contains(event.relatedTarget as Node)) closeMenu();
          }}
          onKeyDown={event => {
            if (event.key === 'Escape') { closeMenu(); summary.current?.focus(); event.stopPropagation(); }
          }}>
          <summary ref={summary} className={styles.toggle} aria-label={translate({id: 'pageMarkdown.options', message: 'Markdown 选项'})}>
            <ChevronDown size={16} aria-hidden="true" />
          </summary>
          <div className={styles.menu}>
            <button type="button" onClick={() => copy()} disabled={status === 'copying'}><Copy size={16} aria-hidden="true" />{copyLabel}</button>
            <button type="button" onClick={view}><FileText size={16} aria-hidden="true" />{viewLabel}</button>
          </div>
        </details>
      </div>
      <span className={styles.srOnly} role="status">{status === 'copied' ? copiedLabel : status === 'copying' ? copyingLabel : ''}</span>
      {status === 'error' && <p className={styles.error} role="alert">{readError}</p>}
      <dialog ref={dialog} className={styles.dialog} aria-labelledby={titleId} aria-describedby={hintId}
        onClose={() => copyButton.current?.focus({preventScroll: true})}
        onClick={event => { if (event.target === dialog.current) dialog.current.close(); }}>
        <div className={styles.preview}>
          <div className={styles.previewHeader}>
            <h2 id={titleId}>{viewLabel}</h2>
            <button type="button" className={styles.closeButton} onClick={() => dialog.current?.close()} aria-label={translate({id: 'pageMarkdown.close', message: '关闭预览'})}><X size={20} aria-hidden="true" /></button>
          </div>
          <p id={hintId} className={styles.hint}>
            {previewError ? copyError : translate({id: 'pageMarkdown.hint', message: '页面正文已转为 Markdown。交互图形可通过文末链接在原网页查看。'})}
          </p>
          <textarea ref={textarea} className={styles.textarea} value={markdown} readOnly spellCheck={false} aria-label={translate({id: 'pageMarkdown.content', message: 'Markdown 正文'})} />
          <div className={styles.previewFooter}>
            <span role="status">{status === 'copied' ? copiedLabel : ''}</span>
            <button type="button" className={styles.previewCopy} onClick={() => copy(markdown)} disabled={status === 'copying'}><Icon size={16} aria-hidden="true" />{label}</button>
          </div>
        </div>
      </dialog>
    </div>
  );
}
