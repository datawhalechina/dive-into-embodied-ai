import {useEffect, useRef, useState} from 'react';
import type {CSSProperties, MouseEventHandler} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {ChevronDown} from 'lucide-react';
import {useLocalPathname} from '@docusaurus/theme-common/internal';
import {getMegaMenuById, type MegaMenuConfig, type MegaMenuItem, type MegaMenuSubItem} from './data';

type NavbarMegaMenuProps = {
  menuId: string;
  mobile?: boolean;
  onClick?: MouseEventHandler<HTMLAnchorElement>;
};

function normalizePath(pathname: string): string {
  if (pathname.length > 1 && pathname.endsWith('/')) {
    return pathname.slice(0, -1);
  }
  return pathname;
}

function matchesPath(pathname: string, basePath: string): boolean {
  const current = normalizePath(pathname);
  const base = normalizePath(basePath);
  return current === base || current.startsWith(`${base}/`);
}

function isSubItemActive(item: MegaMenuSubItem, pathname: string): boolean {
  return matchesPath(pathname, item.activeBasePath ?? item.to);
}

function isItemActive(item: MegaMenuItem, pathname: string): boolean {
  if (item.children?.length) {
    return item.children.some((child) => isSubItemActive(child, pathname));
  }
  const basePath = item.activeBasePath ?? item.to;
  return basePath ? matchesPath(pathname, basePath) : false;
}

function isMenuActive(menu: MegaMenuConfig, pathname: string): boolean {
  return menu.activeBasePaths.some((basePath) => matchesPath(pathname, basePath));
}

function NavbarMegaMenuDesktop({menu}: {menu: MegaMenuConfig}) {
  const [open, setOpen] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  const [panelLeft, setPanelLeft] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pathname = useLocalPathname();
  const menuActive = isMenuActive(menu, pathname);

  const clearCloseTimeout = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  };

  const handleMouseEnter = () => {
    clearCloseTimeout();
    setOpen(true);
  };

  const handleMouseLeave = () => {
    clearCloseTimeout();
    timeoutRef.current = setTimeout(() => {
      timeoutRef.current = null;
      setOpen(false);
    }, 150);
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (containerRef.current?.contains(document.activeElement)) {
          triggerRef.current?.focus();
        }
        clearCloseTimeout();
        setOpen(false);
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      clearCloseTimeout();
    };
  }, []);

  useEffect(() => {
    const onPointerDown = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        clearCloseTimeout();
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
  }, []);

  useEffect(() => {
    if (!open) return;

    const updatePanelLeft = () => {
      if (!triggerRef.current) return;

      const rect = triggerRef.current.getBoundingClientRect();
      const viewportWidth = window.innerWidth;
      const panelWidth = Math.min(menu.panelWidth, viewportWidth - 24);
      const halfPanelWidth = panelWidth / 2;
      const viewportPadding = 12;
      const preferredCenter = rect.left + rect.width / 2;
      const minCenter = halfPanelWidth + viewportPadding;
      const maxCenter = viewportWidth - halfPanelWidth - viewportPadding;
      const clampedCenter = Math.max(minCenter, Math.min(maxCenter, preferredCenter));

      setPanelLeft(clampedCenter);
    };

    updatePanelLeft();
    window.addEventListener('resize', updatePanelLeft);
    window.addEventListener('scroll', updatePanelLeft, true);
    return () => {
      window.removeEventListener('resize', updatePanelLeft);
      window.removeEventListener('scroll', updatePanelLeft, true);
    };
  }, [open, menu.panelWidth]);

  return (
    <div
      ref={containerRef}
      className="navbar__item navbar-mega"
      style={
        {
          '--mega-width': `${menu.panelWidth}px`,
          '--mega-columns': String(menu.columns.length),
        } as CSSProperties
      }
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          clearCloseTimeout();
          setOpen(false);
        }
      }}
    >
      <button
        ref={triggerRef}
        type="button"
        className={clsx('navbar__link', 'navbar-mega__trigger', {
          'navbar__link--active': menuActive,
          'navbar-mega__trigger--open': open,
        })}
        aria-expanded={open}
        aria-controls={`navbar-panel-${menu.id}`}
        id={`navbar-trigger-${menu.id}`}
        onClick={(event) => {
          clearCloseTimeout();
          // Hover already opens the disclosure for a pointer; keyboard activation toggles it.
          setOpen((value) => event.detail === 0 ? !value : true);
        }}
      >
        <span>{menu.label}</span>
        <ChevronDown size={14} aria-hidden="true" />
      </button>

      {open && (
        <div
          className="navbar-mega__backdrop"
          aria-hidden="true"
          onClick={() => {
            clearCloseTimeout();
            setOpen(false);
          }}
        />
      )}

      <div
        id={`navbar-panel-${menu.id}`}
        role="region"
        aria-labelledby={`navbar-trigger-${menu.id}`}
        className={clsx('navbar-mega__panel', {'navbar-mega__panel--open': open})}
        style={panelLeft ? ({left: `${panelLeft}px`} as CSSProperties) : undefined}
      >
        <div className="navbar-mega__surface">
          <div className="navbar-mega__columns">
            {menu.columns.map((column) => (
              <section key={column.title} className="navbar-mega__column">
                <h3 className="navbar-mega__columnTitle">{column.title}</h3>
                <div className="navbar-mega__list">
                  {column.items.map((item, itemIndex) => {
                    if (item.children?.length) {
                      const groupKey = `${column.title}-${item.title}-${itemIndex}`;
                      const sublistId = `navbar-sublist-${menu.id}-${itemIndex}`;
                      const hasActiveChild = item.children.some((child) => isSubItemActive(child, pathname));
                      const isExpanded = expandedGroups[groupKey] ?? hasActiveChild;

                      return (
                        <div
                          key={groupKey}
                          className={clsx('navbar-mega__group', {
                            'navbar-mega__group--expanded': isExpanded,
                          })}
                        >
                          <button
                            type="button"
                            className={clsx('navbar-mega__item', 'navbar-mega__item--trigger', {
                              'navbar-mega__item--featured': item.featured,
                              'navbar-mega__item--active': hasActiveChild,
                              'navbar-mega__item--expanded': isExpanded,
                            })}
                            aria-expanded={isExpanded}
                            aria-controls={sublistId}
                            onClick={() => {
                              setExpandedGroups((prev) => ({
                                ...prev,
                                [groupKey]: !isExpanded,
                              }));
                            }}
                          >
                            <div className="navbar-mega__itemRow">
                              <div className="navbar-mega__itemBody">
                                <div className="navbar-mega__itemHeader">
                                  <div className="navbar-mega__itemTitle">{item.title}</div>
                                  <ChevronDown
                                    size={16}
                                    className="navbar-mega__itemChevron"
                                    aria-hidden="true"
                                  />
                                </div>
                                <p className="navbar-mega__itemDescription">{item.description}</p>
                                {!!item.keywords?.length && (
                                  <div className="navbar-mega__itemKeywords">
                                    {item.keywords.map((keyword) => (
                                      <span key={keyword} className="navbar-mega__keyword">
                                        {keyword}
                                      </span>
                                    ))}
                                  </div>
                                )}
                              </div>
                            </div>
                          </button>

                          {isExpanded && (
                            <div id={sublistId} className="navbar-mega__sublist">
                              {item.children.map((child) => (
                                <Link
                                  key={child.to}
                                  to={child.to}
                                  className={clsx('navbar-mega__item', 'navbar-mega__subitem', {
                                    'navbar-mega__item--active': isSubItemActive(child, pathname),
                                  })}
                                  onClick={() => {
                                    clearCloseTimeout();
                                    setOpen(false);
                                  }}
                                >
                                  <div className="navbar-mega__itemRow">
                                    <div className="navbar-mega__itemBody">
                                      <div className="navbar-mega__itemTitle">{child.title}</div>
                                      <p className="navbar-mega__itemDescription">{child.description}</p>
                                      {!!child.keywords?.length && (
                                        <div className="navbar-mega__itemKeywords">
                                          {child.keywords.map((keyword) => (
                                            <span key={keyword} className="navbar-mega__keyword">
                                              {keyword}
                                            </span>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                </Link>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    }

                    return (
                      <Link
                        key={item.to ?? item.title}
                        to={item.to}
                        className={clsx('navbar-mega__item', {
                          'navbar-mega__item--featured': item.featured,
                          'navbar-mega__item--active': isItemActive(item, pathname),
                        })}
                        onClick={() => {
                          clearCloseTimeout();
                          setOpen(false);
                        }}
                      >
                        <div className="navbar-mega__itemRow">
                          <div className="navbar-mega__itemBody">
                            <div className="navbar-mega__itemTitle">{item.title}</div>
                            <p className="navbar-mega__itemDescription">{item.description}</p>
                            {!!item.keywords?.length && (
                              <div className="navbar-mega__itemKeywords">
                                {item.keywords.map((keyword) => (
                                  <span key={keyword} className="navbar-mega__keyword">
                                    {keyword}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      </Link>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>

          <div className="navbar-mega__footer">
            <span className="navbar-mega__footerText">{menu.footer.text}</span>
            <Link
              className="navbar-mega__footerLink"
              to={menu.footer.to}
              onClick={() => {
                clearCloseTimeout();
                setOpen(false);
              }}
            >
              {menu.footer.ctaLabel}
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

function NavbarMegaMenuMobile({menu, onClick}: {menu: MegaMenuConfig; onClick?: MouseEventHandler<HTMLAnchorElement>}) {
  const pathname = useLocalPathname();
  const active = isMenuActive(menu, pathname);
  return (
    <Link
      to={menu.footer.to}
      onClick={onClick}
      className={clsx('menu__link', {'menu__link--active': active})}>
      {menu.label}
    </Link>
  );
}

export default function NavbarMegaMenu({menuId, mobile = false, onClick}: NavbarMegaMenuProps) {
  const menu = getMegaMenuById(menuId);

  if (!menu) {
    return null;
  }

  if (mobile) {
    return <NavbarMegaMenuMobile menu={menu} onClick={onClick} />;
  }

  return <NavbarMegaMenuDesktop menu={menu} />;
}
