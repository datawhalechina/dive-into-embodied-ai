import {useEffect, useRef, useState} from 'react';
import type {CSSProperties, MouseEventHandler} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {ChevronDown} from 'lucide-react';
import {useLocalPathname} from '@docusaurus/theme-common/internal';
import {getMegaMenuById, type MegaMenuConfig, type MegaMenuItem} from './data';

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

function getActiveItemTo(menu: MegaMenuConfig, pathname: string): string | null {
  const allItems: MegaMenuItem[] = menu.columns.flatMap((column) => column.items);
  const exactMatch = allItems.find((item) => matchesPath(pathname, item.to));
  if (exactMatch) {
    return exactMatch.to;
  }
  const baseMatch = allItems.find((item) => {
    const basePaths = item.activeBasePaths ?? (item.activeBasePath ? [item.activeBasePath] : []);
    return basePaths.some((basePath) => matchesPath(pathname, basePath));
  });
  return baseMatch?.to ?? null;
}

function isMenuActive(menu: MegaMenuConfig, pathname: string): boolean {
  return menu.activeBasePaths.some((basePath) => matchesPath(pathname, basePath));
}

const activeMenuListeners = new Set<(menuId: string | null) => void>();

function notifyMenuOpened(menuId: string) {
  activeMenuListeners.forEach((listener) => listener(menuId));
}

function NavbarMegaMenuDesktop({menu}: {menu: MegaMenuConfig}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLAnchorElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const focusFirstOnOpenRef = useRef(false);
  const lastPointerTypeRef = useRef<string>('mouse');
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pathname = useLocalPathname();
  const menuActive = isMenuActive(menu, pathname);
  const activeItemTo = getActiveItemTo(menu, pathname);

  const clearCloseTimeout = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  };

  const openMenu = () => {
    clearCloseTimeout();
    notifyMenuOpened(menu.id);
    setOpen(true);
  };

  const closeMenu = () => {
    clearCloseTimeout();
    setOpen(false);
  };

  const handleMouseEnter = () => {
    openMenu();
  };

  const handleMouseLeave = () => {
    clearCloseTimeout();
    timeoutRef.current = setTimeout(() => {
      timeoutRef.current = null;
      setOpen(false);
    }, 140);
  };

  useEffect(() => {
    const listener = (openedId: string | null) => {
      if (openedId !== menu.id) {
        clearCloseTimeout();
        setOpen(false);
      }
    };
    activeMenuListeners.add(listener);
    return () => {
      activeMenuListeners.delete(listener);
    };
  }, [menu.id]);

  useEffect(() => {
    closeMenu();
  }, [pathname]);

  useEffect(() => {
    if (open && focusFirstOnOpenRef.current) {
      focusFirstOnOpenRef.current = false;
      panelRef.current?.querySelector<HTMLAnchorElement>('a')?.focus();
    }
  }, [open]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (containerRef.current?.contains(document.activeElement)) {
          triggerRef.current?.focus();
        }
        closeMenu();
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      clearCloseTimeout();
    };
  }, []);

  useEffect(() => {
    const onPointerDown = (event: PointerEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        closeMenu();
      }
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, []);

  return (
    <div
      ref={containerRef}
      className="navbar__item navbar-mega"
      style={{'--mega-columns': String(menu.columns.length)} as CSSProperties}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          closeMenu();
        }
      }}
    >
      <div
        className="navbar-mega__triggerWrap"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <Link
          ref={triggerRef}
          to={menu.to}
          className={clsx('navbar__link', 'navbar-mega__trigger', {
            'navbar__link--active': menuActive,
            'navbar-mega__trigger--open': open,
          })}
          aria-expanded={open}
          aria-controls={`navbar-panel-${menu.id}`}
          id={`navbar-trigger-${menu.id}`}
          onPointerDown={(event) => {
            lastPointerTypeRef.current = event.pointerType;
          }}
          onKeyDown={(event) => {
            if (event.key === 'ArrowDown') {
              event.preventDefault();
              if (open) {
                panelRef.current?.querySelector<HTMLAnchorElement>('a')?.focus();
              } else {
                focusFirstOnOpenRef.current = true;
                openMenu();
              }
            }
          }}
          onClick={(event) => {
            if (lastPointerTypeRef.current === 'touch' && !open) {
              event.preventDefault();
              openMenu();
              return;
            }
            closeMenu();
          }}
        >
          <span>{menu.label}</span>
          <ChevronDown size={13} aria-hidden="true" />
        </Link>
      </div>

      {open && (
        <div
          className="navbar-mega__backdrop"
          aria-hidden="true"
          onClick={closeMenu}
        />
      )}

      <div
        ref={panelRef}
        id={`navbar-panel-${menu.id}`}
        role="region"
        aria-labelledby={`navbar-trigger-${menu.id}`}
        className={clsx('navbar-mega__panel', {'navbar-mega__panel--open': open})}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <div className="navbar-mega__surface">
          <div className="navbar-mega__columns">
            {menu.columns.map((column) => (
              <section key={column.title} className="navbar-mega__column">
                <h3 className="navbar-mega__columnTitle">{column.title}</h3>
                <ul className="navbar-mega__list">
                  {column.items.map((item) => {
                    const active = activeItemTo === item.to;
                    return (
                      <li key={item.to}>
                        <Link
                          to={item.to}
                          aria-current={active ? 'page' : undefined}
                          className={clsx('navbar-mega__item', {
                            'navbar-mega__item--active': active,
                          })}
                          onClick={closeMenu}
                        >
                          {item.title}
                        </Link>
                      </li>
                    );
                  })}
                </ul>
              </section>
            ))}
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
      to={menu.to}
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
