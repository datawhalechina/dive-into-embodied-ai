/**
 * Custom rehype-mathjax plugin with two-pass rendering.
 *
 * The default rehype-mathjax processes math nodes in document order via
 * MathJax's `document.convert()`. Forward references (\eqref before \label)
 * render as "(?)". This plugin does two passes:
 *   Pass 1 – render display math (which carry \label definitions)
 *   Pass 2 – render inline math (which may carry \eqref references)
 * Because all labels are registered in pass 1, forward \eqref in inline math
 * resolves correctly.
 */

import {toText} from 'hast-util-to-text';
import {SKIP, visitParents} from 'unist-util-visit-parents';
import {liteAdaptor as liteAdapter} from 'mathjax-full/js/adaptors/liteAdaptor.js';
import {RegisterHTMLHandler as registerHtmlHandler} from 'mathjax-full/js/handlers/html.js';
import {AllPackages as allPackages} from 'mathjax-full/js/input/tex/AllPackages.js';
import {TeX as Tex} from 'mathjax-full/js/input/tex.js';
import {SVG as Svg} from 'mathjax-full/js/output/svg.js';
import {mathjax} from 'mathjax-full/js/mathjax.js';
import {h} from 'hastscript';

const emptyClasses = [];

function fromLiteElement(liteElement) {
  const children = [];
  for (const node of liteElement.children) {
    children.push(
      'value' in node
        ? {type: 'text', value: node.value}
        : fromLiteElement(node),
    );
  }
  return h(liteElement.kind, liteElement.attributes, children);
}

function hoistEquationAnchorIds(node, ids = []) {
  if (!node || node.type !== 'element') {
    return ids;
  }

  const id = node.properties?.id;
  if (typeof id === 'string' && id.startsWith('mjx-eqn:')) {
    ids.push(id);
    delete node.properties.id;
  }

  if (Array.isArray(node.children)) {
    for (const child of node.children) {
      hoistEquationAnchorIds(child, ids);
    }
  }

  return ids;
}

export default function rehypeMathjaxTwoPass(options = {}) {
  return function (tree) {
    // Collect all math nodes with metadata
    const mathNodes = [];

    visitParents(tree, 'element', function (element, parents) {
      const classes = Array.isArray(element.properties.className)
        ? element.properties.className
        : emptyClasses;
      const languageMath = classes.includes('language-math');
      const mathDisplay = classes.includes('math-display');
      const mathInline = classes.includes('math-inline');
      let display = mathDisplay;

      if (!languageMath && !mathDisplay && !mathInline) return;

      let parent = parents[parents.length - 1];
      let scope = element;

      if (
        element.tagName === 'code' &&
        languageMath &&
        parent &&
        parent.type === 'element' &&
        parent.tagName === 'pre'
      ) {
        scope = parent;
        parent = parents[parents.length - 2];
        display = true;
      }

      if (!parent) return;

      const text = toText(scope, {whitespace: 'pre'});
      mathNodes.push({element, scope, parent, text, display});

      return SKIP;
    });

    if (mathNodes.length === 0) return;

    // Set up MathJax
    const adapter = liteAdapter();
    const handler = registerHtmlHandler(adapter);
    const input = new Tex({packages: allPackages, ...options.tex});
    const output = new Svg(options.svg || {});
    const document = mathjax.document('', {InputJax: input, OutputJax: output});

    function renderNode(node) {
      const liteElement = document.convert(node.text, {display: node.display});
      const rendered = fromLiteElement(liteElement);
      // Preserve the source for the page's Markdown export, alongside the SVG.
      rendered.properties['data-markdown-math'] = node.text;
      const result = [];

      if (node.display) {
        const anchorIds = hoistEquationAnchorIds(rendered);
        for (const anchorId of anchorIds) {
          result.push(
            h('a', {
              id: anchorId,
              'aria-hidden': 'true',
              tabindex: '-1',
            }),
          );
        }
      }

      result.push(rendered);
      const index = node.parent.children.indexOf(node.scope);
      if (index !== -1) {
        node.parent.children.splice(index, 1, ...result);
      }
    }

    // Pass 1: render all display math first (registers \label)
    const displayNodes = mathNodes.filter((n) => n.display);
    const inlineNodes = mathNodes.filter((n) => !n.display);

    for (const node of displayNodes) {
      renderNode(node);
    }

    // Pass 2: render inline math (\eqref can now resolve)
    for (const node of inlineNodes) {
      renderNode(node);
    }

    // Inject stylesheet into <head>
    let headElement = tree;
    visitParents(tree, 'element', function (element) {
      if (element.tagName === 'head') {
        headElement = element;
        return SKIP;
      }
    });
    const styleNode = fromLiteElement(output.styleSheet(document));
    styleNode.properties.id = undefined;
    headElement.children.push(styleNode);

    mathjax.handlers.unregister(handler);
  };
}
