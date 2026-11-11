const path = require('node:path');
const {generateSimulationAssets} = require('./simulation-assets.cjs');
module.exports = context => ({
  name: 'simulation-lab-assets',
  getPathsToWatch: () => [path.join(context.siteDir, 'simulation/**/*')],
  loadContent: () => generateSimulationAssets(context.siteDir),
});
