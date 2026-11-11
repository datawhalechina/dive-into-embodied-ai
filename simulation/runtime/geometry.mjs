export function extractGeometry(model) {
  const geometry = [];
  for (let id = 0; id < model.ngeom; id++) {
    const type = model.geom_type[id];
    // The generic viewer draws the infinite ground. Omit invisible default geoms.
    if (type === 0 || model.geom_rgba[id * 4 + 3] === 0) continue;
    const material = model.geom_matid[id];
    const color = material >= 0 ? model.mat_rgba.slice(material * 4, material * 4 + 4) : model.geom_rgba.slice(id * 4, id * 4 + 4);
    const item = {id, type, size: model.geom_size.slice(id * 3, id * 3 + 3), color};
    if (type === 7) {
      const mesh = model.geom_dataid[id];
      const va = 3 * model.mesh_vertadr[mesh], fa = 3 * model.mesh_faceadr[mesh];
      Object.assign(item, {mesh, vertices: model.mesh_vert.slice(va, va + 3 * model.mesh_vertnum[mesh]), faces: model.mesh_face.slice(fa, fa + 3 * model.mesh_facenum[mesh])});
    } else if (![2, 3, 4, 5, 6].includes(type)) {
      throw new Error(`Unsupported visible geometry type: ${type}`);
    }
    geometry.push(item);
  }
  return geometry;
}
