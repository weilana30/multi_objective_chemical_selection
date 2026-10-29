(function () {
  let activeViewer;
  let libraryPromise;
  let generation = 0;

  function disposeViewer() {
    generation += 1;
    if (activeViewer) {
      try {
        activeViewer.viewer.clear();
        activeViewer.viewer.render();
      } catch (_) {
        // The modal may already have detached the renderer's canvas.
      }
      activeViewer = null;
    }
  }

  function loadLibrary() {
    if (window.$3Dmol) return Promise.resolve();
    if (!libraryPromise) {
      libraryPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = "/vendor/3Dmol-min.js";
        script.async = true;
        script.onload = resolve;
        script.onerror = () => {
          libraryPromise = null;
          reject(new Error("The local 3D viewer library did not load."));
        };
        document.head.appendChild(script);
      });
    }
    return libraryPromise;
  }

  function residuesNearLigand(receptorModel, ligandModel) {
    const receptorAtoms = receptorModel.selectedAtoms({});
    const ligandAtoms = ligandModel.selectedAtoms({});
    const nearby = new Map();
    const cutoffSquared = 25;
    receptorAtoms.forEach((atom) => {
      const close = ligandAtoms.some((ligand) => {
        const dx = atom.x - ligand.x;
        const dy = atom.y - ligand.y;
        const dz = atom.z - ligand.z;
        return dx * dx + dy * dy + dz * dz <= cutoffSquared;
      });
      if (close) nearby.set(`${atom.chain || ""}:${atom.resi}`, { chain: atom.chain || "", resi: atom.resi });
    });
    return Array.from(nearby.values());
  }

  function setReceptorVisible(state, visible) {
    state.receptorModel.setStyle({}, visible
      ? { cartoon: { color: "#aab5c0", opacity: 0.48 } } : {});
    if (visible) {
      state.pocketResidues.forEach((residue) => {
        state.receptorModel.setStyle(residue, { stick: { radius: 0.13, color: "#d8a45d" } }, true);
      });
    }
    state.visible = visible;
    state.viewer.render();
  }

  async function initialize() {
    const element = document.getElementById("molecule-3d-viewer");
    if (!element || element.dataset.loaded === "true") return;
    if (activeViewer && activeViewer.element !== element) disposeViewer();
    element.dataset.loaded = "true";
    const myGeneration = ++generation;
    const status = document.getElementById("molecule-3d-status");
    try {
      await loadLibrary();
      if (myGeneration !== generation || !element.isConnected ||
          document.getElementById("molecule-3d-viewer") !== element) return;
      const [poseResponse, receptorResponse] = await Promise.all([
        fetch(element.dataset.poseUrl), fetch(element.dataset.receptorUrl),
      ]);
      if (myGeneration !== generation || !element.isConnected ||
          document.getElementById("molecule-3d-viewer") !== element) return;
      if (!poseResponse.ok || !receptorResponse.ok) throw new Error("Saved pose or receptor file is unavailable.");
      const [pose, receptor] = await Promise.all([poseResponse.text(), receptorResponse.text()]);
      if (myGeneration !== generation || !element.isConnected ||
          document.getElementById("molecule-3d-viewer") !== element) return;
      const viewer = window.$3Dmol.createViewer(element, { backgroundColor: "white", antialias: true });
      const receptorModel = viewer.addModel(receptor, "pdb");
      receptorModel.setStyle({}, { cartoon: { color: "#aab5c0", opacity: 0.48 } });
      const ligandModel = viewer.addModel(pose, "sdf");
      ligandModel.setStyle({}, { stick: { radius: 0.18, colorscheme: "Jmol" } });
      const pocketResidues = residuesNearLigand(receptorModel, ligandModel);
      pocketResidues.forEach((residue) => {
        receptorModel.setStyle(residue, { stick: { radius: 0.13, color: "#d8a45d" } }, true);
      });
      viewer.zoomTo({ model: 1 });
      viewer.render();
      activeViewer = { viewer, receptorModel, ligandModel, pocketResidues, element, visible: true };
      if (status) status.remove();
    } catch (error) {
      if (myGeneration !== generation || !element.isConnected) return;
      element.dataset.loaded = "false";
      if (status) status.textContent = `${error.message} Showing the 2D structure above.`;
    }
  }

  function activate() {
    const element = document.getElementById("molecule-3d-viewer");
    if (!element) {
      if (activeViewer) disposeViewer();
      return;
    }
    if (activeViewer && activeViewer.element !== element) disposeViewer();
    if (!element || element.dataset.loaded) return;
    initialize();
  }

  document.addEventListener("click", (event) => {
    if (event.target.closest("#viewer-reset") && activeViewer) {
      activeViewer.viewer.zoomTo({ model: 1 });
      activeViewer.viewer.render();
    }
    if (event.target.closest("#viewer-toggle-receptor") && activeViewer) {
      setReceptorVisible(activeViewer, !activeViewer.visible);
      const button = event.target.closest("#viewer-toggle-receptor");
      button.textContent = activeViewer.visible ? "Hide receptor" : "Show receptor";
    }
  });

  new MutationObserver(activate).observe(document.body, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", activate);
})();
