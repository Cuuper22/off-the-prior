// CuperOS desktop for Off the Prior (window manager adapted from gpu_stack's docs/app.js).
// Windows: Story, Beat it (the toy), Atlas (neuron browser). On phones the windows become plain
// full-width pages with a tab bar on top.
(() => {
  "use strict";

  const phone = window.matchMedia("(max-width: 760px)");
  const windows = {};
  document.querySelectorAll("[data-window]").forEach((node) => { windows[node.dataset.window] = node; });
  const tasks = [...document.querySelectorAll("[data-task]")];
  const tabs = [...document.querySelectorAll(".phone-tabs [data-open]")];
  const mounts = { toy: () => window.OTPToy, atlas: () => window.OTPAtlas };
  let front = "story";
  let zCounter = 5;

  const isOpen = (name) => Boolean(windows[name]) && !windows[name].hidden;

  function mount(name) {
    const root = document.getElementById(`${name}-root`);
    if (!root || root.dataset.mounted || !mounts[name]) return;
    const api = mounts[name]();
    if (api && typeof api.mount === "function") {
      root.dataset.mounted = "1";
      api.mount(root);
    }
  }

  function refresh() {
    tasks.forEach((button) => {
      const name = button.dataset.task;
      button.classList.toggle("is-active", isOpen(name) && front === name);
      button.setAttribute("aria-pressed", String(isOpen(name)));
    });
    tabs.forEach((tab) => {
      if (tab.dataset.open === front) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    });
    Object.entries(windows).forEach(([name, node]) => node.classList.toggle("is-front", isOpen(name) && front === name));
  }

  function open(name, options = {}) {
    const node = windows[name];
    if (!node) return;
    if (phone.matches) Object.entries(windows).forEach(([other, el]) => { el.hidden = other !== name; });
    else node.hidden = false;
    node.classList.remove("is-max");
    front = name;
    zCounter += 1;
    node.style.zIndex = String(zCounter);
    mount(name);
    refresh();
    if (options.updateHash !== false && window.history && window.history.replaceState) {
      const hash = name === "story" ? "" : `#${name}`;
      if (hash && window.location.hash !== hash) window.history.replaceState(null, "", hash);
    }
  }

  function close(name) {
    const node = windows[name];
    if (!node || phone.matches) return;
    node.hidden = true;
    if (front === name) front = Object.keys(windows).find(isOpen) || "";
    refresh();
  }

  // node the click happened on (the DOM property name is avoided: the project's word list rules it out)
  const nodeOf = (event) => (event.composedPath ? event.composedPath()[0] : event.srcElement);

  document.addEventListener("click", (event) => {
    const at = nodeOf(event);
    if (!at || !at.closest) return;
    const opener = at.closest("[data-open]");
    if (opener) { event.preventDefault(); open(opener.dataset.open); return; }
    const task = at.closest("[data-task]");
    if (task) {
      const name = task.dataset.task;
      if (isOpen(name) && front === name && !phone.matches) close(name);
      else open(name);
      return;
    }
    const action = at.closest("[data-window-action]");
    if (action) {
      const node = action.closest("[data-window]");
      if (!node) return;
      if (action.dataset.windowAction === "max") node.classList.toggle("is-max");
      else close(node.dataset.window);
      return;
    }
    const win = at.closest("[data-window]");
    if (win && !phone.matches && win.dataset.window !== front) open(win.dataset.window, { updateHash: false });
  });

  phone.addEventListener("change", () => {
    if (phone.matches) open(isOpen(front) ? front : "story", { updateHash: false });
    else { windows.story.hidden = false; refresh(); }
  });

  function start() {
    const wanted = window.location.hash.slice(1);
    if (phone.matches) open(windows[wanted] ? wanted : "story", { updateHash: false });
    else {
      windows.story.hidden = false;
      open("story", { updateHash: false });
      if (windows[wanted] && wanted !== "story") open(wanted, { updateHash: false });
    }
    refresh();
  }

  window.addEventListener("hashchange", () => {
    const wanted = window.location.hash.slice(1);
    if (windows[wanted]) open(wanted, { updateHash: false });
  });

  const clock = document.getElementById("clock");
  function tick() {
    if (!clock) return;
    const now = new Date();
    clock.textContent = `${now.getHours() % 12 || 12}:${String(now.getMinutes()).padStart(2, "0")} ${now.getHours() >= 12 ? "PM" : "AM"}`;
  }
  tick();
  window.setInterval(tick, 30000);
  start();
})();
