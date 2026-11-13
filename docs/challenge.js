"use strict";
// Renders the repository's CHALLENGE.md, fetched live so edits appear without republishing the site.
(async () => {
  const el = id => document.getElementById(id);
  try {
    const r = await fetch("challenge.json", {cache: "no-cache"}); if (!r.ok) return;
    const config = await r.json();
    // Pages URLs are <owner>.github.io/<repo>/; older sites lack the repository field.
    const [, repoPath] = location.pathname.split("/");
    const repo = config.repository || (location.hostname.endsWith(".github.io") && repoPath
      ? `${location.hostname.split(".")[0]}/${repoPath}` : null);
    if (!repo || !window.marked || !window.DOMPurify) return;
    const md = await fetch(`https://raw.githubusercontent.com/${repo}/HEAD/CHALLENGE.md`, {cache: "no-cache"});
    if (!md.ok) return;
    const content = el("challenge-content");
    content.innerHTML = DOMPurify.sanitize(marked.parse(await md.text()));
    // Relative links point into the repository, not the Pages site.
    for (const [selector, attr, base] of [["a[href]", "href", `https://github.com/${repo}/blob/HEAD/`], ["img[src]", "src", `https://raw.githubusercontent.com/${repo}/HEAD/`]]) {
      for (const node of content.querySelectorAll(selector)) {
        const value = node.getAttribute(attr);
        if (value.startsWith("#")) continue;
        if (!/^[a-z]+:/i.test(value)) node.setAttribute(attr, new URL(value.replace(/^\.?\//, ""), base).href);
        if (node.tagName === "A") { node.target = "_blank"; node.rel = "noopener"; }
      }
    }
    const used = new Set(), links = [];
    for (const h of content.querySelectorAll("h1, h2, h3")) {
      let id = h.textContent.toLowerCase().replace(/[^\w]+/g, "-").replace(/^-|-$/g, "") || "section";
      while (used.has(id) || document.getElementById(id)) id += "-";
      used.add(id); h.id = id;
      if (h.tagName !== "H2") continue;
      const a = document.createElement("a"); a.href = `#${id}`; a.textContent = h.textContent; links.push([a, h]);
    }
    el("toc").replaceChildren(...links.map(([a]) => a));
    el("challenge-source").href = `https://github.com/${repo}/blob/HEAD/CHALLENGE.md`;
    el("challenge").hidden = el("page-nav").hidden = false;
    // Highlight the section being read.
    const observer = new IntersectionObserver(() => {
      const current = links.filter(([, h]) => h.getBoundingClientRect().top < innerHeight * 0.3).pop();
      for (const [a] of links) a.classList.toggle("active", a === current?.[0]);
    }, {rootMargin: "0px 0px -70% 0px"});
    for (const [, h] of links) observer.observe(h);
    if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  } catch { /* The leaderboard remains usable without the guide. */ }
})();
