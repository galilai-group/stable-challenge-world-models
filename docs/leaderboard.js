"use strict";
const el = id => document.getElementById(id);
// Renders text as plain text, turning only [label](https://…) into links.
const linkify = (node, text) => {
  node.replaceChildren(); let last = 0;
  for (const m of text.matchAll(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g)) {
    const a = document.createElement("a"); a.href = m[2]; a.textContent = m[1]; a.target = "_blank"; a.rel = "noopener";
    node.append(text.slice(last, m.index), a); last = m.index + m[0].length;
  }
  node.append(text.slice(last));
};
(async () => {
  try {
    const [config, data] = await Promise.all(["challenge.json", "leaderboard.json"].map(async url => {
      const r = await fetch(url, {cache: "no-cache"}); if (!r.ok) throw new Error("Unavailable"); return r.json();
    }));
    document.title = config.title; el("title").textContent = config.title;
    linkify(el("description"), config.description);
    const url = config.submit_url ? new URL(config.submit_url) : new URL("/submit", config.service_url);
    if (url.protocol !== "https:" && url.hostname !== "localhost") throw new Error("Invalid service URL");
    url.searchParams.set("challenge", config.slug);
    el("submit").href = url.href; el("submit").hidden = false;
    el("details").textContent = config.accepting_submissions === false ? "Submissions are closed" : config.closes_at ? `Deadline: ${new Date(config.closes_at).toLocaleString()}` : "Open for submissions";
    const limits = [config.max_bytes && `Maximum file size: ${(config.max_bytes / 1024 ** 2).toLocaleString()} MiB`, config.daily_limit && `${config.daily_limit} upload attempts per day`].filter(Boolean);
    if (limits.length && config.accepting_submissions !== false) el("limits").textContent = limits.join(" · ");
    const anonymous = (data.anonymous ?? config.anonymous) !== false;
    el('identity-policy').textContent = anonymous ? 'Participants use stable IDs; email addresses and compiled model files remain private.' : 'Participant email addresses are displayed publicly. Compiled model files remain private.';
    el('group-filter').replaceChildren(new Option('All participants', ''));
    for (const group of (data.groups || [])) el('group-filter').append(new Option(group, group));
    el('group-filter').hidden = el('group-filter-label').hidden = !(data.groups || []).length;
    const primary = data.metric || config.metric;
    const metrics = [primary, ...new Set((data.metrics || data.rows.flatMap(r => Object.keys(r.metrics || {}))).filter(k => k !== primary))];
    el("columns").replaceChildren();
    for (const label of ["Rank", anonymous ? "Participant ID" : "Email", ...metrics.map(m => m === primary ? `${m} ${(data.direction || config.direction) === 'maximize' ? '↑' : '↓'}` : m), "Submitted", "Status", "Logs"]) {
      const th = document.createElement('th'); th.scope = 'col'; th.textContent = label; el('columns').append(th);
    }
    el("policy").textContent = `Latest accepted compiled model per participant · Failed evaluations below ranked results · Ranks are global · Evaluation ${data.evaluator_version}`;
    el('close-logs').onclick = () => el('logs-dialog').close();
    function render() {
      el("rows").replaceChildren();
      const query = el('search').value.toLowerCase(), group = el('group-filter').value;
      const rows = data.rows.filter(row => (row.participant.toLowerCase().includes(query) || (row.participant_id || '').toLowerCase().includes(query))
        && (!group || (row.groups || []).includes(group)));
      for (const row of rows) {
        const failed = row.status === 'failed';
        const tr = document.createElement("tr"); if (failed) tr.className = 'failed';
        const values = [failed ? '—' : row.rank, row.participant,
          ...metrics.map(m => failed ? '—' : (row.metrics?.[m] ?? (m === primary ? row.score : null) ?? '—')),
          new Date(row.submitted_at).toLocaleString(), failed ? 'Failed' : 'Succeeded'];
        for (const value of values) {
          const td = document.createElement("td"); td.textContent = String(value); tr.appendChild(td);
        }
        const logs = document.createElement('td');
        if (failed) {
          const button = document.createElement('button'); button.type = 'button'; button.className = 'log-button'; button.textContent = 'View logs';
          button.onclick = () => {
            el('log-submission').textContent = `Participant ${row.participant} · Submission ${row.submission_id || 'unavailable'}`;
            el('log-content').textContent = row.logs || 'No public diagnostics available. Contact the organizers.';
            el('logs-dialog').showModal();
          };
          logs.append(button);
        } else logs.textContent = '—';
        tr.append(logs); el("rows").appendChild(tr);
      }
      el("status").textContent = rows.length ? "" : (data.rows.length ? "No matching participants." : "No evaluated submissions yet.");
    }
    el("search").addEventListener("input", render); el("group-filter").addEventListener("change", render); render();
  } catch { el("status").textContent = "Scores are temporarily unavailable. Please reload shortly."; }
})();
