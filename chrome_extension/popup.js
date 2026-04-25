// FocusLens Chrome Extension — popup.js
// Handles: server status check, port storage, dashboard launch

const DEFAULT_PORT = 8501;

// ── Load saved port ─────────────────────────────────────────────────────────
function getPort(callback) {
  chrome.storage.local.get(["focuslens_port"], (result) => {
    callback(result.focuslens_port || DEFAULT_PORT);
  });
}

function getDashboardUrl(port) {
  return `http://localhost:${port}`;
}

// ── Check if Streamlit server is running ─────────────────────────────────────
async function checkServerStatus(port) {
  const dot  = document.getElementById("status-dot");
  const text = document.getElementById("server-status");

  try {
    // Use a no-cors fetch to detect if the port is open
    // Streamlit returns HTML — we just need a response, not the body
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    const response = await fetch(getDashboardUrl(port), {
      method: "GET",
      mode: "no-cors",
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    // If fetch didn't throw, server is reachable
    dot.className = "status-dot dot-green";
    text.innerHTML = '<span class="status-dot dot-green"></span>Running ✓';
  } catch (err) {
    dot.className = "status-dot dot-red";
    text.innerHTML = '<span class="status-dot dot-red"></span>Not running';
  }
}

// ── Open dashboard in new tab ────────────────────────────────────────────────
document.getElementById("open-btn").addEventListener("click", () => {
  getPort((port) => {
    chrome.tabs.create({ url: getDashboardUrl(port) });
    window.close();
  });
});

// ── Save custom port ──────────────────────────────────────────────────────────
document.getElementById("save-port-btn").addEventListener("click", () => {
  const input = document.getElementById("port-input");
  const port = parseInt(input.value, 10);

  if (!port || port < 1024 || port > 65535) {
    input.style.borderColor = "#FF6B6B";
    return;
  }

  chrome.storage.local.set({ focuslens_port: port }, () => {
    document.getElementById("port-display").textContent = port;
    input.style.borderColor = "#00C9A7";
    input.value = "";
    checkServerStatus(port);
  });
});

// ── Toggle instructions ───────────────────────────────────────────────────────
document.getElementById("instructions-btn").addEventListener("click", () => {
  const div = document.getElementById("instructions");
  div.style.display = div.style.display === "none" ? "block" : "none";
});

// ── Init on popup open ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  getPort((port) => {
    document.getElementById("port-display").textContent = port;
    document.getElementById("port-input").placeholder = port;
    checkServerStatus(port);
  });
});
