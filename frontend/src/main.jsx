import { StrictMode, useState } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import Live from './Live.jsx'

function Root() {
  const params = new URLSearchParams(window.location.search);
  const mode = params.get("mode") || "auto";
  const backend = params.get("backend") || "http://localhost:8000";

  // Auto-detect: mobile → live, desktop → demo dashboard
  const isMobile = mode === "live" || (mode === "auto" && /Mobi|Android|iPhone/i.test(navigator.userAgent));

  const [view, setView] = useState(isMobile ? "live" : "demo");

  if (view === "live") {
    return <Live backendUrl={backend} />;
  }
  return <App />;
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Root />
  </StrictMode>,
)
