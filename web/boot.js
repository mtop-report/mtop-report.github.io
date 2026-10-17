/* Start the app after all views have registered; enable offline use where supported. */
(function () {
  "use strict";
  if (window.MTOPApp) window.MTOPApp.boot();
  try {
    if ("serviceWorker" in navigator && location.protocol === "https:" && document.querySelector('link[rel="manifest"]')) {
      navigator.serviceWorker.register("sw.js").catch(() => {});
    }
  } catch (e) { /* not available in this frame */ }
})();
