/* Where the API lives.
 *
 * The frontend and the API are deployed separately, so the base URL cannot be
 * baked in at build time. Resolution order, most specific first:
 *
 *   1. ?api=https://...        - override per visit, useful for testing a
 *                                staging API from the production frontend
 *   2. window.OILSPILL_API_BASE - set by the host (a <script> tag, or a
 *                                platform's env-var injection)
 *   3. same origin             - the local single-command demo, where the API
 *                                serves this page itself
 */
(function () {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("api");

  const base =
    override ||
    window.OILSPILL_API_BASE ||
    "";

  // Trailing slashes produce "//api/health", which some routers reject.
  window.API_BASE = base.replace(/\/+$/, "");

  window.apiUrl = function (path) {
    return window.API_BASE + (path.startsWith("/") ? path : "/" + path);
  };

  if (window.API_BASE) {
    console.info("Oil spill API base:", window.API_BASE);
  }
})();
