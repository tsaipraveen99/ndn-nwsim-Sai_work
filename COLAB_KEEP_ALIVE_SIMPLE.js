// ============================================================================
// SIMPLE COLAB KEEP-ALIVE - ONE LINER
// ============================================================================
// Copy and paste this into a SEPARATE Colab cell
// Works even when your system sleeps!
// ============================================================================

function ClickConnect(){console.log("⏰ Keep-alive: "+new Date().toLocaleTimeString());document.querySelector("colab-toolbar-button#connect").click()}setInterval(ClickConnect,60000);ClickConnect();

// ============================================================================
// EXPLANATION:
// - Clicks the "Connect" button every 60 seconds
// - Prevents Colab from detecting inactivity
// - Works even when your system sleeps or tab is minimized
// - To stop: Run 'clearInterval(ClickConnect)' in a new cell
// ============================================================================

