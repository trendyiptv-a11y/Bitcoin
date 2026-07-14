package app.cohesivx.btcmonitor;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.graphics.Bitmap;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends Activity {
    private static final String START_URL = "https://coezivx.vercel.app/btc-swing-strategy/mecanism.html";
    private static final String BASE_URL = "https://coezivx.vercel.app/btc-swing-strategy/";
    private static final int PULL_REFRESH_DISTANCE_PX = 180;
    private static final String PREF_LAST_SUCCESSFUL_LOAD = "last_successful_load";
    private static final String OFFLINE_HTML_FILE = "offline_snapshot.html";
    private static final String HISTORY_PREFIX = "snapshot_";
    private static final String HISTORY_SUFFIX = ".html";
    private static final int MAX_HISTORY_SNAPSHOTS = 30;
    private static final String[] SNAPSHOT_FILES = new String[]{
            "coeziv_state.json",
            "risk_window.json",
            "participation_cohesion_test.json",
            "participation_cohesion_history_summary.json"
    };

    private WebView webView;
    private ProgressBar progressBar;
    private TextView offlineMessage;
    private View splashOverlay;
    private TextView btnTheme;
    private float pullStartY = 0f;
    private boolean pullRefreshArmed = false;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        configureSystemBars();
        setContentView(R.layout.activity_main);
        applySystemBarPadding();
        webView        = findViewById(R.id.webView);
        progressBar    = findViewById(R.id.progressBar);
        offlineMessage = findViewById(R.id.offlineMessage);
        btnTheme       = findViewById(R.id.btnTheme);
        if (btnTheme != null) btnTheme.setText("LIGHT");
        if (btnTheme != null) btnTheme.setOnClickListener(v -> {
            if (webView != null) {
                webView.evaluateJavascript("document.getElementById('theme-toggle') && document.getElementById('theme-toggle').click();", value -> new Handler(Looper.getMainLooper()).postDelayed(this::syncThemeButton, 250));
            }
        });
        View infoButton = findViewById(R.id.btnInfo);
        if (infoButton != null) infoButton.setOnClickListener(v -> showAboutDialog());
        createSplashOverlay();
        NotificationHelper.ensureReady(this);
        SignalCheckWorker.schedule(this);
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setLoadWithOverviewMode(true);
        settings.setUseWideViewPort(true);
        settings.setBuiltInZoomControls(false);
        settings.setDisplayZoomControls(false);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setCacheMode(WebSettings.LOAD_DEFAULT);
        webView.setOnTouchListener((view, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                pullStartY = event.getY();
                pullRefreshArmed = webView.getScrollY() == 0;
            } else if (event.getAction() == MotionEvent.ACTION_UP) {
                float distance = event.getY() - pullStartY;
                if (pullRefreshArmed && distance > PULL_REFRESH_DISTANCE_PX && webView.getScrollY() == 0) {
                    Toast.makeText(this, "Actualizare mecanism...", Toast.LENGTH_SHORT).show();
                    progressBar.setVisibility(View.VISIBLE);
                    webView.getSettings().setCacheMode(WebSettings.LOAD_DEFAULT);
                    webView.reload();
                    cacheSnapshotFilesInBackground();
                }
                pullRefreshArmed = false;
            } else if (event.getAction() == MotionEvent.ACTION_CANCEL) {
                pullRefreshArmed = false;
            }
            return false;
        });
        webView.setWebChromeClient(new WebChromeClient() {
            @Override public void onProgressChanged(WebView view, int newProgress) {
                progressBar.setProgress(newProgress);
                progressBar.setVisibility(newProgress < 100 ? View.VISIBLE : View.GONE);
            }
        });
        webView.setWebViewClient(new WebViewClient() {
            @Override public void onPageStarted(WebView view, String url, Bitmap favicon) {
                progressBar.setVisibility(View.VISIBLE); super.onPageStarted(view, url, favicon);
            }
            @Override public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                if (!hasNetwork()) { WebResourceResponse cached = cachedJsonResponse(request.getUrl()); if (cached != null) return cached; }
                return super.shouldInterceptRequest(view, request);
            }
            @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                String url = request.getUrl().toString(); if (url.startsWith("https://coezivx.vercel.app")) return false; view.loadUrl(url); return true;
            }
            @Override public void onPageFinished(WebView view, String url) {
                progressBar.setVisibility(View.GONE); hideSplashOverlay(); injectNativeTopBarCss();
                if (url != null && url.startsWith("https://coezivx.vercel.app") && hasNetwork()) {
                    getPreferences(MODE_PRIVATE).edit().putLong(PREF_LAST_SUCCESSFUL_LOAD, System.currentTimeMillis()).apply();
                    cacheSnapshotFilesInBackground(); saveRenderedSnapshotDelayed();
                }
                super.onPageFinished(view, url);
            }
        });
        showDisclaimerOnce(); loadApp();
    }

    private void injectNativeTopBarCss() {
        if (webView == null) return;
        String js = "(function(){if(document.getElementById('cohesivx-native-topbar-css'))return;var s=document.createElement('style');s.id='cohesivx-native-topbar-css';s.textContent='.title-bar{display:none!important;}#theme-toggle{display:none!important;}#app-download-banner{display:none!important;}.top-controls{display:flex!important;justify-content:flex-start!important;margin:0 0 10px!important;}';document.head.appendChild(s);})()";
        webView.evaluateJavascript(js, value -> syncThemeButton());
    }
    private void syncThemeButton() {
        if (webView == null || btnTheme == null) return;
        webView.evaluateJavascript("(function(){return document.body.classList.contains('light-mode')?'LIGHT':'DARK';})()", value -> {
            String mode = (value != null && value.contains("LIGHT")) ? "LIGHT" : "DARK";
            String action = "LIGHT".equals(mode) ? "DARK" : "LIGHT";
            runOnUiThread(() -> btnTheme.setText(action));
        });
    }
    private void saveRenderedSnapshotDelayed() {
        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            String js = "(function(){var c=document.documentElement.cloneNode(true);var s=c.querySelectorAll('script');for(var i=0;i<s.length;i++){s[i].parentNode.removeChild(s[i]);}var b=c.querySelector('body');if(b){var n=document.createElement('div');n.style.cssText='position:fixed;left:0;right:0;bottom:0;z-index:999999;background:#020617;color:#94A3B8;text-align:center;font:12px sans-serif;padding:8px;';n.textContent='Snapshot offline salvat local de CohesivX BTC Monitor';b.appendChild(n);}return '<!DOCTYPE html>'+c.outerHTML;})()";
            webView.evaluateJavascript(js, value -> { try { String html = new JSONArray("[" + value + "]").getString(0); if (html.contains("MECANISM COEZIV BTC") || html.contains("Bitcoin")) saveOfflineHtml(html); } catch (Exception ignored) {} });
        }, 3500);
    }
    private void saveOfflineHtml(String html) {
        try {
            byte[] bytes = html.getBytes(StandardCharsets.UTF_8);
            File latest = new File(getFilesDir(), OFFLINE_HTML_FILE);
            try (FileOutputStream fos = new FileOutputStream(latest, false)) { fos.write(bytes); }
            String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            File history = new File(getFilesDir(), HISTORY_PREFIX + stamp + HISTORY_SUFFIX);
            try (FileOutputStream fos = new FileOutputStream(history, false)) { fos.write(bytes); }
            pruneSnapshotHistory();
        } catch (Exception ignored) {}
    }
    private void pruneSnapshotHistory() {
        File[] files = getFilesDir().listFiles((dir, name) -> name.startsWith(HISTORY_PREFIX) && name.endsWith(HISTORY_SUFFIX));
        if (files == null || files.length <= MAX_HISTORY_SNAPSHOTS) return;
        Arrays.sort(files, Comparator.comparingLong(File::lastModified));
        int toDelete = files.length - MAX_HISTORY_SNAPSHOTS;
        for (int i = 0; i < toDelete; i++) { try { files[i].delete(); } catch (Exception ignored) {} }
    }
    private int countSnapshotHistory() { File[] files = getSnapshotHistoryFiles(); return files == null ? 0 : files.length; }
    private File[] getSnapshotHistoryFiles() {
        File[] files = getFilesDir().listFiles((dir, name) -> name.startsWith(HISTORY_PREFIX) && name.endsWith(HISTORY_SUFFIX));
        if (files == null) return new File[0]; Arrays.sort(files, (a, b) -> Long.compare(b.lastModified(), a.lastModified())); return files;
    }
    private String labelForSnapshot(File file) {
        String name = file.getName();
        try { String raw = name.replace(HISTORY_PREFIX, "").replace(HISTORY_SUFFIX, ""); if (raw.length() >= 15) return raw.substring(6, 8) + "." + raw.substring(4, 6) + "." + raw.substring(0, 4) + " " + raw.substring(9, 11) + ":" + raw.substring(11, 13) + ":" + raw.substring(13, 15); } catch (Exception ignored) {}
        return name;
    }
    private void showSnapshotHistoryDialog() { showSnapshotHistoryDialog(false); }
    private void showSnapshotHistoryDialog(boolean english) {
        File[] files = getSnapshotHistoryFiles();
        if (files.length == 0) { Toast.makeText(this, english ? "No local snapshots yet." : "Nu există încă snapshoturi locale.", Toast.LENGTH_LONG).show(); return; }
        String[] labels = new String[files.length]; for (int i = 0; i < files.length; i++) labels[i] = labelForSnapshot(files[i]);
        new AlertDialog.Builder(this).setTitle(english ? "Local history" : "Istoric local").setItems(labels, (dialog, which) -> loadHistorySnapshot(files[which])).setNegativeButton(english ? "Close" : "Închide", null).show();
    }
    private void loadHistorySnapshot(File file) {
        if (file == null || !file.exists() || file.length() <= 0) { Toast.makeText(this, "Snapshot indisponibil.", Toast.LENGTH_SHORT).show(); return; }
        try { byte[] bytes = readAllBytes(new FileInputStream(file)); String html = new String(bytes, StandardCharsets.UTF_8); offlineMessage.setVisibility(View.GONE); webView.setVisibility(View.VISIBLE); webView.loadDataWithBaseURL(START_URL, html, "text/html", "UTF-8", START_URL); hideSplashOverlay(); Toast.makeText(this, "Snapshot încărcat: " + labelForSnapshot(file), Toast.LENGTH_LONG).show(); } catch (Exception e) { Toast.makeText(this, "Nu am putut încărca snapshotul.", Toast.LENGTH_SHORT).show(); }
    }
    private void loadOfflineRenderedSnapshot() {
        File f = new File(getFilesDir(), OFFLINE_HTML_FILE); if (!f.exists() || f.length() <= 0) return;
        try { byte[] bytes = readAllBytes(new FileInputStream(f)); String html = new String(bytes, StandardCharsets.UTF_8); offlineMessage.setVisibility(View.GONE); webView.setVisibility(View.VISIBLE); webView.loadDataWithBaseURL(START_URL, html, "text/html", "UTF-8", START_URL); hideSplashOverlay(); Toast.makeText(this, "Mod offline: snapshot local complet.", Toast.LENGTH_LONG).show(); } catch (Exception ignored) {}
    }
    private WebResourceResponse cachedJsonResponse(Uri uri) {
        if (uri == null) return null; String name = fileNameFromUrl(uri.toString()); if (name == null) return null; File f = new File(getFilesDir(), name); if (!f.exists() || f.length() <= 0) return null;
        try { return new WebResourceResponse("application/json", "UTF-8", new FileInputStream(f)); } catch (Exception ignored) { return null; }
    }
    private String fileNameFromUrl(String url) {
        if (url == null) return null; for (String name : SNAPSHOT_FILES) if (url.endsWith("/" + name) || url.endsWith(name) || url.contains(name + "?")) return name; return null;
    }
    private void cacheSnapshotFilesInBackground() {
        if (!hasNetwork()) return; new Thread(() -> { for (String file : SNAPSHOT_FILES) cacheOneJson(BASE_URL + file, file); cacheOneJson("https://coezivx.vercel.app/data/participation_cohesion_test.json", "participation_cohesion_test.json"); cacheOneJson("https://coezivx.vercel.app/data/participation_cohesion_history_summary.json", "participation_cohesion_history_summary.json"); NotificationHelper.checkStructuralChangeFromCache(this); }).start();
    }
    private void cacheOneJson(String urlText, String fileName) {
        HttpURLConnection conn = null;
        try { URL url = new URL(urlText + (urlText.contains("?") ? "&" : "?") + "t=" + System.currentTimeMillis()); conn = (HttpURLConnection) url.openConnection(); conn.setConnectTimeout(6000); conn.setReadTimeout(6000); conn.setUseCaches(false); if (conn.getResponseCode() != 200) return; byte[] bytes = readAllBytes(conn.getInputStream()); if (bytes == null || bytes.length == 0) return; String probe = new String(bytes, StandardCharsets.UTF_8).trim(); if (!(probe.startsWith("{") || probe.startsWith("["))) return; File out = new File(getFilesDir(), fileName); try (FileOutputStream fos = new FileOutputStream(out, false)) { fos.write(bytes); } } catch (Exception ignored) {} finally { if (conn != null) conn.disconnect(); }
    }
    private byte[] readAllBytes(InputStream input) throws Exception {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream(); byte[] data = new byte[4096]; int n; while ((n = input.read(data)) != -1) buffer.write(data, 0, n); return buffer.toByteArray();
    }
    private void createSplashOverlay() {
        android.widget.FrameLayout inner = findViewById(R.id.webViewFrame); if (inner == null) return;
        LinearLayout box = new LinearLayout(this); box.setOrientation(LinearLayout.VERTICAL); box.setGravity(android.view.Gravity.CENTER); box.setPadding(48, 48, 48, 48); box.setBackgroundColor(getColor(R.color.app_background));
        ImageView logo = new ImageView(this); logo.setImageResource(R.drawable.ic_launcher_foreground); LinearLayout.LayoutParams logoParams = new LinearLayout.LayoutParams(220, 220); box.addView(logo, logoParams);
        TextView title = new TextView(this); title.setText("COHESIVX"); title.setTextColor(getColor(R.color.text_primary)); title.setTextSize(26); title.setGravity(android.view.Gravity.CENTER); title.setLetterSpacing(0.18f); title.setTypeface(null, android.graphics.Typeface.BOLD); LinearLayout.LayoutParams titleParams = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT); titleParams.topMargin = 28; box.addView(title, titleParams);
        TextView subtitle = new TextView(this); subtitle.setText("BTC MONITOR"); subtitle.setTextColor(getColor(R.color.accent_primary)); subtitle.setTextSize(15); subtitle.setGravity(android.view.Gravity.CENTER); subtitle.setLetterSpacing(0.12f); LinearLayout.LayoutParams subParams = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT); subParams.topMargin = 10; box.addView(subtitle, subParams);
        android.widget.ProgressBar spinner = new android.widget.ProgressBar(this); LinearLayout.LayoutParams spinParams = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT); spinParams.topMargin = 36; box.addView(spinner, spinParams);
        inner.addView(box, new android.widget.FrameLayout.LayoutParams(android.widget.FrameLayout.LayoutParams.MATCH_PARENT, android.widget.FrameLayout.LayoutParams.MATCH_PARENT)); splashOverlay = box;
    }
    private void hideSplashOverlay() { if (splashOverlay == null) return; splashOverlay.animate().alpha(0f).setDuration(250).withEndAction(() -> splashOverlay.setVisibility(View.GONE)).start(); }
    private void showAboutDialog() {
        if (webView == null) { showAboutDialog(false); return; }
        String js = "(function(){try{return (localStorage.getItem('coeziv_btc_lang') || document.documentElement.lang || 'ro');}catch(e){return 'ro';}})()";
        webView.evaluateJavascript(js, value -> runOnUiThread(() -> showAboutDialog(isEnglishValue(value))));
    }
    private boolean isEnglishValue(String value) { return value != null && value.toLowerCase(Locale.US).contains("en"); }
    private void showAboutDialog(boolean english) {
        int localCount = countSnapshotHistory(); String version = getAppVersionName();
        String message = english
                ? "App version: " + version + "\n\n" +
                "CohesivX BTC Monitor is an experimental tool for structural observation of the Bitcoin ecosystem.\n\n" +
                "The app does not provide entry or exit prompts. It observes live price, mechanism price, participation, liquidity, risk and similar historical contexts.\n\n" +
                "Active modules:\n" +
                "• Structural BTC Radar\n" +
                "• Cohesive BTC Mechanism\n" +
                "• Participation Cohesion\n" +
                "• Risk Window\n" +
                "• Cohesive Fear & Greed\n" +
                "• Contextual Backtest\n" +
                "• Structural Notifications\n\n" +
                "Updates:\n" +
                "• automatic snapshot\n" +
                "• live BTC data\n" +
                "• manual pull-to-refresh\n" +
                "• local cache for JSON files\n" +
                "• complete local HTML snapshot for offline mode\n" +
                "• local history: " + localCount + " / " + MAX_HISTORY_SNAPSHOTS + " snapshots\n\n" +
                "Model author: Sergiu Bulboacă, Coeziv 3.14 project.\n\n" +
                "This is not financial advice. It does not execute transactions and does not manage funds."
                : "Versiune aplicație: " + version + "\n\n" +
                "CohesivX BTC Monitor este un instrument experimental de observare structurală a ecosistemului Bitcoin.\n\n" +
                "Aplicația nu oferă îndemnuri de intrare sau ieșire. Ea observă prețul live, prețul mecanismului, participarea, lichiditatea, riscul și contexte istorice similare.\n\n" +
                "Module active:\n" +
                "• Radar structural BTC\n" +
                "• Mecanism Coeziv BTC\n" +
                "• Coeziune Participativă\n" +
                "• Fereastră de Risc\n" +
                "• Fear & Greed Coeziv\n" +
                "• Backtest contextual\n" +
                "• Notificări structurale\n\n" +
                "Actualizare:\n" +
                "• snapshot automat\n" +
                "• date BTC live\n" +
                "• refresh manual prin tragere în jos\n" +
                "• cache local pentru JSON-uri\n" +
                "• snapshot HTML local complet pentru modul offline\n" +
                "• istoric local: " + localCount + " / " + MAX_HISTORY_SNAPSHOTS + " snapshoturi\n\n" +
                "Autor model: Sergiu Bulboacă, proiectul Coeziv 3.14.\n\n" +
                "Nu este recomandare financiară. Nu execută tranzacții și nu administrează fonduri.";
        new AlertDialog.Builder(this).setTitle(english ? "About CohesivX" : "Despre CohesivX").setMessage(message).setNeutralButton(english ? "Local history" : "Istoric local", (dialog, which) -> showSnapshotHistoryDialog(english)).setPositiveButton(english ? "Close" : "Închide", null).show();
    }
    private String getAppVersionName() { try { return getPackageManager().getPackageInfo(getPackageName(), 0).versionName; } catch (Exception ignored) { return "0.2.7"; } }
    private void configureSystemBars() { Window window = getWindow(); window.setStatusBarColor(getColor(R.color.app_background)); window.setNavigationBarColor(getColor(R.color.app_background)); window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS); window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_NAVIGATION); }
    private void applySystemBarPadding() { final View root = findViewById(R.id.appRoot); if (root == null) return; root.setOnApplyWindowInsetsListener((view, insets) -> { android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars()); view.setPadding(bars.left, bars.top, bars.right, bars.bottom); return insets; }); root.requestApplyInsets(); }
    @Override public void onBackPressed() { if (webView != null && webView.canGoBack()) webView.goBack(); else super.onBackPressed(); }
    private void loadApp() {
        if (hasNetwork()) { offlineMessage.setVisibility(View.GONE); webView.setVisibility(View.VISIBLE); webView.getSettings().setCacheMode(WebSettings.LOAD_DEFAULT); webView.loadUrl(START_URL); }
        else { File offlineHtml = new File(getFilesDir(), OFFLINE_HTML_FILE); if (offlineHtml.exists() && offlineHtml.length() > 0) { loadOfflineRenderedSnapshot(); return; } long lastLoad = getPreferences(MODE_PRIVATE).getLong(PREF_LAST_SUCCESSFUL_LOAD, 0L); if (lastLoad > 0L) { offlineMessage.setVisibility(View.GONE); webView.setVisibility(View.VISIBLE); webView.getSettings().setCacheMode(WebSettings.LOAD_CACHE_ELSE_NETWORK); Toast.makeText(this, "Mod offline: se încarcă ultimul snapshot salvat.", Toast.LENGTH_LONG).show(); webView.loadUrl(START_URL); } else { webView.setVisibility(View.GONE); offlineMessage.setVisibility(View.VISIBLE); hideSplashOverlay(); } }
    }
    private boolean hasNetwork() { ConnectivityManager cm = (ConnectivityManager) getSystemService(CONNECTIVITY_SERVICE); if (cm == null) return false; Network network = cm.getActiveNetwork(); if (network == null) return false; NetworkCapabilities caps = cm.getNetworkCapabilities(network); return caps != null && caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET); }
    private void showDisclaimerOnce() { boolean accepted = getPreferences(MODE_PRIVATE).getBoolean("disclaimer_accepted", false); if (accepted) return; new AlertDialog.Builder(this).setTitle(R.string.disclaimer_title).setMessage(R.string.disclaimer_text).setCancelable(false).setPositiveButton(R.string.disclaimer_accept, (dialog, which) -> getPreferences(MODE_PRIVATE).edit().putBoolean("disclaimer_accepted", true).apply()).show(); }
}
