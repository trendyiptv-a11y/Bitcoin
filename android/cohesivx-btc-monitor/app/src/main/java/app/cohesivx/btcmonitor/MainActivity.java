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
import android.view.Gravity;
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
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class MainActivity extends Activity {
    private static final String START_URL = "https://coezivx.vercel.app/btc-swing-strategy/mecanism.html";
    private static final String BASE_URL = "https://coezivx.vercel.app/btc-swing-strategy/";
    private static final int PULL_REFRESH_DISTANCE_PX = 180;
    private static final String PREF_LAST_SUCCESSFUL_LOAD = "last_successful_load";
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
    private float pullStartY = 0f;
    private boolean pullRefreshArmed = false;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        configureSystemBars();
        setContentView(R.layout.activity_main);
        applySystemBarPadding();

        webView = findViewById(R.id.webView);
        progressBar = findViewById(R.id.progressBar);
        offlineMessage = findViewById(R.id.offlineMessage);
        createSplashOverlay();
        createAboutButton();

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
            @Override
            public void onProgressChanged(WebView view, int newProgress) {
                progressBar.setProgress(newProgress);
                progressBar.setVisibility(newProgress < 100 ? View.VISIBLE : View.GONE);
            }
        });

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageStarted(WebView view, String url, Bitmap favicon) {
                progressBar.setVisibility(View.VISIBLE);
                super.onPageStarted(view, url, favicon);
            }

            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                if (!hasNetwork()) {
                    WebResourceResponse cached = cachedJsonResponse(request.getUrl());
                    if (cached != null) return cached;
                }
                return super.shouldInterceptRequest(view, request);
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                String url = request.getUrl().toString();
                if (url.startsWith("https://coezivx.vercel.app")) {
                    return false;
                }
                view.loadUrl(url);
                return true;
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                progressBar.setVisibility(View.GONE);
                hideSplashOverlay();
                if (url != null && url.startsWith("https://coezivx.vercel.app") && hasNetwork()) {
                    getPreferences(MODE_PRIVATE).edit()
                            .putLong(PREF_LAST_SUCCESSFUL_LOAD, System.currentTimeMillis())
                            .apply();
                    cacheSnapshotFilesInBackground();
                }
                super.onPageFinished(view, url);
            }
        });

        showDisclaimerOnce();
        loadApp();
    }

    private WebResourceResponse cachedJsonResponse(Uri uri) {
        if (uri == null) return null;
        String name = fileNameFromUrl(uri.toString());
        if (name == null) return null;
        File f = new File(getFilesDir(), name);
        if (!f.exists() || f.length() <= 0) return null;
        try {
            return new WebResourceResponse("application/json", "UTF-8", new FileInputStream(f));
        } catch (Exception ignored) {
            return null;
        }
    }

    private String fileNameFromUrl(String url) {
        if (url == null) return null;
        for (String name : SNAPSHOT_FILES) {
            if (url.endsWith("/" + name) || url.endsWith(name) || url.contains(name + "?")) {
                return name;
            }
        }
        return null;
    }

    private void cacheSnapshotFilesInBackground() {
        if (!hasNetwork()) return;
        new Thread(() -> {
            for (String file : SNAPSHOT_FILES) {
                cacheOneJson(BASE_URL + file, file);
            }
            cacheOneJson("https://coezivx.vercel.app/data/participation_cohesion_test.json", "participation_cohesion_test.json");
            cacheOneJson("https://coezivx.vercel.app/data/participation_cohesion_history_summary.json", "participation_cohesion_history_summary.json");
        }).start();
    }

    private void cacheOneJson(String urlText, String fileName) {
        HttpURLConnection conn = null;
        try {
            URL url = new URL(urlText + (urlText.contains("?") ? "&" : "?") + "t=" + System.currentTimeMillis());
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(6000);
            conn.setReadTimeout(6000);
            conn.setUseCaches(false);
            if (conn.getResponseCode() != 200) return;
            byte[] bytes = readAllBytes(conn.getInputStream());
            if (bytes == null || bytes.length == 0) return;
            String probe = new String(bytes, StandardCharsets.UTF_8).trim();
            if (!(probe.startsWith("{") || probe.startsWith("["))) return;
            File out = new File(getFilesDir(), fileName);
            try (FileOutputStream fos = new FileOutputStream(out, false)) {
                fos.write(bytes);
            }
        } catch (Exception ignored) {
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    private byte[] readAllBytes(InputStream input) throws Exception {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] data = new byte[4096];
        int n;
        while ((n = input.read(data)) != -1) {
            buffer.write(data, 0, n);
        }
        return buffer.toByteArray();
    }

    private void createAboutButton() {
        FrameLayout root = findViewById(R.id.appRoot);
        if (root == null) return;
        TextView button = new TextView(this);
        button.setText("ⓘ");
        button.setTextColor(getColor(R.color.accent_primary));
        button.setTextSize(22);
        button.setGravity(Gravity.CENTER);
        button.setBackgroundColor(getColor(R.color.surface_dark));
        button.setAlpha(0.88f);
        button.setOnClickListener(v -> showAboutDialog());
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(82, 82);
        params.gravity = Gravity.TOP | Gravity.END;
        params.topMargin = 16;
        params.rightMargin = 16;
        root.addView(button, params);
    }

    private void showAboutDialog() {
        String message = "Versiune aplicație: 0.2.1\n\n" +
                "CohesivX BTC Monitor este un instrument experimental de observare structurală a ecosistemului Bitcoin.\n\n" +
                "Module active:\n" +
                "• Mecanism Coeziv BTC\n" +
                "• Coeziune Participativă\n" +
                "• Fereastră de Risc\n" +
                "• Fear & Greed Coeziv\n" +
                "• Backtest contextual\n\n" +
                "Actualizare:\n" +
                "• snapshot automat\n" +
                "• date BTC live\n" +
                "• refresh manual prin tragere în jos\n" +
                "• cache local pentru JSON-urile mecanismului\n\n" +
                "Autor model: Sergiu Bulboacă, proiectul Coeziv 3.14.\n\n" +
                "Nu este recomandare financiară. Nu execută tranzacții și nu administrează fonduri.";
        new AlertDialog.Builder(this)
                .setTitle("Despre CohesivX")
                .setMessage(message)
                .setPositiveButton("Închide", null)
                .show();
    }

    private void createSplashOverlay() {
        FrameLayout root = findViewById(R.id.appRoot);
        if (root == null) return;

        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setGravity(Gravity.CENTER);
        box.setPadding(48, 48, 48, 48);
        box.setBackgroundColor(getColor(R.color.app_background));

        ImageView logo = new ImageView(this);
        logo.setImageResource(R.drawable.ic_launcher_foreground);
        LinearLayout.LayoutParams logoParams = new LinearLayout.LayoutParams(220, 220);
        box.addView(logo, logoParams);

        TextView title = new TextView(this);
        title.setText("COHESIVX");
        title.setTextColor(getColor(R.color.text_primary));
        title.setTextSize(26);
        title.setGravity(Gravity.CENTER);
        title.setLetterSpacing(0.18f);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        LinearLayout.LayoutParams titleParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        titleParams.topMargin = 28;
        box.addView(title, titleParams);

        TextView subtitle = new TextView(this);
        subtitle.setText("BTC MONITOR");
        subtitle.setTextColor(getColor(R.color.accent_primary));
        subtitle.setTextSize(15);
        subtitle.setGravity(Gravity.CENTER);
        subtitle.setLetterSpacing(0.12f);
        LinearLayout.LayoutParams subParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        subParams.topMargin = 10;
        box.addView(subtitle, subParams);

        ProgressBar spinner = new ProgressBar(this);
        LinearLayout.LayoutParams spinParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        spinParams.topMargin = 36;
        box.addView(spinner, spinParams);

        FrameLayout.LayoutParams overlayParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
        );
        root.addView(box, overlayParams);
        splashOverlay = box;
    }

    private void hideSplashOverlay() {
        if (splashOverlay == null) return;
        splashOverlay.animate()
                .alpha(0f)
                .setDuration(250)
                .withEndAction(() -> splashOverlay.setVisibility(View.GONE))
                .start();
    }

    private void configureSystemBars() {
        Window window = getWindow();
        window.setStatusBarColor(getColor(R.color.app_background));
        window.setNavigationBarColor(getColor(R.color.app_background));
        window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
        window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_NAVIGATION);
    }

    private void applySystemBarPadding() {
        final View root = findViewById(R.id.appRoot);
        if (root == null) return;
        root.setOnApplyWindowInsetsListener((view, insets) -> {
            android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars());
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            return insets;
        });
        root.requestApplyInsets();
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    private void loadApp() {
        if (hasNetwork()) {
            offlineMessage.setVisibility(View.GONE);
            webView.setVisibility(View.VISIBLE);
            webView.getSettings().setCacheMode(WebSettings.LOAD_DEFAULT);
            webView.loadUrl(START_URL);
        } else {
            long lastLoad = getPreferences(MODE_PRIVATE).getLong(PREF_LAST_SUCCESSFUL_LOAD, 0L);
            if (lastLoad > 0L) {
                offlineMessage.setVisibility(View.GONE);
                webView.setVisibility(View.VISIBLE);
                webView.getSettings().setCacheMode(WebSettings.LOAD_CACHE_ELSE_NETWORK);
                Toast.makeText(this, "Mod offline: se încarcă ultimul snapshot salvat.", Toast.LENGTH_LONG).show();
                webView.loadUrl(START_URL);
            } else {
                webView.setVisibility(View.GONE);
                offlineMessage.setVisibility(View.VISIBLE);
                hideSplashOverlay();
            }
        }
    }

    private boolean hasNetwork() {
        ConnectivityManager cm = (ConnectivityManager) getSystemService(CONNECTIVITY_SERVICE);
        if (cm == null) return false;
        Network network = cm.getActiveNetwork();
        if (network == null) return false;
        NetworkCapabilities caps = cm.getNetworkCapabilities(network);
        return caps != null && caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET);
    }

    private void showDisclaimerOnce() {
        boolean accepted = getPreferences(MODE_PRIVATE).getBoolean("disclaimer_accepted", false);
        if (accepted) return;
        new AlertDialog.Builder(this)
                .setTitle(R.string.disclaimer_title)
                .setMessage(R.string.disclaimer_text)
                .setCancelable(false)
                .setPositiveButton(R.string.disclaimer_accept, (dialog, which) ->
                        getPreferences(MODE_PRIVATE).edit().putBoolean("disclaimer_accepted", true).apply())
                .show();
    }
}
