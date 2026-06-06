package app.cohesivx.btcmonitor;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.graphics.Bitmap;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

public class MainActivity extends Activity {
    private static final String START_URL = "https://coezivx.vercel.app/btc-swing-strategy/mecanism.html";

    private WebView webView;
    private ProgressBar progressBar;
    private TextView offlineMessage;
    private View splashOverlay;

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
                super.onPageFinished(view, url);
            }
        });

        showDisclaimerOnce();
        loadApp();
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
            webView.loadUrl(START_URL);
        } else {
            webView.setVisibility(View.GONE);
            offlineMessage.setVisibility(View.VISIBLE);
            hideSplashOverlay();
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
