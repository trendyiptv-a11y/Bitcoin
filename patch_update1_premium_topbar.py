from pathlib import Path
import re

main_path = Path('android/cohesivx-btc-monitor/app/src/main/java/app/cohesivx/btcmonitor/MainActivity.java')
gradle_path = Path('android/cohesivx-btc-monitor/app/build.gradle')

s = main_path.read_text(encoding='utf-8')

s = s.replace('        createAboutButton();', '        createPremiumTopBar();')

old = r'''    private void createAboutButton\(\) \{.*?\n    \}\n\n    private void showAboutDialog\(\) \{'''
new = '''    private void createPremiumTopBar() {
        FrameLayout root = findViewById(R.id.appRoot);
        if (root == null) return;

        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(18, 8, 18, 8);
        bar.setBackgroundColor(getColor(R.color.surface_dark));
        bar.setAlpha(0.94f);
        bar.setElevation(18f);

        TextView themeButton = new TextView(this);
        themeButton.setText("DARK");
        themeButton.setTextColor(getColor(R.color.accent_primary));
        themeButton.setTextSize(12);
        themeButton.setGravity(Gravity.CENTER);
        themeButton.setPadding(12, 6, 12, 6);
        themeButton.setOnClickListener(v -> {
            if (webView != null) {
                webView.evaluateJavascript("document.getElementById('theme-toggle') && document.getElementById('theme-toggle').click();", null);
            }
        });

        TextView title = new TextView(this);
        title.setText("MECANISM COEZIV BTC");
        title.setTextColor(getColor(R.color.text_secondary));
        title.setTextSize(13);
        title.setGravity(Gravity.CENTER);
        title.setLetterSpacing(0.18f);

        TextView infoButton = new TextView(this);
        infoButton.setText("ⓘ");
        infoButton.setTextColor(getColor(R.color.accent_primary));
        infoButton.setTextSize(23);
        infoButton.setGravity(Gravity.CENTER);
        infoButton.setOnClickListener(v -> showAboutDialog());

        LinearLayout.LayoutParams sideParams = new LinearLayout.LayoutParams(88, 56);
        LinearLayout.LayoutParams titleParams = new LinearLayout.LayoutParams(0, 56, 1f);
        bar.addView(themeButton, sideParams);
        bar.addView(title, titleParams);
        bar.addView(infoButton, sideParams);

        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                72
        );
        params.gravity = Gravity.TOP;
        root.addView(bar, params);

        if (webView != null) {
            webView.setPadding(0, 72, 0, 0);
            webView.setClipToPadding(false);
        }
    }

    private void showAboutDialog() {'''

if 'private void createPremiumTopBar()' not in s:
    s = re.sub(old, new, s, count=1, flags=re.S)

s = s.replace('Versiune aplicație: 0.2.5', 'Versiune aplicație: 0.2.6')
main_path.write_text(s, encoding='utf-8')

g = gradle_path.read_text(encoding='utf-8')
g = re.sub(r'versionCode\s+\d+', 'versionCode 4', g)
g = re.sub(r"versionName\s+'[^']+'", "versionName '0.2.6'", g)
gradle_path.write_text(g, encoding='utf-8')
