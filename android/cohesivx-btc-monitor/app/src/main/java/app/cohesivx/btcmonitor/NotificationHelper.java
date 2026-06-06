package app.cohesivx.btcmonitor;

import android.Manifest;
import android.app.Activity;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;

import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public final class NotificationHelper {
    private static final String CHANNEL_ID = "cohesivx_structure";
    private static final String CHANNEL_NAME = "CohesivX structură";
    private static final String PREFS = "cohesivx_notifications";
    private static final String KEY_LAST_STRUCTURE = "last_structure_key";
    private static final int NOTIFICATION_ID = 31401;

    private NotificationHelper() {}

    public static void ensureReady(Activity activity) {
        createChannel(activity);
        if (Build.VERSION.SDK_INT >= 33 &&
                activity.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            activity.requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 314);
        }
    }

    public static void checkStructuralChangeFromCache(Context context) {
        try {
            StructuralSnapshot snap = readStructuralSnapshot(context);
            if (snap.isEmpty()) return;

            String key = snap.key();
            String old = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY_LAST_STRUCTURE, "");
            if (old == null || old.isEmpty()) {
                context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY_LAST_STRUCTURE, key).apply();
                return;
            }
            if (old.equals(key)) return;

            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY_LAST_STRUCTURE, key).apply();

            notify(
                    context,
                    "CohesivX: structură modificată",
                    snap.notificationText()
            );
        } catch (Exception ignored) {
        }
    }

    public static void checkContextChangeFromCache(Context context) {
        checkStructuralChangeFromCache(context);
    }

    private static StructuralSnapshot readStructuralSnapshot(Context context) {
        StructuralSnapshot snap = new StructuralSnapshot();

        JSONObject state = readJson(context, "coeziv_state.json");
        if (state != null) {
            JSONObject regime = state.optJSONObject("market_regime");
            snap.marketRegime = regime != null ? clean(regime.optString("label", "")) : "";
            JSONObject fg = state.optJSONObject("fg");
            if (fg != null) {
                snap.fearGreedZone = clean(fg.optString("combined_zone", ""));
                double combined = fg.optDouble("combined", Double.NaN);
                if (!Double.isNaN(combined)) snap.fearGreedValue = String.valueOf(Math.round(combined));
            }
        }

        JSONObject risk = readJson(context, "risk_window.json");
        if (risk != null) {
            snap.riskLabel = firstNonEmpty(
                    risk.optString("label", ""),
                    risk.optString("risk_label", ""),
                    risk.optString("status", ""),
                    risk.optString("context", "")
            );
            snap.riskLabel = clean(snap.riskLabel);
        }

        JSONObject participationRoot = readJson(context, "participation_cohesion_test.json");
        if (participationRoot != null) {
            JSONObject p = participationRoot.optJSONObject("participation_cohesion_test");
            if (p == null) p = participationRoot;
            snap.participationLabel = clean(p.optString("label", ""));
            double score = p.optDouble("score", Double.NaN);
            if (!Double.isNaN(score)) snap.participationScore = String.valueOf(Math.round(score));
        }

        snap.generalState = inferGeneralState(snap);
        return snap;
    }

    private static String inferGeneralState(StructuralSnapshot s) {
        int pressure = 0;
        String joined = (s.marketRegime + " " + s.riskLabel + " " + s.participationLabel + " " + s.fearGreedZone).toLowerCase();
        if (joined.contains("degrad")) pressure += 3;
        if (joined.contains("fragil")) pressure += 2;
        if (joined.contains("risc")) pressure += 2;
        if (joined.contains("scădere") || joined.contains("scadere")) pressure += 2;
        if (joined.contains("tension")) pressure += 1;
        if (joined.contains("fear")) pressure += 1;
        if (joined.contains("coeziv")) pressure -= 2;
        if (joined.contains("neutral") || joined.contains("neutru")) pressure -= 1;
        if (joined.contains("pozitiv")) pressure -= 1;

        if (pressure >= 5) return "structură degradată";
        if (pressure >= 3) return "structură fragilă";
        if (pressure >= 1) return "structură tensionată";
        return "structură stabilă";
    }

    private static JSONObject readJson(Context context, String fileName) {
        try {
            File f = new File(context.getFilesDir(), fileName);
            if (!f.exists() || f.length() <= 0) return null;
            return new JSONObject(readText(f));
        } catch (Exception ignored) {
            return null;
        }
    }

    private static String firstNonEmpty(String... values) {
        if (values == null) return "";
        for (String v : values) {
            String c = clean(v);
            if (!c.isEmpty()) return c;
        }
        return "";
    }

    private static String clean(String s) {
        if (s == null) return "";
        return s.replace("_", " ").replace("-", " ").trim();
    }

    private static void createChannel(Context context) {
        if (Build.VERSION.SDK_INT < 26) return;
        NotificationManager nm = (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm == null) return;
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_DEFAULT
        );
        channel.setDescription("Notificări când starea structurală CohesivX se modifică.");
        nm.createNotificationChannel(channel);
    }

    private static void notify(Context context, String title, String text) {
        if (Build.VERSION.SDK_INT >= 33 &&
                context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            return;
        }
        android.app.Notification.Builder builder = Build.VERSION.SDK_INT >= 26
                ? new android.app.Notification.Builder(context, CHANNEL_ID)
                : new android.app.Notification.Builder(context);

        builder.setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle(title)
                .setContentText(text)
                .setStyle(new android.app.Notification.BigTextStyle().bigText(text))
                .setAutoCancel(true)
                .setShowWhen(true);

        NotificationManager nm = (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm != null) nm.notify(NOTIFICATION_ID, builder.build());
    }

    private static String readText(File file) throws Exception {
        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] data = new byte[(int) file.length()];
            int read = fis.read(data);
            if (read <= 0) return "";
            return new String(data, 0, read, StandardCharsets.UTF_8);
        }
    }

    private static final class StructuralSnapshot {
        String generalState = "";
        String participationLabel = "";
        String participationScore = "";
        String riskLabel = "";
        String fearGreedZone = "";
        String fearGreedValue = "";
        String marketRegime = "";

        boolean isEmpty() {
            return key().replace("|", "").trim().isEmpty();
        }

        String key() {
            return generalState + "|" + participationLabel + "|" + participationScore + "|" + riskLabel + "|" + fearGreedZone + "|" + fearGreedValue + "|" + marketRegime;
        }

        String notificationText() {
            List<String> parts = new ArrayList<>();
            if (!generalState.isEmpty()) parts.add("Stare generală: " + generalState);
            if (!participationLabel.isEmpty()) {
                String p = "Participare: " + participationLabel;
                if (!participationScore.isEmpty()) p += " (" + participationScore + "/100)";
                parts.add(p);
            }
            if (!riskLabel.isEmpty()) parts.add("Fereastră risc: " + riskLabel);
            if (!fearGreedZone.isEmpty()) {
                String fg = "Fear & Greed: " + fearGreedZone;
                if (!fearGreedValue.isEmpty()) fg += " (" + fearGreedValue + ")";
                parts.add(fg);
            }
            if (!marketRegime.isEmpty()) parts.add("Regim: " + marketRegime);
            if (parts.isEmpty()) return "Mecanismul a detectat o schimbare în structura rețelei.";
            StringBuilder out = new StringBuilder();
            for (int i = 0; i < parts.size(); i++) {
                if (i > 0) out.append("\n");
                out.append(parts.get(i));
            }
            return out.toString();
        }
    }
}
