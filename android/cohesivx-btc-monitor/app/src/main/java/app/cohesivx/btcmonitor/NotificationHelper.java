package app.cohesivx.btcmonitor;

import android.Manifest;
import android.app.Activity;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Build;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;

/**
 * CohesivX local notifications.
 *
 * Trigger principal:
 *   coeziv_state.json -> trader_signal.key
 *
 * Notifică doar când semnalul se schimbă:
 *   wait -> accumulate / buy / attention / sell / risk etc.
 *
 * La prima citire NU notifică. Doar salvează starea inițială.
 */
public final class NotificationHelper {

    private static final String CHANNEL_ID = "cohesivx_btc_signal";
    private static final String CHANNEL_NAME = "CohesivX BTC Signal";

    private static final String PREFS = "cohesivx_notifications";
    private static final String PREF_LAST_TRADER_KEY = "last_trader_signal_key";
    private static final String PREF_LAST_TRADER_LABEL = "last_trader_signal_label";
    private static final String PREF_LAST_STRUCTURAL_SIGNATURE = "last_structural_signature";

    private static final int REQ_POST_NOTIFICATIONS = 314;
    private static final int NOTIFICATION_ID_TRADER_SIGNAL = 9101;

    private NotificationHelper() {}

    public static void ensureReady(Activity activity) {
        if (activity == null) return;
        createChannel(activity);

        if (Build.VERSION.SDK_INT >= 33 &&
                activity.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                        != PackageManager.PERMISSION_GRANTED) {
            activity.requestPermissions(
                    new String[]{Manifest.permission.POST_NOTIFICATIONS},
                    REQ_POST_NOTIFICATIONS
            );
        }
    }

    public static void checkStructuralChangeFromCache(Context context) {
        if (context == null) return;

        try {
            File stateFile = new File(context.getFilesDir(), "coeziv_state.json");
            if (!stateFile.exists() || stateFile.length() <= 0) return;

            JSONObject state = new JSONObject(readUtf8(stateFile));

            /*
             * 1) Regula nouă, curată:
             * notificare la schimbarea trader_signal.key.
             */
            JSONObject trader = state.optJSONObject("trader_signal");
            if (trader != null) {
                checkTraderSignalChange(context, trader);
                return;
            }

            /*
             * 2) Fallback vechi:
             * dacă JSON-ul nu are încă trader_signal, păstrăm o notificare structurală simplă,
             * ca aplicația să nu rămână mută pe versiuni vechi de date.
             */
            checkStructuralFallback(context);

        } catch (Exception ignored) {
        }
    }

    private static void checkTraderSignalChange(Context context, JSONObject trader) {
        String key = clean(trader.optString("key", ""));
        if (key.isEmpty()) return;

        String label = clean(trader.optString("label", ""));
        if (label.isEmpty()) label = key.toUpperCase();

        String subtitle = clean(trader.optString("subtitle", ""));
        String attitude = clean(trader.optString("attitude", ""));
        String reason = clean(trader.optString("reason", ""));

        SharedPreferences prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        String lastKey = prefs.getString(PREF_LAST_TRADER_KEY, "");

        /*
         * Prima rulare:
         * salvăm starea, dar nu trimitem notificare.
         * Altfel utilizatorul primește alertă imediat după instalare/deschidere,
         * chiar dacă nu s-a schimbat nimic.
         */
        if (lastKey == null || lastKey.isEmpty()) {
            prefs.edit()
                    .putString(PREF_LAST_TRADER_KEY, key)
                    .putString(PREF_LAST_TRADER_LABEL, label)
                    .apply();
            return;
        }

        if (lastKey.equals(key)) return;

        String previousLabel = prefs.getString(PREF_LAST_TRADER_LABEL, lastKey);

        prefs.edit()
                .putString(PREF_LAST_TRADER_KEY, key)
                .putString(PREF_LAST_TRADER_LABEL, label)
                .apply();

        String title = "BTC: " + label;

        String body;
        if (!subtitle.isEmpty()) {
            body = subtitle;
        } else {
            body = "Semnalul s-a schimbat din " + previousLabel + " în " + label + ".";
        }

        String expanded = "Semnalul s-a schimbat din " + previousLabel + " în " + label + ".";
        if (!attitude.isEmpty()) expanded += "\nAtitudine: " + attitude + ".";
        if (!reason.isEmpty()) expanded += "\nMotiv: " + reason + ".";

        notify(context, title, body, expanded);
    }

    private static void checkStructuralFallback(Context context) {
        try {
            File riskFile = new File(context.getFilesDir(), "risk_window.json");
            if (!riskFile.exists() || riskFile.length() <= 0) return;

            JSONObject risk = new JSONObject(readUtf8(riskFile));
            String currentRegime = clean(risk.optString("current_regime", ""));
            String level = clean(risk.optString("level", ""));
            String signature = currentRegime + "|" + level;

            if (signature.trim().equals("|") || signature.trim().isEmpty()) return;

            SharedPreferences prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
            String lastSignature = prefs.getString(PREF_LAST_STRUCTURAL_SIGNATURE, "");

            if (lastSignature == null || lastSignature.isEmpty()) {
                prefs.edit().putString(PREF_LAST_STRUCTURAL_SIGNATURE, signature).apply();
                return;
            }

            if (lastSignature.equals(signature)) return;

            prefs.edit().putString(PREF_LAST_STRUCTURAL_SIGNATURE, signature).apply();

            notify(
                    context,
                    "BTC: schimbare structurală",
                    currentRegime.isEmpty() ? "Contextul structural s-a modificat." : currentRegime,
                    "Contextul structural BTC s-a modificat.\nRegim: " + currentRegime + "\nNivel: " + level
            );
        } catch (Exception ignored) {
        }
    }

    private static void notify(Context context, String title, String body, String expanded) {
        createChannel(context);

        if (Build.VERSION.SDK_INT >= 33 &&
                context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                        != PackageManager.PERMISSION_GRANTED) {
            return;
        }

        Intent intent = new Intent(context, MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);

        int flags = PendingIntent.FLAG_UPDATE_CURRENT;
        if (Build.VERSION.SDK_INT >= 23) flags |= PendingIntent.FLAG_IMMUTABLE;

        PendingIntent pendingIntent = PendingIntent.getActivity(
                context,
                0,
                intent,
                flags
        );

        android.app.Notification.Builder builder =
                Build.VERSION.SDK_INT >= 26
                        ? new android.app.Notification.Builder(context, CHANNEL_ID)
                        : new android.app.Notification.Builder(context);

        builder.setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle(title)
                .setContentText(body)
                .setStyle(new android.app.Notification.BigTextStyle().bigText(expanded))
                .setContentIntent(pendingIntent)
                .setAutoCancel(true)
                .setShowWhen(true)
                .setWhen(System.currentTimeMillis());

        if (Build.VERSION.SDK_INT < 26) {
            builder.setPriority(android.app.Notification.PRIORITY_HIGH);
        }

        NotificationManager nm =
                (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);

        if (nm != null) {
            nm.notify(NOTIFICATION_ID_TRADER_SIGNAL, builder.build());
        }
    }

    private static void createChannel(Context context) {
        if (context == null || Build.VERSION.SDK_INT < 26) return;

        NotificationManager nm =
                (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);

        if (nm == null) return;

        NotificationChannel existing = nm.getNotificationChannel(CHANNEL_ID);
        if (existing != null) return;

        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_HIGH
        );
        channel.setDescription("Alerte când semnalul BTC se schimbă.");
        nm.createNotificationChannel(channel);
    }

    private static String readUtf8(File file) throws Exception {
        try (FileInputStream in = new FileInputStream(file)) {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] data = new byte[4096];
            int n;
            while ((n = in.read(data)) != -1) {
                buffer.write(data, 0, n);
            }
            return new String(buffer.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    private static String clean(String s) {
        return s == null ? "" : s.trim();
    }
}
