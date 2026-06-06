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

public final class NotificationHelper {
    private static final String CHANNEL_ID = "cohesivx_context";
    private static final String CHANNEL_NAME = "CohesivX context";
    private static final String PREFS = "cohesivx_notifications";
    private static final String KEY_LAST_CONTEXT = "last_context_key";
    private static final int NOTIFICATION_ID = 31401;

    private NotificationHelper() {}

    public static void ensureReady(Activity activity) {
        createChannel(activity);
        if (Build.VERSION.SDK_INT >= 33 &&
                activity.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            activity.requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 314);
        }
    }

    public static void checkContextChangeFromCache(Context context) {
        try {
            File f = new File(context.getFilesDir(), "coeziv_state.json");
            if (!f.exists() || f.length() <= 0) return;
            String json = readText(f);
            JSONObject state = new JSONObject(json);

            String signal = state.optString("signal", "").trim();
            JSONObject regime = state.optJSONObject("market_regime");
            String regimeLabel = regime != null ? regime.optString("label", "").trim() : "";
            JSONObject fg = state.optJSONObject("fg");
            String fgZone = fg != null ? fg.optString("combined_zone", "").trim() : "";

            String key = signal + "|" + regimeLabel + "|" + fgZone;
            if (key.replace("|", "").trim().isEmpty()) return;

            String old = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY_LAST_CONTEXT, "");
            if (old == null || old.isEmpty()) {
                context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY_LAST_CONTEXT, key).apply();
                return;
            }
            if (old.equals(key)) return;

            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY_LAST_CONTEXT, key).apply();

            String title = "CohesivX: context nou detectat";
            String text = buildText(signal, regimeLabel, fgZone);
            notify(context, title, text);
        } catch (Exception ignored) {
        }
    }

    private static String buildText(String signal, String regimeLabel, String fgZone) {
        String s = signal == null ? "" : signal;
        String r = regimeLabel == null ? "" : regimeLabel;
        String f = fgZone == null ? "" : fgZone;
        StringBuilder out = new StringBuilder();
        if (!s.isEmpty()) out.append("Semnal: ").append(s);
        if (!r.isEmpty()) {
            if (out.length() > 0) out.append(" • ");
            out.append(r);
        }
        if (!f.isEmpty()) {
            if (out.length() > 0) out.append(" • ");
            out.append("FG: ").append(f);
        }
        return out.length() == 0 ? "Mecanismul a detectat o schimbare structurală." : out.toString();
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
        channel.setDescription("Notificări când contextul structural CohesivX se schimbă.");
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
}
