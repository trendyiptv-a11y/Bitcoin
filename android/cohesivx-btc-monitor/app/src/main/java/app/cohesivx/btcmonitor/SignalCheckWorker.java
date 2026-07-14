package app.cohesivx.btcmonitor;

import android.app.AlarmManager;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Build;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class SignalCheckWorker extends BroadcastReceiver {

    private static final String ACTION_CHECK = "app.cohesivx.btcmonitor.ACTION_CHECK_SIGNAL";
    private static final String BASE_URL = "https://coezivx.vercel.app/btc-swing-strategy/";
    private static final long PERIOD_MS = 6L * 60L * 60L * 1000L;

    private static final String[] SNAPSHOT_FILES = new String[]{
            "coeziv_state.json",
            "risk_window.json"
    };

    public static void schedule(Context context) {
        if (context == null) return;

        Context appContext = context.getApplicationContext();
        AlarmManager alarmManager = (AlarmManager) appContext.getSystemService(Context.ALARM_SERVICE);
        if (alarmManager == null) return;

        PendingIntent pendingIntent = buildPendingIntent(appContext);
        long firstRunAt = System.currentTimeMillis() + 60_000L;

        alarmManager.cancel(pendingIntent);
        alarmManager.setInexactRepeating(
                AlarmManager.RTC_WAKEUP,
                firstRunAt,
                PERIOD_MS,
                pendingIntent
        );

        runCheckAsync(appContext);
    }

    @Override
    public void onReceive(Context context, Intent intent) {
        if (context == null) return;

        String action = intent == null ? "" : String.valueOf(intent.getAction());

        if (Intent.ACTION_BOOT_COMPLETED.equals(action) || Intent.ACTION_MY_PACKAGE_REPLACED.equals(action)) {
            schedule(context);
            return;
        }

        PendingResult pendingResult = goAsync();
        Context appContext = context.getApplicationContext();

        new Thread(() -> {
            try {
                runCheck(appContext);
            } finally {
                pendingResult.finish();
            }
        }, "cohesivx-signal-check").start();
    }

    private static PendingIntent buildPendingIntent(Context context) {
        Intent intent = new Intent(context, SignalCheckWorker.class);
        intent.setAction(ACTION_CHECK);

        int flags = PendingIntent.FLAG_UPDATE_CURRENT;
        if (Build.VERSION.SDK_INT >= 23) flags |= PendingIntent.FLAG_IMMUTABLE;

        return PendingIntent.getBroadcast(context, 31415, intent, flags);
    }

    private static void runCheckAsync(Context context) {
        if (context == null) return;
        Context appContext = context.getApplicationContext();

        new Thread(() -> runCheck(appContext), "cohesivx-signal-check-now").start();
    }

    private static void runCheck(Context context) {
        if (context == null) return;

        boolean downloadedAny = false;

        for (String fileName : SNAPSHOT_FILES) {
            downloadedAny = downloadJson(context, BASE_URL + fileName, fileName) || downloadedAny;
        }

        if (downloadedAny) {
            NotificationHelper.checkStructuralChangeFromCache(context);
        }
    }

    private static boolean downloadJson(Context context, String urlText, String fileName) {
        HttpURLConnection conn = null;

        try {
            URL url = new URL(urlText + (urlText.contains("?") ? "&" : "?") + "t=" + System.currentTimeMillis());
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(8000);
            conn.setReadTimeout(8000);
            conn.setUseCaches(false);

            if (conn.getResponseCode() != 200) return false;

            byte[] bytes = readAllBytes(conn.getInputStream());
            if (bytes == null || bytes.length == 0) return false;

            String probe = new String(bytes, StandardCharsets.UTF_8).trim();
            if (!(probe.startsWith("{") || probe.startsWith("["))) return false;

            File out = new File(context.getFilesDir(), fileName);
            try (FileOutputStream fos = new FileOutputStream(out, false)) {
                fos.write(bytes);
            }

            return true;
        } catch (Exception ignored) {
            return false;
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    private static byte[] readAllBytes(InputStream input) throws Exception {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] data = new byte[4096];
        int n;
        while ((n = input.read(data)) != -1) {
            buffer.write(data, 0, n);
        }
        return buffer.toByteArray();
    }
}
