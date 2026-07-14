package app.cohesivx.btcmonitor;

import android.content.Context;

import androidx.annotation.NonNull;
import androidx.work.Constraints;
import androidx.work.ExistingPeriodicWorkPolicy;
import androidx.work.ExistingWorkPolicy;
import androidx.work.NetworkType;
import androidx.work.OneTimeWorkRequest;
import androidx.work.PeriodicWorkRequest;
import androidx.work.WorkManager;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

public class SignalCheckWorker extends Worker {

    private static final String UNIQUE_PERIODIC_WORK = "cohesivx_btc_signal_periodic_check";
    private static final String UNIQUE_IMMEDIATE_WORK = "cohesivx_btc_signal_immediate_check";
    private static final String BASE_URL = "https://coezivx.vercel.app/btc-swing-strategy/";

    private static final String[] SNAPSHOT_FILES = new String[]{
            "coeziv_state.json",
            "risk_window.json"
    };

    public SignalCheckWorker(@NonNull Context context, @NonNull WorkerParameters workerParams) {
        super(context, workerParams);
    }

    @NonNull
    @Override
    public Result doWork() {
        Context context = getApplicationContext();

        try {
            boolean downloadedAny = false;

            for (String fileName : SNAPSHOT_FILES) {
                downloadedAny = downloadJson(context, BASE_URL + fileName, fileName) || downloadedAny;
            }

            if (downloadedAny) {
                NotificationHelper.checkStructuralChangeFromCache(context);
            }

            return Result.success();
        } catch (Exception ignored) {
            return Result.retry();
        }
    }

    public static void schedule(Context context) {
        if (context == null) return;

        Constraints constraints = new Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build();

        PeriodicWorkRequest periodic = new PeriodicWorkRequest.Builder(
                SignalCheckWorker.class,
                6,
                TimeUnit.HOURS,
                30,
                TimeUnit.MINUTES
        )
                .setConstraints(constraints)
                .addTag(UNIQUE_PERIODIC_WORK)
                .build();

        WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                UNIQUE_PERIODIC_WORK,
                ExistingPeriodicWorkPolicy.UPDATE,
                periodic
        );

        OneTimeWorkRequest immediate = new OneTimeWorkRequest.Builder(SignalCheckWorker.class)
                .setConstraints(constraints)
                .addTag(UNIQUE_IMMEDIATE_WORK)
                .build();

        WorkManager.getInstance(context).enqueueUniqueWork(
                UNIQUE_IMMEDIATE_WORK,
                ExistingWorkPolicy.REPLACE,
                immediate
        );
    }

    private boolean downloadJson(Context context, String urlText, String fileName) {
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

    private byte[] readAllBytes(InputStream input) throws Exception {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] data = new byte[4096];
        int n;
        while ((n = input.read(data)) != -1) {
            buffer.write(data, 0, n);
        }
        return buffer.toByteArray();
    }
}
