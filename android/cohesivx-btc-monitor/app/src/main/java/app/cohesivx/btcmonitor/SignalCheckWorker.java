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
       