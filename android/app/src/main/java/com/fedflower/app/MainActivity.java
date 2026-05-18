package com.fedflower.app;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int REQ_CAMERA   = 1001;
    private static final int REQ_GALLERY  = 1002;
    private static final int REQ_CAM_PERM = 1003;

    // Must stay below this to show "not recognized" message
    private static final float CONFIDENCE_THRESHOLD = 0.35f;

    private Module model;
    private final List<String> flowerNames = new ArrayList<>();
    private Uri cameraImageUri;

    private ImageView ivFlower;
    private Button btnCamera, btnGallery;
    private ProgressBar progressBar;
    private CardView layoutResults;
    private TextView tvResult, tvConfidence, tvInference;

    // ──────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ──────────────────────────────────────────────────────────────────────────

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ivFlower      = findViewById(R.id.iv_flower);
        btnCamera     = findViewById(R.id.btn_camera);
        btnGallery    = findViewById(R.id.btn_gallery);
        progressBar   = findViewById(R.id.progress_bar);
        layoutResults = findViewById(R.id.layout_results);
        tvResult      = findViewById(R.id.tv_result);
        tvConfidence  = findViewById(R.id.tv_confidence);
        tvInference   = findViewById(R.id.tv_inference);

        loadModelAndNames();

        btnCamera.setOnClickListener(v -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(
                        this, new String[]{Manifest.permission.CAMERA}, REQ_CAM_PERM);
            } else {
                launchCamera();
            }
        });

        btnGallery.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK,
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            intent.setType("image/*");
            startActivityForResult(intent, REQ_GALLERY);
        });
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Model & label loading (background thread — happens once on startup)
    // ──────────────────────────────────────────────────────────────────────────

    private void loadModelAndNames() {
        new Thread(() -> {
            try {
                model = Module.load(assetFilePath("flower_traced.pt"));

                try (InputStream is = getAssets().open("flower_names.txt")) {
                    byte[] buf = new byte[is.available()];
                    //noinspection ResultOfMethodCallIgnored
                    is.read(buf);
                    for (String line : new String(buf, "UTF-8").trim().split("\\n")) {
                        String name = line.trim();
                        if (!name.isEmpty()) flowerNames.add(name);
                    }
                }
            } catch (IOException e) {
                runOnUiThread(() -> {
                    tvResult.setText("Failed to load model: " + e.getMessage());
                    layoutResults.setVisibility(View.VISIBLE);
                });
            }
        }).start();
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Camera & gallery
    // ──────────────────────────────────────────────────────────────────────────

    private void launchCamera() {
        File photoFile = new File(getExternalCacheDir(), "flower_photo.jpg");
        cameraImageUri = FileProvider.getUriForFile(
                this, getPackageName() + ".fileprovider", photoFile);
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, cameraImageUri);
        startActivityForResult(intent, REQ_CAMERA);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK) return;

        Bitmap bitmap = null;
        try {
            if (requestCode == REQ_CAMERA) {
                bitmap = BitmapFactory.decodeStream(
                        getContentResolver().openInputStream(cameraImageUri));
            } else if (requestCode == REQ_GALLERY && data != null) {
                bitmap = MediaStore.Images.Media.getBitmap(
                        getContentResolver(), data.getData());
            }
        } catch (IOException e) {
            return;
        }

        if (bitmap != null) runInference(bitmap);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Inference
    // ──────────────────────────────────────────────────────────────────────────

    private void runInference(final Bitmap rawBitmap) {
        ivFlower.setImageBitmap(rawBitmap);
        layoutResults.setVisibility(View.GONE);
        progressBar.setVisibility(View.VISIBLE);
        btnCamera.setEnabled(false);
        btnGallery.setEnabled(false);

        new Thread(() -> {
            try {
                // Preprocessing: Resize(256) → CenterCrop(224) → /255 → ImageNet normalize
                Bitmap preprocessed = resizeAndCenterCrop(rawBitmap, 256, 224);
                Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                        preprocessed,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB);

                long startMs = SystemClock.elapsedRealtime();
                Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
                long inferenceMs = SystemClock.elapsedRealtime() - startMs;

                float[] probs = softmax(outputTensor.getDataAsFloatArray());

                // Sort indices by descending probability to get top-3
                Integer[] indices = new Integer[probs.length];
                for (int i = 0; i < indices.length; i++) indices[i] = i;
                Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));

                final float   top1Conf = probs[indices[0]];
                final int[]   top3Idx  = {indices[0], indices[1], indices[2]};
                final float[] top3Prob = {probs[indices[0]], probs[indices[1]], probs[indices[2]]};
                final long    ms       = inferenceMs;

                runOnUiThread(() -> showResults(top1Conf, top3Idx, top3Prob, ms));

            } catch (Exception e) {
                runOnUiThread(() -> {
                    tvResult.setText("Inference error: " + e.getMessage());
                    tvConfidence.setText("");
                    layoutResults.setVisibility(View.VISIBLE);
                    progressBar.setVisibility(View.GONE);
                    btnCamera.setEnabled(true);
                    btnGallery.setEnabled(true);
                });
            }
        }).start();
    }

    private void showResults(float top1Conf, int[] top3Idx, float[] top3Prob, long inferenceMs) {
        progressBar.setVisibility(View.GONE);
        layoutResults.setVisibility(View.VISIBLE);
        btnCamera.setEnabled(true);
        btnGallery.setEnabled(true);

        if (top1Conf < CONFIDENCE_THRESHOLD) {
            tvResult.setText("Flower not recognized.");
            tvConfidence.setText("Try a closer, well-lit photo.");
        } else {
            tvResult.setText(labelName(top3Idx[0]));

            StringBuilder sb = new StringBuilder();
            sb.append(String.format("Confidence: %.1f%%\n", top3Prob[0] * 100));
            sb.append(String.format("2nd: %s (%.1f%%)\n", labelName(top3Idx[1]), top3Prob[1] * 100));
            sb.append(String.format("3rd: %s (%.1f%%)",   labelName(top3Idx[2]), top3Prob[2] * 100));
            tvConfidence.setText(sb.toString());
        }

        tvInference.setText(String.format("⚡ %d ms on-device", inferenceMs));
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────────

    /** Resize so the shorter side = targetShort, then center-crop to cropSize × cropSize. */
    private Bitmap resizeAndCenterCrop(Bitmap src, int targetShort, int cropSize) {
        int w = src.getWidth(), h = src.getHeight();
        float scale = (w < h) ? (float) targetShort / w : (float) targetShort / h;
        int newW = Math.round(w * scale);
        int newH = Math.round(h * scale);
        Bitmap resized = Bitmap.createScaledBitmap(src, newW, newH, true);
        int x = (newW - cropSize) / 2;
        int y = (newH - cropSize) / 2;
        return Bitmap.createBitmap(resized, x, y, cropSize, cropSize);
    }

    private float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        float sum = 0;
        float[] exp = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < exp.length; i++) exp[i] /= sum;
        return exp;
    }

    private String labelName(int idx) {
        return (idx >= 0 && idx < flowerNames.size())
                ? flowerNames.get(idx)
                : "Class " + (idx + 1);
    }

    /** Copy an asset file to internal storage (required by PyTorch Mobile's Module.load). */
    private String assetFilePath(String assetName) throws IOException {
        File file = new File(getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) return file.getAbsolutePath();
        try (InputStream is = getAssets().open(assetName);
             FileOutputStream os = new FileOutputStream(file)) {
            byte[] buf = new byte[8 * 1024];
            int read;
            while ((read = is.read(buf)) != -1) os.write(buf, 0, read);
            os.flush();
        }
        return file.getAbsolutePath();
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] results) {
        super.onRequestPermissionsResult(code, perms, results);
        if (code == REQ_CAM_PERM
                && results.length > 0
                && results[0] == PackageManager.PERMISSION_GRANTED) {
            launchCamera();
        }
    }
}
