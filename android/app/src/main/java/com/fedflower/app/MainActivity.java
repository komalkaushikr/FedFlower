package com.fedflower.app;

import android.Manifest;
import android.app.Activity;
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
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
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
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CAMERA_PERMISSION = 100;
    private static final int REQUEST_CAMERA            = 101;
    private static final int REQUEST_GALLERY           = 102;

    // ImageNet normalization — must exactly match training
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD  = {0.229f, 0.224f, 0.225f};

    private ImageView   ivFlower;
    private Button      btnCamera, btnGallery;
    private TextView    tvResult, tvConfidence, tvInference;
    private CardView    layoutResults;
    private ProgressBar progressBar;

    private Module   torchModel;
    private String[] flowerNames;
    private Uri      cameraImageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ivFlower      = findViewById(R.id.iv_flower);
        btnCamera     = findViewById(R.id.btn_camera);
        btnGallery    = findViewById(R.id.btn_gallery);
        tvResult      = findViewById(R.id.tv_result);
        tvConfidence  = findViewById(R.id.tv_confidence);
        tvInference   = findViewById(R.id.tv_inference);
        layoutResults = findViewById(R.id.layout_results);
        progressBar   = findViewById(R.id.progress_bar);

        loadModel();
        loadFlowerNames();

        btnCamera.setOnClickListener(v -> checkCameraPermissionAndOpen());
        btnGallery.setOnClickListener(v -> openGallery());
    }

    private String assetFilePath(String assetName) throws IOException {
        File file = new File(getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) return file.getAbsolutePath();
        try (InputStream is = getAssets().open(assetName);
             OutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[4096];
            int read;
            while ((read = is.read(buffer)) != -1) os.write(buffer, 0, read);
        }
        return file.getAbsolutePath();
    }

    private void loadModel() {
        new Thread(() -> {
            try {
                torchModel = Module.load(assetFilePath("flower_traced.pt"));
                runOnUiThread(() -> showToast("Model loaded"));
            } catch (Exception e) {
                runOnUiThread(() -> showToast("Error loading model: " + e.getMessage()));
            }
        }).start();
    }

    private void loadFlowerNames() {
        flowerNames = new String[]{
            "Pink Primrose","Hard-leaved Pocket Orchid","Canterbury Bells",
            "Sweet Pea","English Marigold","Tiger Lily","Moon Orchid",
            "Bird of Paradise","Monkshood","Globe Thistle","Snapdragon",
            "Colts Foot","King Protea","Spear Thistle","Yellow Iris",
            "Globe-flower","Purple Coneflower","Peruvian Lily","Balloon Flower",
            "Giant White Arum Lily","Fire Lily","Pincushion Flower","Fritillary",
            "Red Ginger","Grape Hyacinth","Corn Poppy","Prince of Wales Feathers",
            "Stemless Gentian","Artichoke","Sweet William","Carnation",
            "Garden Phlox","Love in the Mist","Cautleya Spicata","Japanese Anemone",
            "Black-eyed Susan","Silverbush","Californian Poppy","Osteospermum",
            "Spring Crocus","Bearded Iris","Windflower","Tree Poppy","Gazania",
            "Azalea","Water Lily","Rose","Thorn Apple","Morning Glory",
            "Passion Flower","Lotus","Toad Lily","Anthurium","Frangipani",
            "Clematis","Hibiscus","Columbine","Desert-rose","Tree Mallow",
            "Magnolia","Cyclamen","Watercress","Canna Lily","Hippeastrum",
            "Bee Balm","Pink Quill","Foxglove","Bougainvillea","Camellia",
            "Mallow","Mexican Petunia","Bromelia","Blanket Flower",
            "Trumpet Creeper","Blackberry Lily","Common Tulip","Wild Pansy",
            "Primula","Sunflower","Pelargonium","Bishop of Llandaff","Gaura",
            "Geranium","Orange Dahlia","Pink-yellow Dahlia","Cautleya Spicata",
            "Japanese Anemone","Buttercup","Oxalis","Water Hyacinth",
            "Prickly Pear","Riverwort","Marigold","Globe Amaranth","Siam Tulip",
            "Lenten Rose","Barbeton Daisy","Daffodil","Sword Lily",
            "Poinsettia","Bolero Deep Blue","Wallflower"
        };
    }

    private void checkCameraPermissionAndOpen() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        } else {
            openCamera();
        }
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        File photoFile = new File(getCacheDir(), "temp_flower.jpg");
        cameraImageUri = FileProvider.getUriForFile(this,
                getPackageName() + ".fileprovider", photoFile);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, cameraImageUri);
        startActivityForResult(intent, REQUEST_CAMERA);
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");
        startActivityForResult(intent, REQUEST_GALLERY);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode,
                                    @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != Activity.RESULT_OK) return;
        try {
            Bitmap bitmap = null;
            if (requestCode == REQUEST_CAMERA) {
                InputStream is = getContentResolver().openInputStream(cameraImageUri);
                bitmap = BitmapFactory.decodeStream(is);
            } else if (requestCode == REQUEST_GALLERY && data != null) {
                InputStream is = getContentResolver().openInputStream(data.getData());
                bitmap = BitmapFactory.decodeStream(is);
            }
            if (bitmap != null) {
                ivFlower.setImageBitmap(bitmap);
                classifyFlower(bitmap);
            }
        } catch (Exception e) {
            showToast("Error loading image: " + e.getMessage());
        }
    }

    /**
     * Preprocess bitmap to match EXACTLY what the model was trained on:
     * 1. Resize shortest side to 256 (preserving aspect ratio)
     * 2. Center crop to 224x224
     * 3. Normalize with ImageNet mean and std
     *
     * This matches Python's:
     * transforms.Resize(256) -> transforms.CenterCrop(224) -> transforms.Normalize(MEAN, STD)
     */
    private Bitmap preprocessBitmap(Bitmap original) {
        int width  = original.getWidth();
        int height = original.getHeight();

        // Step 1: Resize shortest side to 256 preserving aspect ratio
        int newWidth, newHeight;
        if (width < height) {
            newWidth  = 256;
            newHeight = (int) (height * (256.0f / width));
        } else {
            newHeight = 256;
            newWidth  = (int) (width * (256.0f / height));
        }
        Bitmap resized = Bitmap.createScaledBitmap(original, newWidth, newHeight, true);

        // Step 2: Center crop to 224x224
        int startX = (newWidth  - 224) / 2;
        int startY = (newHeight - 224) / 2;
        return Bitmap.createBitmap(resized, startX, startY, 224, 224);
    }

    private void classifyFlower(Bitmap bitmap) {
        if (torchModel == null) {
            showToast("Model still loading, please wait...");
            return;
        }
        progressBar.setVisibility(View.VISIBLE);
        layoutResults.setVisibility(View.GONE);

        new Thread(() -> {
            try {
                // Correct preprocessing matching training pipeline
                Bitmap preprocessed = preprocessBitmap(bitmap);

                // Convert to tensor with ImageNet normalization
                Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                        preprocessed, MEAN, STD);

                long startTime = SystemClock.elapsedRealtime();
                IValue output  = torchModel.forward(IValue.from(inputTensor));
                long timeMs    = SystemClock.elapsedRealtime() - startTime;

                float[] scores = output.toTensor().getDataAsFloatArray();
                float[] probs  = softmax(scores);
                int[]   top3   = topK(probs, 3);

                final String name1 = flowerNames[top3[0]];
                final String name2 = flowerNames[top3[1]];
                final String name3 = flowerNames[top3[2]];
                final float  conf1 = probs[top3[0]] * 100;
                final float  conf2 = probs[top3[1]] * 100;
                final float  conf3 = probs[top3[2]] * 100;

                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    layoutResults.setVisibility(View.VISIBLE);
                    tvResult.setText("🌸 " + name1);
                    tvConfidence.setText(String.format(
                        "Confidence: %.1f%%\n\n2nd: %s (%.1f%%)\n3rd: %s (%.1f%%)",
                        conf1, name2, conf2, name3, conf3));
                    tvInference.setText(String.format(
                        "⚡ %d ms on-device (no internet used)", timeMs));
                });

            } catch (Exception e) {
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    showToast("Inference error: " + e.getMessage());
                });
            }
        }).start();
    }

    private float[] softmax(float[] logits) {
        float max = logits[0];
        for (float v : logits) if (v > max) max = v;
        float[] exp = new float[logits.length];
        float sum = 0;
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < exp.length; i++) exp[i] /= sum;
        return exp;
    }

    private int[] topK(float[] arr, int k) {
        int[] indices  = new int[k];
        boolean[] used = new boolean[arr.length];
        for (int i = 0; i < k; i++) {
            float maxVal = Float.NEGATIVE_INFINITY;
            int maxIdx = 0;
            for (int j = 0; j < arr.length; j++) {
                if (!used[j] && arr[j] > maxVal) { maxVal = arr[j]; maxIdx = j; }
            }
            indices[i] = maxIdx;
            used[maxIdx] = true;
        }
        return indices;
    }

    private void showToast(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (torchModel != null) torchModel.destroy();
    }
}
