package com.ecemoca.bing.prediction;

import android.Manifest;
import android.app.ActivityManager;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.AsyncTask;
import android.os.Debug;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.InputType;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import com.androidplot.xy.BoundaryMode;
import com.androidplot.xy.LineAndPointFormatter;
import com.androidplot.xy.PanZoom;
import com.androidplot.xy.SimpleXYSeries;
import com.androidplot.xy.XYPlot;
import com.androidplot.xy.XYSeries;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import be.tarsos.dsp.util.fft.FFT;
import be.tarsos.dsp.util.fft.HammingWindow;
import umich.cse.yctung.androidlibsvm.LibSVM;
import com.ecemoca.bing.prediction.camera.*;
import com.ecemoca.bing.prediction.tensorflow.Classifier;
import com.ecemoca.bing.prediction.tensorflow.TensorFlowClassifier;
import com.ecemoca.bing.prediction.ui.InstructionActivity;
import com.ecemoca.bing.prediction.ui.OnSwipeTouchListener;
import com.ecemoca.bing.prediction.ui.SettingsActivity;
import com.google.android.gms.vision.CameraSource;
import com.google.android.gms.vision.Detector;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;
import com.google.android.gms.vision.face.LargestFaceFocusingProcessor;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "PREDICTION";
    private static final int  MY_RECORD_PERMISSION_CODE = 101, MY_STORAGE_PERMISSION_CODE = 102, MY_CAMERA_PERMISSION_CODE = 103;
    private MediaPlayer player = null;
    private AudioRecord record = null;
    private Thread recordingThread = null;
    private int sampleRate = 48000;
    private int minBufferSize = 2 * 100 * sampleRate / 1000;     // 200ms
    private Short[] recordingL = new Short[minBufferSize / 2];
    private Short[] recordingR = new Short[minBufferSize / 2];
    private XYPlot plot = null;
    private LineAndPointFormatter seriesFormatTop, seriesFormatBot, seriesFormatSpec;
    private StringBuilder sb = new StringBuilder();
    private StringBuilder sb_p = new StringBuilder();
    private Switch saveSwitch = null, userSwitch = null, micSwitch;
    private LibSVM svm = new LibSVM();
    private String appFolderPath = Environment.getExternalStorageDirectory() + "/EchoPrint/";
    private String userDataPath = appFolderPath + "userData/";
    private String generatedPath = appFolderPath + "generatedFiles/";
    private String temporaryPredictFile = generatedPath + "temporary_data.txt";
    private int n_predictions = 0, total_prediction = 3, n_thred = 2, discard_samples = 0;
    private CameraSource mCameraSource = null;
    private CameraSourcePreview mPreview = null;
    private GraphicOverlay mGraphicOverlay = null;
    private FaceTracker tracker = null;
    private String userName = null;
    private AlertDialog.Builder builder;
    private ArrayAdapter<String> adapter = null;
    private Spinner spinner = null;
    private List<String> userNames = null;
    private ToggleButton startButton = null;
    private ProgressBar progressBar = null;

    /*****************Tensorflow vars*****************/

    /**
     * INPUT
     import/conv2d_1_input
     import/conv2d_1/convolution


     OUTPUT
     import/activation_6/Softmax
     import/dense_2/BiasAdd
     */
    private static final int[] INPUT_SIZE = {1,32,60,1}; //Size = 1*32*60*1
    private static final String INPUT_NAME = "conv2d_1_input"; //conv2d_1
    private static final String OUTPUT_NAME = "activation_6/Softmax";
    private static final String MODEL_FILE = "file:///android_asset/tf_model.pb";
    private static final String LABEL_FILE = "file:///android_asset/user_list.txt";
    private Classifier classifier;
    private Executor executor = Executors.newSingleThreadExecutor();

    public long outputValue;

    /*****************Tensorflow vars*****************/


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        Boolean mic_set = prefs.getBoolean("mic_switch", false);
        Boolean vision_set = prefs.getBoolean("vision_switch", false);

        saveSwitch = findViewById(R.id.saveTraining);
        userSwitch = findViewById(R.id.userData);
        micSwitch = findViewById(R.id.micSwitch);
        micSwitch.setChecked(mic_set);
        progressBar = findViewById(R.id.progressBar);
        progressBar.setVisibility(View.INVISIBLE);
        builder = new AlertDialog.Builder(this);

        mPreview = findViewById(R.id.preview);
        mGraphicOverlay = findViewById(R.id.faceOverlay);

        // request permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.RECORD_AUDIO}, MY_CAMERA_PERMISSION_CODE);
        }
        else {
            createCameraSource();
        }

        initTensorFlowAndLoadModel();

        // create directory folders for this application
        createDirectory();

        // button registration
        setButtonListener();

        // setup plot
        setPlot();

    }

    /** Restart camera */
    @Override
    protected void onResume() {
        super.onResume();
        startCameraSource();
        if (startButton.isChecked()) {
            startAuthentication();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        mPreview.stop();
        stopAuthentication();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mCameraSource != null) {
            mCameraSource.release();
        }
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
    }

    @Override
    public void onBackPressed() {
        stopRecord();
        playStop();
        this.finish();
    }

    @Override
    public void onSaveInstanceState(Bundle savedInstanceState) {
        super.onSaveInstanceState(savedInstanceState);
        savedInstanceState.putBoolean("phone_selection", micSwitch.isChecked());
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            // Standard Android full-screen functionality.
            getWindow().getDecorView().setSystemUiVisibility(
                    View.SYSTEM_UI_FLAG_LAYOUT_STABLE
//                            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
//                            | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {

        if (grantResults.length != 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "Camera permission granted - initialize the camera source");
            // we have permission, so create the camera source
            createCameraSource();
            return;
        }

    }

    private void setButtonListener() {
        startButton = findViewById(R.id.startButton);
        startButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                // start authentication
                if (startButton.isChecked())
                    startAuthentication();
                else
                    stopAuthentication();
            }
        });

        final Button trainButton = findViewById(R.id.trainButton);
        trainButton.setOnClickListener(new View.OnClickListener() {
            public void onClick (View v) {
                // train the model
                svm_train();
            }
        });
        trainButton.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View view) {
                svm_train_against_all();
                return true;
            }
        });

        // popup dialog to confirm delete
        final AlertDialog.Builder deleteDialog = new AlertDialog.Builder(this).setTitle("Delete User")
                .setMessage("Do you really want to remove this user?")
                .setIcon(android.R.drawable.ic_dialog_alert)
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialogInterface, int i) {
                        deleteUser(userName);
                    }
                })
                .setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialogInterface, int i) {

                    }
                });

        final Button clearButton = findViewById(R.id.clearButton);
        clearButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                // remove data
                deleteDialog.show();
            }
        });

        // popup dialog for user name input
        builder.setTitle("Input user name:");
        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        builder.setView(input);
        builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                userName = input.getText().toString();
                if (!Arrays.asList(getUserList()).contains(userName)) {   // create profile for new user
                    createNewUser(userName);
                }
            }
        });
        builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                dialogInterface.cancel();
            }
        });

        saveSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                if (saveSwitch.isChecked()) {
                    if (input.getParent() != null)
                        ((ViewGroup) input.getParent()).removeView(input);
                    builder.show();
                }
            }
        });

        // Spinner for user selection
        spinner = findViewById(R.id.spinner);
        userNames = getUserList();
        adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, userNames);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                userName = adapterView.getItemAtPosition(i).toString();
                Toast.makeText(getApplicationContext(), userName + " selected.", Toast.LENGTH_SHORT).show();
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });

        // Swipe to instructions
        OnSwipeTouchListener onSwipeTouchListener = new OnSwipeTouchListener(this) {
            @Override
            public void onSwipeLeft() {
                showInstruction();
            }

            @Override
            public void onSwipeRight() {
                showSettings();
            }
        };
        findViewById(android.R.id.content).setOnTouchListener(onSwipeTouchListener);

    }


/**==============================================================================================
 //                                        TensorFlow classifier
 //==============================================================================================*/

    /*******Initialize Tensorflow model********/
    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowClassifier.create(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            INPUT_SIZE,
                            INPUT_NAME,
                            OUTPUT_NAME);


                    Log.d(TAG, "Tensorflow Load Success");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    /*******Call this function to classify************/
    private String tensorflowClassify(float[] data) {

        final List<Classifier.Recognition> results = classifier.recognizeFace(data);

        if (results.size() > 0) {
            String value = results.get(0).getTitle();
            return value.split(":")[1];
        }

        return "nobody detected";
    }


/**==============================================================================================
 //                                        Face Detector
 //==============================================================================================*/
    /**
     * Creates the face detector and associated processing pipeline to support either front facing
     * mode or rear facing mode.  Checks if the detector is ready to use, and displays a low storage
     * warning if it was not possible to download the face library.
     */
    @NonNull
    private FaceDetector createFaceDetector(Context context) {
        // For both front facing and rear facing modes, the detector is initialized to do landmark
        // detection (to find the eyes), classification (to determine if the eyes are open), and
        // tracking.
        //
        // Use of "fast mode" enables faster detection for frontward faces, at the expense of not
        // attempting to detect faces at more varied angles (e.g., faces in profile).  Therefore,
        // faces that are turned too far won't be detected under fast mode.
        //
        // For front facing mode only, the detector will use the "prominent face only" setting,
        // which is optimized for tracking a single relatively large face.  This setting allows the
        // detector to take some shortcuts to make tracking faster, at the expense of not being able
        // to track multiple faces.
        //
        // Setting the minimum face size not only controls how large faces must be in order to be
        // detected, it also affects performance.  Since it takes longer to scan for smaller faces,
        // we increase the minimum face size for the rear facing mode a little bit in order to make
        // tracking faster (at the expense of missing smaller faces).  But this optimization is less
        // important for the front facing case, because when "prominent face only" is enabled, the
        // detector stops scanning for faces after it has found the first (large) face.
        FaceDetector detector = new FaceDetector.Builder(context)
                .setLandmarkType(FaceDetector.ALL_LANDMARKS)
                .setClassificationType(FaceDetector.ALL_CLASSIFICATIONS)
                .setTrackingEnabled(true)
                .setMode(FaceDetector.FAST_MODE)
                .setProminentFaceOnly(true)
                .setMinFaceSize(0.35f)
                .build();

        Detector.Processor<Face> processor;
        // For front facing mode, a single tracker instance is used with an associated focusing
        // processor.  This configuration allows the face detector to take some shortcuts to
        // speed up detection, in that it can quit after finding a single face and can assume
        // that the nextIrisPosition face position is usually relatively close to the last seen
        // face position.
        tracker = new FaceTracker(mGraphicOverlay);
        processor = new LargestFaceFocusingProcessor.Builder(detector, tracker).build();

        detector.setProcessor(processor);

        if (!detector.isOperational()) {
            // Note: The first time that an app using face API is installed on a device, GMS will
            // download a native library to the device in order to do detection.  Usually this
            // completes before the app is run for the first time.  But if that download has not yet
            // completed, then the above call will not detect any faces.
            //
            // isOperational() can be used to check if the required native library is currently
            // available.  The detector will automatically become operational once the library
            // download completes on device.
            Log.w(TAG, "Face detector dependencies are not yet available.");

            // Check for low storage.  If there is low storage, the native library will not be
            // downloaded, so detection will not become operational.
            IntentFilter lowStorageFilter = new IntentFilter(Intent.ACTION_DEVICE_STORAGE_LOW);
            boolean hasLowStorage = registerReceiver(null, lowStorageFilter) != null;

            if (hasLowStorage) {
                Toast.makeText(this, "low storage error", Toast.LENGTH_LONG).show();
                Log.w(TAG, "low storage error");
            }
        }
        return detector;
    }

    /** Start camera source */
    private void createCameraSource() {
        Context context = getApplicationContext();
        FaceDetector detector = createFaceDetector(context);
        int facing = CameraSource.CAMERA_FACING_FRONT;
        mCameraSource = new CameraSource.Builder(context, detector)
                .setFacing(facing)
                .setRequestedPreviewSize(1024, 768)
                .setRequestedFps(30.0f)
                .setAutoFocusEnabled(true)
                .build();
    }

    private void startCameraSource() {
        if (mCameraSource != null) {
            try {
                mPreview.start(mCameraSource, mGraphicOverlay);
            } catch (IOException e) {
                Log.e(TAG, "unable to start camera", e);
                mCameraSource.release();
                mCameraSource = null;
            }
        }
    }

/**=============================================================================================
 //                              Acoustic Sensing and Authentication
 //============================================================================================*/
    /** authentication control functions */
    public void startAuthentication() {
        play();
        startRecord();
    }

    public void stopAuthentication() {
        stopRecord();
        playStop();
        saveTrainingData(sb.toString(), userName);
        Log.d("data to save:", sb.toString());
    }

    /** sound emitting functions */
    private void play() {
        player = MediaPlayer.create(getApplicationContext(), R.raw.fmcw);
        player.setLooping(true);
        player.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
            @Override
            public void onPrepared(MediaPlayer mediaPlayer) {
                mediaPlayer.start();
            }
        });
    }

    private void playStop() {
        if (player != null) {
            player.release();
            player = null;
        }
    }

    /** sound recording functions */
    private void startRecord() {
        record = new AudioRecord(MediaRecorder.AudioSource.CAMCORDER,
                48000, AudioFormat.CHANNEL_IN_STEREO,
                AudioFormat.ENCODING_PCM_16BIT, 1024 *4);
        record.startRecording();



        recordingThread = new Thread(new Runnable() {
            @Override
            public void run() {
                // record 200ms samples
                try {
                    //Recording CPU performance - Start recording

                    short[] buffer = new short[minBufferSize];
                    while (record != null && record.read(buffer, 0, minBufferSize) > 0) {


                        for (int i = 0; i < minBufferSize / 2; i++) {
                            recordingR[i] = buffer[2 * i];
                            recordingL[i] = buffer[2 * i + 1];
                        }

                        // do prediction here
                        float[] top = highPassFilter(recordingL);
                        float[] bot = highPassFilter(recordingR);
                        float[] xTop;
                        if (micSwitch.isChecked()) {     // S8
                            xTop = getSegment(bot);
                        } else {     //P9
                            xTop = getSegment(top);
                        }

                        String input_data = "";
                        if (xTop[0] != 0 && xTop[0] == xTop[0]) {  // remove "NaN"
                            float[] spec = getSpectrum(xTop);    // spectrum and plot

//                            float[][] spectrogram = getSpectrogram(xTop);

                            updatePlot(xTop, getFaceSegment(xTop), spec);

                            // Check if face is detected and aligned
                            if (tracker == null) continue;
                            float[] landmarks = tracker.getLandmarks();
                            if (landmarks == null) continue;

                            String topStr = Arrays.toString(xTop);
                            topStr = topStr.substring(1, topStr.length()-1);
                            String[] strs = topStr.split(" ");

                            if (!userSwitch.isChecked() && landmarks[0] != 0)
                                topStr = "1";
                            else
                                topStr = "0";

                            // attach face landmarks from 1 - 16
                            for (int i = 1; i <= 16; i++) {
                                topStr = topStr + " " + i + ":" + landmarks[i-1];
                                input_data = input_data + " " + i + ":" + landmarks[i-1];
                            }
                            topStr += " "; input_data += " ";

                            for(int i = 17; i <= strs.length - discard_samples + 16; i++) {
                                topStr = topStr + i + ":" + strs[i-17+discard_samples];
                                input_data = input_data + i + ":" + strs[i-17+discard_samples];
                            }

                            if (saveSwitch.isChecked()) {
                                sb.append(topStr);
                                sb.append('\n');
                            }
                            sb_p.append(topStr);
                            sb_p.append('\n');
                            n_predictions++;
                            if (n_predictions >= total_prediction) {
                                savePredictData(sb_p.toString(), temporaryPredictFile);
                                sb_p.setLength(0);
                                n_predictions = 0;
                            }
                        }

                        // run prediction
                        if (!saveSwitch.isChecked()) {
//                            final boolean flag = svm_predict();

                            /**** Generate known input for "bing", the result should be "bing" by using this input
String input = "17:0.0037046177 18:1.5014871E-4 19:-0.00669451 20:0.0153367985 21:-0.014249541 22:-0.015166708 23:0.06087198 24:-0.06469111 25:-0.024191525 26:0.1545569 27:-0.16294833 28:-0.051675152 29:0.32782134 30:-0.33286628 31:-0.055050068 32:0.51606 33:-0.56488967 34:0.091241635 35:0.48239243 36:-0.66184735 37:0.3850391 38:0.0702576 39:-0.4789288 40:0.65061194 41:-0.32817292 42:-0.4007625 43:0.94414467 44:-0.8394409 45:0.19231215 46:0.5642847 47:-1.0 48:0.89389735 49:-0.36269125 50:-0.24682838 51:0.6325234 52:-0.70186424 53:0.5301499 54:-0.25289875 55:0.0031014476 56:0.1511827 57:-0.22880255 58:0.2598032 59:-0.24803975 60:0.20324038 61:-0.1323724 62:0.04275664 63:0.024657551 64:-0.03565472 65:0.010049924 66:0.016811226 67:-0.026243709 68:0.0037608047 69:0.042267546 70:-0.0667268 71:0.044878155 72:-0.0062352885 73:-0.017678127 74:0.026635928 75:-0.024286494 76:0.009708704 77:0.0055360054 78:-0.008967411 79:0.0063474295 80:-0.007647145 81:0.010438645 82:-0.009891819 83:0.0071717226 84:-0.00346182 85:5.509681E-4 86:-0.0027254215 87:0.00829695 88:-0.009608453 89:0.0057142465 90:-0.004147095 91:0.008584178 92:-0.012051302 93:0.0069470494 94:0.0022410438 95:-0.006342568 96:0.004581356 97:-0.009629045 98:-0.014028888 99:0.01129593 100:-0.011069747 101:0.038305357 102:-0.03212578 103:-0.08984512 104:0.26837578 105:-0.29767329 106:0.067927085 107:0.22609484 108:-0.2706666 109:0.03520507 110:0.2091859 111:-0.24859248 112:0.10041452 113:0.108361915 114:-0.2626 115:0.26979017 116:-0.085787356 117:-0.18861964 118:0.33041644 119:-0.21060753 120:-0.082574114 121:0.3617789 122:-0.48548836 123:0.3992571 124:-0.1608076 125:-0.074086994 126:0.18801726 127:-0.19567499 128:0.14773387 129:-0.038405105 130:-0.11694711 131:0.22755843 132:-0.21691708 133:0.08801702 134:0.07416804 135:-0.13977541 136:0.05791943 137:0.07271072 138:-0.117206804 139:0.03995471 140:0.09402998 141:-0.18269843 142:0.15543883 143:-0.046342555 144:-0.01864179 145:-0.023948286 146:0.09045381 147:-0.10509147 148:0.10247013 149:-0.09580139 150:0.036056377 151:0.05237554 152:-0.07786543 153:0.04058747 154:-0.014716478 155:-0.0075083002 156:0.059027918 157:-0.06103987 158:-0.040707037 159:0.12363346 160:-0.08945772 161:0.021974696 162:0.018619118 163:-0.06487122 164:0.06856755 165:0.018807095 166:-0.08670895 167:0.055842407 168:-0.029405385 169:0.06241492 170:-0.06957551 171:0.044936635 172:-0.05686269 173:0.08754409 174:-0.09833117 175:0.10857995 176:-0.12243585 177:0.09892043 178:-0.012629347 179:-0.11322887 180:0.19700679 181:-0.1414251 182:-0.024446033 183:0.14715308 184:-0.138806 185:0.036158223 186:0.092382275 187:-0.16603573 188:0.09756885 189:0.06556304 190:-0.14373907 191:0.095805295 192:-0.047878645 193:0.02052151 194:0.048983205 195:-0.08904441 196:0.018731538";
                            String[] input_strs = input.split(" ");
                            float[] xTopTest = new float[180];
                            for (int i = 0; i < 180; i++)
                                xTopTest[i] = Float.valueOf(input_strs[i].split(":")[1]);
                            *********/

                            float[][] spectrogram = getSpectrogram(xTop);     // use xTopTest for testing

                            /** tensorflow classification **/

                            float[] spec = new float[32*60];
                            String msg = "";
                            for (int i = 0; i < 32; ++i){
                                for (int j = 0; j < 60; j++) {
                                    spec[i * 32 + j] = spectrogram[i][j];
                                    msg = msg + " " + spectrogram[i][j];
                                }
                            }

                            Log.i("specmsg", msg);
                            Log.i("specmsg", "" + spectrogram[31][59]);


                            //Start monitoring code
                            long start = Debug.threadCpuTimeNanos();
                            int time = (int) (System.currentTimeMillis());


                            final String name = tensorflowClassify(spec);

                            //End monitoring code
                            int time2 = (int) (System.currentTimeMillis());
                            String ts2 =  Integer.toString(time2 - time);
                            Log.i("Time taken", ts2);

                            ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
                            ActivityManager activityManager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
                            activityManager.getMemoryInfo(mi);
                            double availableMegs = mi.availMem / 0x100000L;


//Percentage can be calculated for API 16+
                            double percentAvail = mi.availMem / (double)mi.totalMem * 100.0;

                            long finish = Debug.threadCpuTimeNanos();
                            long outputValue = finish - start ;
                            Log.i("Recorded CPU Thread time", String.valueOf(outputValue));


                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    TextView textView = findViewById(R.id.text_result);
                                    textView.setTextColor(Color.GREEN);
                                    textView.setText(name);
//                                    if (flag) {
//                                        textView.setTextColor(Color.GREEN);
//                                        textView.setText("PASSED!");
//                                    }
//                                    else {
//                                        textView.setTextColor(Color.RED);
//                                        textView.setText("DENIED!");
//                                    }
                                }
                            });
                        }


                    }




                } finally {
//                    if(record != null) {
//                        record.stop();
//                        record.release();
//                        record = null;
//                    }
                }
            }
        }, "AudioRecorder Thread");






        recordingThread.start();
    }

    private void stopRecord() {
        if(record != null) {
            record.stop();
            record.release();
            record = null;
            if (recordingThread.isAlive())
                try {
                    recordingThread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
        }
    }

/**==============================================================================================
 //                                Acoustic Signal Processing
 //==============================================================================================*/
    /** signal processing functions */
    private float[] highPassFilter(Short[] x) {
        float[] a = {1,0.651471653594408f,0.620472122486498f,0.147379460605609f,0.026168663922855f};
        float[] b = {0.052986854513083f,-0.211947418052334f,0.317921127078501f,-0.211947418052334f,0.052986854513083f};
        float[] y = new float[x.length];
        y[0] = b[0] * x[0];
        for (int i = 1; i < a.length; i++) {
            y[i] = b[0] * x[i];
            for (int j = 1; j <= i; j++) {
                y[i] += b[j] * x[i - j] - a[j] * y[i - j];
            }
        }
        for (int i = a.length; i < x.length - 1; i++) {
            y[i] = b[0] * x[i];
            for (int j = 1; j < a.length; j++) {
                y[i] += b[j] * x[i - j] - a[j] * y[i - j];
            }
        }
        y[y.length - 1] = 0;
        return y;
    }

    private float[] getSegment(float[] x) {
        float[] result;
        float MAX = Float.MIN_VALUE;
        int index = 30;
        for (int i = 30; i < x.length/2 + 30; i++) {
            if (Math.abs(x[i]) >= MAX) {
                MAX = Math.abs(x[i]);
                index = i;
            }
        }
        result = Arrays.copyOfRange(x, index-30, index+150);
        for (int i = 0; i < result.length; i++)
            result[i] = result[i] / Math.abs(x[index]);
        // amplify the signal for face
        for (int i = 80; i < result.length; i++)
            result[i] = result[i] * 10;
        return result;
    }

    private float[] getFaceSegment(float[] segment) {
        int index = 60;
        float[] ans = new float[segment.length];
        float[] ref = Arrays.copyOfRange(segment, 20, 40);
        float MAX = Float.MIN_VALUE;
        for (int i = 60; i < segment.length - ref.length - 10; i++) {
            float sum = 0;
            for (int j = 0; j < ref.length; j++) {
                sum += segment[i+j] * ref[j];
            }
            if (sum >= MAX) {
                MAX = sum;
                index = i;
            }
        }

        for (int i = 0; i < segment.length; i++) {
            ans[i] = -10f;
            if (i >= index - 10 && i < index + 30)
                ans[i] = segment[i];
        }

        return ans;
    }

    private float[][] getSpectrogram(float[] segment) {
        int  N_fft = 64;
        int hop_length = 3;
        float[] fftSeg = new float[N_fft];
        FFT fft = new FFT(N_fft, new HammingWindow());
        float[] amplitudes = new float[N_fft/2];
        float[][] spec = new float[N_fft/2][60];
        float maxSpec = Float.MIN_VALUE;

        for (int i = 0; i < segment.length; i = i+hop_length) {
            // padding the signal using reflection
            for (int id = i -32; id < i + 32; id++) {
                if (id < 0)
                    fftSeg[id - i + 32] = segment[2*i - id - 1];
                else if (id >= segment.length)
                    fftSeg[id - i + 32] = segment[2*i - id + 1];
                else
                    fftSeg[id - i + 32] = segment[id];
            }

            fft.forwardTransform(fftSeg);
            fft.modulus(fftSeg, amplitudes);

            // normalize
            for (int j = 0; j < amplitudes.length; j++) {
//                amplitudes[j] = - (float) (20 * Math.log10(amplitudes[j]));
                spec[j][i/3] = amplitudes[j];
                if (amplitudes[j] > maxSpec)
                    maxSpec = amplitudes[j];
            }

        }
        // normalize ans
        for (int i = 0; i < spec.length; i++) {
            for (int j = 0; j < spec[0].length; j++) {
                spec[i][j] = spec[i][j] / maxSpec;
            }
        }

        return spec;
    }

    private float[] getSpectrum(float[] segment) {
        int  N_fft = 64;
        float[] ans = new float[segment.length];
        FFT fft = new FFT(N_fft, new HammingWindow());
        float[] amplitudes = new float[N_fft/2];

        String msg = "";
        float maxAns = Float.MIN_VALUE;
        for (int i = 0; i < segment.length - N_fft; i++) {
            int maxIndex = -10;
            float maxValue = Float.MIN_VALUE;
            float[] fftSeg = Arrays.copyOfRange(segment, i, i+64);
            fft.forwardTransform(fftSeg);
            fft.modulus(fftSeg, amplitudes);

            for (int j = 0; j < amplitudes.length; j++) {
                amplitudes[j] = (float) (20 * Math.log10(amplitudes[j]));
            }

            for (int j = 0; j < amplitudes.length; j++) {
                if (amplitudes[j] > maxValue) {
                    maxValue = amplitudes[j];
                    maxIndex = j;
                }
            }
            ans[i + N_fft / 2] = maxValue;
            if (maxValue > maxAns)
                maxAns = maxValue;
//            msg += " " + maxValue;
        }
        // normalize ans
        for (int i = 0; i < ans.length; i++) {
            ans[i] = ans[i] / maxAns * 2 - 1;
        }
//        Log.i("fft", msg);
        return ans;
    }

/**==============================================================================================
 //                              Data Saving and Figure Plotting
 //==============================================================================================*/
    /** figure plotting functions */
    private void setPlot() {
        plot = findViewById(R.id.plot);
        plot.getLegend().setVisible(true);
        plot.setRangeBoundaries(-1, BoundaryMode.FIXED, 1, BoundaryMode.FIXED);
        plot.setDomainBoundaries(0, BoundaryMode.FIXED, 180, BoundaryMode.FIXED);
        seriesFormatTop = new LineAndPointFormatter(Color.RED, null, null, null);   // BOT microphone
        seriesFormatBot = new LineAndPointFormatter(Color.BLUE, null, null, null);  // TOP microphone
        seriesFormatSpec = new LineAndPointFormatter(Color.LTGRAY, null, null, null);
        PanZoom.attach(plot);
    }

    private void updatePlot(float[] top, float[] face, float[] spec) {
        plot.clear();
        List<Float> soundBufTop = new ArrayList<>();
        List<Float> soundBufFace = new ArrayList<>();
        List<Float> soundSpec = new ArrayList<>();
        for (float f : top)
            soundBufTop.add(f);

        for (float f : face)
            soundBufFace.add(f);

        for (float f : spec)
            soundSpec.add(f);

        XYSeries seriesTop = new SimpleXYSeries(soundBufTop, SimpleXYSeries.ArrayFormat.Y_VALS_ONLY, "Top");
        XYSeries seriesFace = new SimpleXYSeries(soundBufFace, SimpleXYSeries.ArrayFormat.Y_VALS_ONLY, "Face");
        XYSeries seriesSpec = new SimpleXYSeries(soundSpec, SimpleXYSeries.ArrayFormat.Y_VALS_ONLY, "Spec");

        plot.addSeries(seriesSpec, seriesFormatSpec);
        plot.addSeries(seriesTop, seriesFormatTop);
        plot.addSeries(seriesFace, seriesFormatBot);
        plot.redraw();
    }

    /** save training data */
    private void saveTrainingData(String data, String filename) {
        if (!saveSwitch.isChecked() || data == null || data.length() == 0)
            return;

        FileWriter fw = null;
        BufferedWriter bw = null;
        PrintWriter out = null;

        String fileToWrite = userDataPath + filename + ".txt";
        data = data.replace(',', ' ');
        try {
            fw = new FileWriter(fileToWrite, true);
            bw = new BufferedWriter(fw);
            out = new PrintWriter(bw);
            out.write(data);
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (out != null) {
                out.close();
            }
            if (bw != null) {
                try {
                    bw.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (fw != null) {
                try {
                    fw.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    /** save predict data */
    private void savePredictData(String data, String filename) {
        if (saveSwitch.isChecked() || data == null || data.length() == 0)
            return;

        FileWriter fw = null;
        BufferedWriter bw = null;

        data = data.replace(',', ' ');
        try {
            fw = new FileWriter(filename, false);
            bw = new BufferedWriter(fw);
            bw.write(data);
            bw.close();
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (bw != null) {
                try {
                    bw.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (fw != null) {
                try {
                    fw.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private List getUserList() {
        List<String> list = new ArrayList<>();
        File folder = new File(userDataPath);
        if (folder.exists()) {
            String[] ans = folder.list();
            if (ans != null)
                for (int i = 0; i < ans.length; i++)
                    list.add(ans[i].split("\\.")[0]);
            return list;
        }
        return null;
    }

    private void createNewUser(String userName) {
        String filepath = userDataPath + userName + ".txt";
        File file = new File(filepath);
        Boolean successed = false;
        try {
            successed = file.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (successed)
            updateSpinnerList(userName);
    }

    private void deleteUser(String userName) {
        // remove training data
        String filepath = userDataPath + userName + ".txt";
        File file = new File(filepath);
        if (file.delete()) {
            userNames.remove(userName);
            adapter.notifyDataSetChanged();
        }
        // remove trained model
        filepath = generatedPath + userName + "_model";
        file = new File(filepath);
        if (file.exists()) {
            file.delete();
        }
        filepath = generatedPath + userName + "_mixed.txt";
        file = new File(filepath);
        if (file.exists()) {
            file.delete();
        }
        Toast.makeText(getApplicationContext(), userName + " profile is removed.", Toast.LENGTH_SHORT).show();
    }

    private void updateSpinnerList(String newUser) {
        userNames.add(newUser);
        adapter.notifyDataSetChanged();
        spinner.setSelection(adapter.getPosition(newUser));
    }

    private void createDirectory() {
        File dir = new File(appFolderPath);
        if (!dir.exists())
            dir.mkdir();
        dir = new File(generatedPath);
        if (!dir.exists())
            dir.mkdir();
        dir = new File(userDataPath);
        if (!dir.exists())
            dir.mkdir();
    }

/**==============================================================================================
 //                                    SVM Classification
 //==============================================================================================*/
    /** libsvm library functions */
    private void svm_train() {
        File file = new File(userDataPath + userName + ".txt");
        if (file.exists()) {
            new AsyncTrainTask().execute("-t 0", userDataPath + userName + ".txt", generatedPath + userName + "_model");
            Toast.makeText(getApplicationContext(), "Executing quick svm-train, please wait...", Toast.LENGTH_SHORT).show();
        }
        else {
            Toast.makeText(getApplicationContext(), "No training data found...", Toast.LENGTH_SHORT).show();
        }
    }

    private void svm_train_against_all() {
        progressBar.setVisibility(View.VISIBLE);
        if (userName != null)
            try {
                generateTrainData(userName);
            } catch (IOException e) {
                e.printStackTrace();
            }
        File file = new File(generatedPath + userName + "_mixed.txt");
        if (file.exists()) {
            new AsyncTrainTask().execute("-t 0", generatedPath + userName + "_mixed.txt", generatedPath + userName + "_model");
            Toast.makeText(getApplicationContext(), "Executing slow one vs all svm-train, please wait...", Toast.LENGTH_LONG).show();
        }
        else {
            Toast.makeText(getApplicationContext(), "No training data found...", Toast.LENGTH_SHORT).show();
            progressBar.setVisibility(View.INVISIBLE);
        }
    }

    private boolean svm_predict() {
        File model = new File(generatedPath + userName + "_model");
        if (!model.exists()) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(getApplicationContext(), "Please train a model first!", Toast.LENGTH_SHORT).show();
                    startButton.performClick();
                }
            });
            return false;
        }
        File predict = new File(temporaryPredictFile);
        if (!predict.exists()) {
            return false;
        }

        int numberPassed = 0;
        svm.predict(temporaryPredictFile + " " + generatedPath + userName + "_model " + generatedPath + "temporary_result.txt");
        FileReader fr = null;
        try {
            fr = new FileReader(generatedPath + "temporary_result.txt");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        if (fr == null)
            return false;

        BufferedReader br = new BufferedReader(fr);
        String line = "";
        try {
            while ((line = br.readLine()) != null)
                numberPassed += Integer.valueOf(line);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return numberPassed >= n_thred;
    }

    private class AsyncTrainTask extends AsyncTask<String, Void, Void>
    {
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
//            Toast.makeText(getApplicationContext(), "Executing svm-train, please wait...", Toast.LENGTH_SHORT).show();
            Log.d("svm_train", "==================\nStart of SVM TRAIN\n==================");
        }

        @Override
        protected Void doInBackground(String... params) {
            LibSVM.getInstance().train(TextUtils.join(" ", params));
            return null;
        }

        @Override
        protected void onPostExecute(Void result) {
            progressBar.setVisibility(View.INVISIBLE);
            Toast.makeText(getApplicationContext(), "svm-train has executed successfully!", Toast.LENGTH_SHORT).show();
            Log.d("svm_train", "==================\nEnd of SVM TRAIN\n==================");
        }
    }

    private void generateTrainData(String username) throws IOException{
        PrintWriter pw = new PrintWriter(generatedPath + username + "_mixed.txt");
        File allUsersDir = new File(userDataPath);
        String[] allUsers = allUsersDir.list();
        for (String user : allUsers) {
            File file = new File(userDataPath + user);
            BufferedReader br = new BufferedReader(new FileReader(file));
            if (user.equals(username + ".txt")) {
                String line = br.readLine();
                while (line != null) {
                    pw.println(line);
                    line = br.readLine();
                }
            }
            else {
                String line = br.readLine();
                String newLine = "0" + line.substring(1);
                while (line != null) {
                    pw.println(newLine);
                    line = br.readLine();
                    if (line != null)
                        newLine = "0" + line.substring(1);
                }
            }
        }
        pw.flush();
        pw.close();
    }

/**==============================================================================================
 //                       Activity Switches: Instructions, Settings
 //==============================================================================================*/
    /**Swipe to the left to show instructions*/
    private void showInstruction() {
        Intent intent = new Intent(this, InstructionActivity.class);
        startActivity(intent);
    }

    private void showSettings() {
        Intent intent = new Intent(this, SettingsActivity.class);
        startActivity(intent);
    }

}

