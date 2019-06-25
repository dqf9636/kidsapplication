package com.example.opencv_example;

import android.annotation.TargetApi;
import android.app.Activity;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.concurrent.Semaphore;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity
        implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "opencv";
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat matInput;
    private Mat matResult;
    private Mat matTensor;
    private static int VIEW_INDEX = 1; // framelayout index

    public native long loadCascade(String cascadeFileName );
    public native void detect(long cascadeClassifier_face,
                               long matAddrInput, long matAddrResult, long matAddrTensor);
    //public native void convertToBmp(long matAddrTensor, long bmpAddrBmp);
    public long cascadeClassifier_face = 0;

    private final Semaphore writeLock = new Semaphore(1);

    public void getWriteLock() throws InterruptedException {
        writeLock.acquire();
    }

    public void releaseWriteLock() {
        writeLock.release();
    }

    static {
        System.loadLibrary("opencv_java4");
        System.loadLibrary("native-lib");
    }

    private void copyFile(String filename) {
        String baseDir = Environment.getExternalStorageDirectory().getPath();
        String pathDir = baseDir + File.separator + filename;

        AssetManager assetManager = this.getAssets();

        InputStream inputStream = null;
        OutputStream outputStream = null;

        try {
            Log.d( TAG, "copyFile :: 다음 경로로 파일복사 "+ pathDir);
            inputStream = assetManager.open(filename);
            outputStream = new FileOutputStream(pathDir);

            byte[] buffer = new byte[1024];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
            inputStream.close();
            inputStream = null;
            outputStream.flush();
            outputStream.close();
            outputStream = null;
        } catch (Exception e) {
            Log.d(TAG, "copyFile :: 파일 복사 중 예외 발생 "+e.toString() );
        }

    }

    private void read_cascade_file(){
        copyFile("haarcascade_frontface.xml");

        Log.d(TAG, "read_cascade_file:");

        cascadeClassifier_face = loadCascade( "haarcascade_frontface.xml");
        Log.d(TAG, "read_cascade_file:");
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    private void changeView(int index) {
        LinearLayout take_layout = (LinearLayout) findViewById(R.id.take_view);
        LinearLayout result_layout = (LinearLayout) findViewById(R.id.result_view);

        switch (index) {
            case 0:
                take_layout.setVisibility(View.VISIBLE);
                result_layout.setVisibility(View.INVISIBLE);
                VIEW_INDEX = 1;
                break;
            case 1:
                take_layout.setVisibility(View.INVISIBLE);
                result_layout.setVisibility(View.VISIBLE);
                VIEW_INDEX = 0;
                break;
        }
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);


        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            //퍼미션 상태 확인
            if (!hasPermissions(PERMISSIONS)) {

                //퍼미션 허가 안되어있다면 사용자에게 요청
                requestPermissions(PERMISSIONS, PERMISSIONS_REQUEST_CODE);
            }
            else {
                read_cascade_file();
            }
        }
        else {
            read_cascade_file();
        }

        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(1); // front-camera(1),  back-camera(0)
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        Button button = (Button)findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                try {
                    getWriteLock();

                    File path = new File(Environment.getExternalStorageDirectory() + "/Images/");
                    path.mkdirs();
                    File file = new File(path, "tensor.png");

                    String filename = file.toString();

                    Imgproc.cvtColor(matTensor, matTensor, Imgproc.COLOR_BGR2GRAY, 4);
                    boolean ret  = Imgcodecs.imwrite( filename, matTensor);

                    //Mat -> Bitmap 이미지 형식 변환
                    Bitmap bmp = Bitmap.createBitmap(matTensor.cols(), matTensor.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(matTensor, bmp);

                    Intent mediaScanIntent = new Intent( Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
                    mediaScanIntent.setData(Uri.fromFile(file));
                    sendBroadcast(mediaScanIntent);

                    changeView(VIEW_INDEX);

                    //tf 처리
                    ImageView iv = findViewById(R.id.face_photo);
                    iv.setScaleType(ImageView.ScaleType.FIT_XY); // [300,300] 에 꽉 차게 표시
                    iv.setImageBitmap(bmp);

                    // 입력 배열 생성
                    float[][][][] bytes_img = new float[1][48][48][1];

                    for(int y = 0; y<48; y++) {
                        for (int x = 0; x < 48; x++) {
                            int pixel = bmp.getPixel(x, y);
                            bytes_img[0][x][y][0] = (pixel & 0xff) / (float) 255;
                            bytes_img[0][x][y][0] = bytes_img[0][x][y][0] - (float) 0.5;
                            bytes_img[0][x][y][0] = bytes_img[0][x][y][0] * (float) 0.2;
                        }
                    }

                    // 파이썬에서 만든 모델 파일 로딩
                    Interpreter tf_lite = getTfliteInterpreter("converted_model.tflite");
                    Log.d("model", "model pull");

                    // 출력 배열 생성
                    float[][] output = new float[1][7];
                    tf_lite.run(bytes_img, output);

                    Log.d("predict", Arrays.toString(output[0]));

                    //텍스트뷰 2개. 0~9 사이의 숫자 예측
                    int[] id_Array = {R.id.result_1, R.id.result_2, R.id.result_3, R.id.result_4, R.id.result_5, R.id.result_6, R.id.result_7,};
                    String s[] = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};

                    for(int i=0; i<7; i++){
                        TextView tv = findViewById(id_Array[i]);
                        tv.setText(String.format("%s: %f", s[i], output[0][i]));
                        Log.d("result", String.format("%f", output[0][i]));
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                releaseWriteLock();
            }
        });
    }

    // 모델 파일 인터프리터를 생성하는 공통 함수
    // loadModelFile 함수에 예외가 포함되어 있기 때문에 반드시 try, catch 블록이 필요하다
    private Interpreter getTfliteInterpreter(String modelPath){
        try {
            return new Interpreter(loadModelFile(MainActivity.this, modelPath));
        }
        catch (Exception e){
            e.printStackTrace();
        }
        return null;
    }

    // 모델을 읽어오는 함수
    // MappedByteBuffer 바이트 버퍼를 Interpreter 객체에 전달하면 모델 해석을 할 수 있다.
    private MappedByteBuffer loadModelFile(Activity activity, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();

        long startOffset = fileDescriptor.getStartOffset();
        long declaredLegth = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLegth);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "onResume :: Internal OpenCV library not found.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "onResum :: OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        try {
            getWriteLock();
            matInput = inputFrame.rgba();

            if (matResult == null)
                matResult = new Mat(matInput.rows(), matInput.cols(), matInput.type());

            if(matTensor == null)
                matTensor = new Mat(48, 48, matInput.type());

            Core.flip(matInput, matInput, 1);

            detect(cascadeClassifier_face, matInput.getNativeObjAddr(),
                    matResult.getNativeObjAddr(), matTensor.getNativeObjAddr());
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        releaseWriteLock();

        return matResult;
    }



    //여기서부턴 퍼미션 관련 메소드
    //외장 저장소에 파일 저장 위한 퍼미션 추가
    static final int PERMISSIONS_REQUEST_CODE = 1000;
    String[] PERMISSIONS  = {"android.permission.CAMERA",  "android.permission.WRITE_EXTERNAL_STORAGE"};

    private boolean hasPermissions(String[] permissions) {
        int result;

        //스트링 배열에 있는 퍼미션들의 허가 상태 여부 확인
        for (String perms : permissions){

            result = ContextCompat.checkSelfPermission(this, perms);

            if (result == PackageManager.PERMISSION_DENIED){
                //허가 안된 퍼미션 발견
                return false;
            }
        }

        //모든 퍼미션이 허가되었음
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch(requestCode){

            case PERMISSIONS_REQUEST_CODE:
                if (grantResults.length > 0) {
                    boolean cameraPermissionAccepted = grantResults[0]
                            == PackageManager.PERMISSION_GRANTED;

                    //if (!cameraPermissionAccepted)
                    //    showDialogForPermission("앱을 실행하려면 퍼미션을 허가하셔야합니다.");

                    boolean writePermissionAccepted = grantResults[1]
                            == PackageManager.PERMISSION_GRANTED;

                    if (!cameraPermissionAccepted || !writePermissionAccepted) {
                        showDialogForPermission("앱을 실행하려면 퍼미션을 허가하셔야합니다.");
                        return;
                    }else
                    {
                        read_cascade_file();
                    }
                }
                break;
        }
    }

    @TargetApi(Build.VERSION_CODES.M)
    private void showDialogForPermission(String msg) {

        AlertDialog.Builder builder = new AlertDialog.Builder( MainActivity.this);
        builder.setTitle("알림");
        builder.setMessage(msg);
        builder.setCancelable(false);
        builder.setPositiveButton("예", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int id){
                requestPermissions(PERMISSIONS, PERMISSIONS_REQUEST_CODE);
            }
        });
        builder.setNegativeButton("아니오", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface arg0, int arg1) {
                finish();
            }
        });
        builder.create().show();
    }

}