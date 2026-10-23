package com.example.plantdiseaseapp

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.MeteringPointFactory
import androidx.camera.core.Preview as CameraPreviewUseCase
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.plantdiseaseapp.ui.theme.PlantDiseaseAppTheme
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.InputStream
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.Locale
import kotlin.coroutines.resume
import kotlin.math.exp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            PlantDiseaseAppTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Box(modifier = Modifier.padding(innerPadding)) {
                        CameraScreen()
                    }
                }
            }
        }
    }
}

class ONNXClassifier(context: Context) {
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var labels: List<String> = emptyList()
    var initializationError: String? = null
        private set

    init {
        runCatching {
            env = OrtEnvironment.getEnvironment()
            // Read single unified ONNX model directly from assets
            val modelBytes = context.assets.open("crop_disease_model.onnx").readBytes()
            session = env?.createSession(modelBytes, OrtSession.SessionOptions())
            labels = context.assets.open("labels.txt").bufferedReader().readLines()
        }.onFailure { e ->
            initializationError = e.localizedMessage ?: "Failed to load ONNX model"
            e.printStackTrace()
        }
    }

    @Suppress("UNCHECKED_CAST")
    fun classify(bitmap: Bitmap): Pair<String, Float> {
        val currentSession = session
        val currentEnv = env
        if (currentSession == null || currentEnv == null) {
            return Pair(initializationError ?: "Classifier non-functional", 0.0f)
        }

        return try {
            val targetSize = 224
            val resized = Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)
            val inputBuffer = FloatBuffer.allocate(1 * 3 * targetSize * targetSize)

            val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
            val std = floatArrayOf(0.229f, 0.224f, 0.225f)

            val pixels = IntArray(targetSize * targetSize)
            resized.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)

            for (i in 0 until targetSize) {
                for (j in 0 until targetSize) {
                    val pixel = pixels[i * targetSize + j]
                    val r = (((pixel shr 16) and 0xFF) / 255.0f - mean[0]) / std[0]
                    val g = (((pixel shr 8) and 0xFF) / 255.0f - mean[1]) / std[1]
                    val b = ((pixel and 0xFF) / 255.0f - mean[2]) / std[2]

                    inputBuffer.put(0 * targetSize * targetSize + i * targetSize + j, r)
                    inputBuffer.put(1 * targetSize * targetSize + i * targetSize + j, g)
                    inputBuffer.put(2 * targetSize * targetSize + i * targetSize + j, b)
                }
            }
            inputBuffer.rewind()

            val inputName = currentSession.inputNames.iterator().next()
            val shape = longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong())
            val inputTensor = OnnxTensor.createTensor(currentEnv, inputBuffer, shape)

            val results = currentSession.run(mapOf(inputName to inputTensor))
            val rawOutput = (results[0].value as Array<FloatArray>)[0]

            val exps = rawOutput.map { exp(it.toDouble()).toFloat() }
            val sumExps = exps.sum()
            val probs = exps.map { it / sumExps }

            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            Pair(labels.getOrElse(maxIdx) { "Unknown" }, probs[maxIdx] * 100)
        } catch (e: Exception) {
            Pair("Inference error: ${e.message}", 0.0f)
        }
    }

    fun close() {
        runCatching {
            session?.close()
            env?.close()
        }
    }
}

@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    imageCapture: ImageCapture,
    onTapToFocus: (Offset) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember { PreviewView(context) }
    var camera by remember { mutableStateOf<Camera?>(null) }

    LaunchedEffect(Unit) {
        val cameraProvider = context.getCameraProvider()
        val preview = CameraPreviewUseCase.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            cameraProvider.unbindAll()
            camera = cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageCapture
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    AndroidView(
        factory = { previewView },
        modifier = modifier
            .pointerInput(Unit) {
                detectTapGestures { offset ->
                    onTapToFocus(offset)
                    val factory: MeteringPointFactory = SurfaceOrientedMeteringPointFactory(
                        previewView.width.toFloat(),
                        previewView.height.toFloat()
                    )
                    val point = factory.createPoint(offset.x, offset.y)
                    val action = FocusMeteringAction.Builder(point).build()
                    camera?.cameraControl?.startFocusAndMetering(action)
                }
            }
    )
}

private suspend fun Context.getCameraProvider(): ProcessCameraProvider =
    suspendCancellableCoroutine { continuation ->
        ProcessCameraProvider.getInstance(this).also { future ->
            future.addListener({
                continuation.resume(future.get())
            }, ContextCompat.getMainExecutor(this))
        }
    }

@Composable
fun CameraScreen() {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    var hasCameraPermission by remember { mutableStateOf(false) }
    var trackingOffset by remember { mutableStateOf<Offset?>(null) }

    // Captured Inference UI State
    var predictionResult by remember { mutableStateOf<String?>(null) }
    var confidenceResult by remember { mutableStateOf<Float?>(null) }
    var isAnalyzing by remember { mutableStateOf(false) }

    val classifier = remember { ONNXClassifier(context) }
    DisposableEffect(Unit) {
        onDispose { classifier.close() }
    }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
    }

    LaunchedEffect(Unit) {
        launcher.launch(Manifest.permission.CAMERA)
    }

    if (hasCameraPermission) {
        val imageCapture = remember { ImageCapture.Builder().build() }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black),
            verticalArrangement = Arrangement.SpaceBetween,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Top Header Overlay Card
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color.DarkGray.copy(alpha = 0.85f))
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Status:",
                        color = Color.LightGray,
                        fontSize = 12.sp
                    )
                    Text(
                        text = when {
                            isAnalyzing -> "Analyzing captured image..."
                            predictionResult != null -> predictionResult!!
                            else -> "Take a photo to identify plant disease"
                        },
                        color = if (predictionResult != null) Color.Green else Color.White,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                    confidenceResult?.let { conf ->
                        Text(
                            text = "Confidence: ${"%.1f".format(conf)}%",
                            color = Color.Green,
                            fontSize = 14.sp
                        )
                    }
                }
            }

            // Square Camera Viewport Area
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(RoundedCornerShape(16.dp))
                    .border(2.dp, Color.Gray, RoundedCornerShape(16.dp))
            ) {
                CameraPreview(
                    modifier = Modifier.fillMaxSize(),
                    imageCapture = imageCapture,
                    onTapToFocus = { offset -> trackingOffset = offset }
                )

                // Tap focus box indicator
                trackingOffset?.let { offset ->
                    Box(
                        modifier = Modifier
                            .offset {
                                IntOffset(
                                    (offset.x - 30.dp.toPx()).toInt(),
                                    (offset.y - 30.dp.toPx()).toInt()
                                )
                            }
                            .size(60.dp)
                            .border(
                                2.dp,
                                Color.Yellow,
                                RoundedCornerShape(8.dp)
                            )
                    )
                }
            }

            // Bottom Text Button Row
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(32.dp),
                contentAlignment = Alignment.Center
            ) {
                Button(
                    onClick = {
                        isAnalyzing = true
                        takePhotoAndClassify(
                            context = context,
                            imageCapture = imageCapture,
                            classifier = classifier,
                            coroutineScope = coroutineScope,
                            onResult = { label, conf ->
                                isAnalyzing = false
                                predictionResult = label
                                confidenceResult = conf
                            }
                        )
                    },
                    enabled = !isAnalyzing,
                    colors = ButtonDefaults.buttonColors(
                        disabledContainerColor = Color.Gray,
                        disabledContentColor = Color.Black
                    ),
                    modifier = Modifier.fillMaxWidth(0.8f)
                ) {
                    Text(
                        text = if (isAnalyzing) "Capturing..." else "Identify Disease",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(vertical = 4.dp)
                    )
                }
            }
        }
    } else {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Camera Permission Required")
        }
    }
}

private fun takePhotoAndClassify(
    context: Context,
    imageCapture: ImageCapture,
    classifier: ONNXClassifier,
    coroutineScope: kotlinx.coroutines.CoroutineScope,
    onResult: (String, Float) -> Unit
) {
    val name = SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-SSS", Locale.US)
        .format(System.currentTimeMillis())

    val contentValues = ContentValues().apply {
        put(MediaStore.MediaColumns.DISPLAY_NAME, name)
        put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CameraX-Image")
    }

    val outputOptions = ImageCapture.OutputFileOptions.Builder(
        context.contentResolver,
        MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
        contentValues
    ).build()

    imageCapture.takePicture(
        outputOptions,
        ContextCompat.getMainExecutor(context),
        object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                outputFileResults.savedUri?.let { uri ->
                    coroutineScope.launch(Dispatchers.IO) {
                        val inputStream: InputStream? = context.contentResolver.openInputStream(uri)
                        val bitmap = BitmapFactory.decodeStream(inputStream)
                        val (label, confidence) = if (bitmap != null) {
                            classifier.classify(bitmap)
                        } else {
                            Pair("Error loading image", 0.0f)
                        }

                        withContext(Dispatchers.Main) {
                            onResult(label, confidence)
                            Toast.makeText(
                                context,
                                "Disease Identified: $label (${"%.1f".format(confidence)}%)",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }
                }
            }

            override fun onError(exception: ImageCaptureException) {
                Toast.makeText(context, "Error: ${exception.message}", Toast.LENGTH_SHORT).show()
            }
        }
    )
}