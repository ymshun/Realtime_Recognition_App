package com.example.realtimeimagerecognitionapp

import android.content.Context
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class ImageAnalyze(context: Context) : ImageAnalysis.Analyzer {

    private lateinit var listener: OnAnalyzeListener    // Viewを更新するためのカスタムリスナ
    private var lastAnalyzedTimestamp = 0L
    //ネットワークモデルのモデルのロード
    private val resnet = Module.load(getAssetFilePath(context, "resnet.pt"))

    interface OnAnalyzeListener {
        fun getAnalyzeResult(inferredCategory: String, score: Float)
    }

    override fun analyze(image: ImageProxy, rotationDegrees: Int) {
        val currentTimestamp = System.currentTimeMillis()

        if (currentTimestamp - lastAnalyzedTimestamp >= 0.5) {  // 0.5秒ごとに推論
            lastAnalyzedTimestamp = currentTimestamp

            /// テンソルに変換 (imageのformat調べてみたらYUV_420_888とかいうのだった)
            val inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(
                image.image,
                rotationDegrees,
                224,
                224,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )
            /// 学習済みモデルで推論する
            val outputTensor = resnet.forward(IValue.from(inputTensor)).toTensor()
            val scores = outputTensor.dataAsFloatArray

            var maxScore = 0F
            var maxScoreIdx = 0
            for (i in scores.indices) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i]
                    maxScoreIdx = i
                }
            }

            // スコアからカテゴリ名を取得
            val inferredCategory = ImageNetClasses().IMAGENET_CLASSES[maxScoreIdx]
            listener.getAnalyzeResult(inferredCategory, maxScore)  // Viewを更新
        }
    }

    //// assetファイルからパスを取得する関数
    private fun getAssetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
            return file.absolutePath
        }
    }

    fun setOnAnalyzeListener(listener: OnAnalyzeListener){
        this.listener = listener
    }
}