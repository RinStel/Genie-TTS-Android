package dev.rinstel.genie_tts.inference

import java.io.File
import java.io.RandomAccessFile
import kotlin.math.roundToInt

object WavFileWriter {
    fun writeMonoPcm16(outputFile: File, samples: FloatArray, sampleRate: Int) {
        outputFile.parentFile?.mkdirs()
        RandomAccessFile(outputFile, "rw").use { file ->
            file.setLength(0)
            val pcmBytes = toPcm16LeBytes(samples)
            val dataSize = pcmBytes.size
            file.writeAscii("RIFF")
            file.writeIntLe(36 + dataSize)
            file.writeAscii("WAVE")
            file.writeAscii("fmt ")
            file.writeIntLe(16)
            file.writeShortLe(1)
            file.writeShortLe(1)
            file.writeIntLe(sampleRate)
            file.writeIntLe(sampleRate * 2)
            file.writeShortLe(2)
            file.writeShortLe(16)
            file.writeAscii("data")
            file.writeIntLe(dataSize)
            file.write(pcmBytes)
        }
    }

    fun toPcm16LeBytes(samples: FloatArray): ByteArray {
        val bytes = ByteArray(samples.size * 2)
        var byteIndex = 0
        for (sample in samples) {
            val pcm = (sample.coerceIn(-1f, 1f) * Short.MAX_VALUE).roundToInt()
            bytes[byteIndex] = (pcm and 0xff).toByte()
            bytes[byteIndex + 1] = ((pcm shr 8) and 0xff).toByte()
            byteIndex += 2
        }
        return bytes
    }

    private fun RandomAccessFile.writeAscii(value: String) {
        write(value.toByteArray(Charsets.US_ASCII))
    }

    private fun RandomAccessFile.writeIntLe(value: Int) {
        write(value and 0xff)
        write((value shr 8) and 0xff)
        write((value shr 16) and 0xff)
        write((value shr 24) and 0xff)
    }

    private fun RandomAccessFile.writeShortLe(value: Int) {
        write(value and 0xff)
        write((value shr 8) and 0xff)
    }
}
