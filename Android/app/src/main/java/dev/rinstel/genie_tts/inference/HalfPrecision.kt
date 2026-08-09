package dev.rinstel.genie_tts.inference

import java.nio.ByteBuffer
import java.nio.ByteOrder

object HalfPrecision {
    fun fp16BytesToFp32Bytes(fp16Bytes: ByteArray): ByteArray {
        require(fp16Bytes.size % 2 == 0) { "FP16 byte array size must be even." }

        val output = ByteBuffer
            .allocate(fp16Bytes.size * 2)
            .order(ByteOrder.LITTLE_ENDIAN)

        var index = 0
        while (index < fp16Bytes.size) {
            val bits = (fp16Bytes[index].toInt() and 0xFF) or
                ((fp16Bytes[index + 1].toInt() and 0xFF) shl 8)
            output.putFloat(fp16BitsToFloat(bits))
            index += 2
        }

        return output.array()
    }

    fun fp16BitsToFloat(bits: Int): Float {
        val sign = (bits ushr 15) and 0x1
        val exponent = (bits ushr 10) and 0x1F
        val fraction = bits and 0x3FF

        val floatBits = when (exponent) {
            0 -> {
                if (fraction == 0) {
                    sign shl 31
                } else {
                    var mantissa = fraction
                    var adjustedExponent = -14
                    while ((mantissa and 0x400) == 0) {
                        mantissa = mantissa shl 1
                        adjustedExponent -= 1
                    }
                    mantissa = mantissa and 0x3FF
                    (sign shl 31) or ((adjustedExponent + 127) shl 23) or (mantissa shl 13)
                }
            }
            0x1F -> {
                (sign shl 31) or 0x7F800000 or (fraction shl 13)
            }
            else -> {
                (sign shl 31) or ((exponent - 15 + 127) shl 23) or (fraction shl 13)
            }
        }

        return Float.fromBits(floatBits)
    }
}
