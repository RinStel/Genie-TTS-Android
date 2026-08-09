package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertArrayEquals
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class HalfPrecisionTest {
    @Test
    fun convertsLittleEndianFp16BytesToLittleEndianFp32Bytes() {
        val fp16 = byteArrayOf(
            0x00, 0x3C, // 1.0
            0x00, 0xC0.toByte(), // -2.0
        )

        val actual = HalfPrecision.fp16BytesToFp32Bytes(fp16)

        val floats = ByteBuffer.wrap(actual)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()
        val result = FloatArray(floats.remaining())
        floats.get(result)

        assertArrayEquals(floatArrayOf(1.0f, -2.0f), result, 0.0f)
    }
}
