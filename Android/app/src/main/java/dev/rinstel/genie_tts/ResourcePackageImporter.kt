package dev.rinstel.genie_tts

import dev.rinstel.genie_tts.inference.RuntimeAssetRepository
import java.io.BufferedInputStream
import java.io.File
import java.io.InputStream
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID
import java.util.zip.ZipInputStream

/** Installs the two user-downloadable package types without exposing ZIP paths to the filesystem. */
class ResourcePackageImporter(
    private val filesRoot: File,
) {
    enum class PackageType {
        RUNTIME,
        CHARACTER,
    }

    data class ImportResult(
        val packageType: PackageType,
        val fileCount: Int,
        val totalBytes: Long,
        val characterIds: List<String> = emptyList(),
    )

    fun importRuntime(input: InputStream): ImportResult =
        importPackage(input, PackageType.RUNTIME, RUNTIME_ROOT) { stagingRoot, stats ->
            val missing = RuntimeAssetRepository(stagingRoot).inspect().missingFiles
            require(missing.isEmpty()) {
                "公共资源包缺少文件：${missing.joinToString("、")}"
            }
            replaceDirectories(
                stagingRoot = stagingRoot,
                replacements = listOf(
                    DirectoryReplacement(
                        source = File(stagingRoot, RUNTIME_ROOT),
                        target = File(filesRoot, RUNTIME_ROOT),
                    ),
                ),
            )
            ImportResult(PackageType.RUNTIME, stats.fileCount, stats.totalBytes)
        }

    fun importCharacter(input: InputStream): ImportResult =
        importPackage(input, PackageType.CHARACTER, CHARACTER_ROOT) { stagingRoot, stats ->
            val characterRoots = discoverCharacterRoots(File(stagingRoot, CHARACTER_ROOT))
            require(characterRoots.isNotEmpty()) {
                "角色模型包缺少 CharacterModels/<版本>/<角色>/tts_models 目录。"
            }
            characterRoots.forEach { characterRoot ->
                val onnxFiles = File(characterRoot.source, "tts_models")
                    .walkTopDown()
                    .filter { it.isFile && it.extension.equals("onnx", ignoreCase = true) }
                    .toList()
                require(onnxFiles.isNotEmpty()) {
                    "角色 ${characterRoot.id} 的 tts_models 目录缺少 ONNX 模型。"
                }
            }

            val replacements = characterRoots.map { characterRoot ->
                DirectoryReplacement(
                    source = characterRoot.source,
                    target = File(filesRoot, "$CHARACTER_ROOT/${characterRoot.relativeId}"),
                )
            }
            replaceDirectories(stagingRoot, replacements)
            ImportResult(
                packageType = PackageType.CHARACTER,
                fileCount = stats.fileCount,
                totalBytes = stats.totalBytes,
                characterIds = characterRoots.map(CharacterRoot::relativeId),
            )
        }

    private fun importPackage(
        input: InputStream,
        packageType: PackageType,
        expectedRoot: String,
        install: (File, ExtractionStats) -> ImportResult,
    ): ImportResult {
        require(filesRoot.isDirectory || filesRoot.mkdirs()) {
            "无法创建应用资源目录：${filesRoot.absolutePath}"
        }
        val stagingRoot = File(
            filesRoot,
            ".genie-import-${packageType.name.lowercase()}-${UUID.randomUUID()}",
        )
        require(stagingRoot.mkdirs()) {
            "无法创建资源导入临时目录：${stagingRoot.absolutePath}"
        }
        return try {
            val stats = extractZip(input, stagingRoot, expectedRoot)
            install(stagingRoot, stats)
        } finally {
            stagingRoot.deleteRecursively()
        }
    }

    private fun extractZip(
        input: InputStream,
        stagingRoot: File,
        expectedRoot: String,
    ): ExtractionStats {
        val seenEntries = HashSet<String>()
        val rootCanonical = stagingRoot.canonicalFile
        var fileCount = 0
        var totalBytes = 0L
        val buffer = ByteArray(BUFFER_SIZE)

        ZipInputStream(BufferedInputStream(input)).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                val relativePath = normalizeEntryPath(entry.name, expectedRoot)
                require(seenEntries.add(relativePath)) {
                    "资源包包含重复文件：$relativePath"
                }

                val output = File(stagingRoot, relativePath)
                require(isWithin(rootCanonical, output.canonicalFile)) {
                    "资源包路径越界：${entry.name}"
                }
                if (entry.isDirectory) {
                    require(output.mkdirs() || output.isDirectory) {
                        "无法创建资源目录：${output.absolutePath}"
                    }
                    continue
                }

                require(output.parentFile?.let { it.isDirectory || it.mkdirs() } != false) {
                    "无法创建资源父目录：${output.parentFile?.absolutePath}"
                }
                output.outputStream().use { destination ->
                    while (true) {
                        val read = zip.read(buffer)
                        if (read < 0) break
                        val nextTotal = totalBytes + read
                        require(nextTotal <= MAX_UNCOMPRESSED_BYTES) {
                            "资源包解压内容超过 ${MAX_UNCOMPRESSED_BYTES / BYTES_PER_GIB} GiB 限制。"
                        }
                        destination.write(buffer, 0, read)
                        totalBytes = nextTotal
                    }
                }
                fileCount += 1
                require(fileCount <= MAX_FILE_COUNT) {
                    "资源包文件数量超过 $MAX_FILE_COUNT 个限制。"
                }
            }
        }
        return ExtractionStats(fileCount, totalBytes)
    }

    private fun normalizeEntryPath(entryName: String, expectedRoot: String): String {
        val path = entryName.replace('\\', '/').trimEnd('/')
        require(path.isNotBlank() && !path.startsWith('/')) {
            "资源包包含绝对路径：$entryName"
        }
        val parts = path.split('/')
        require(parts.none { it.isBlank() || it == "." || it == ".." }) {
            "资源包包含非法路径：$entryName"
        }
        require(parts.firstOrNull() == expectedRoot) {
            "资源包必须以 $expectedRoot/ 为根：$entryName"
        }
        return parts.joinToString("/")
    }

    private fun discoverCharacterRoots(characterRoot: File): List<CharacterRoot> {
        if (!characterRoot.isDirectory) return emptyList()
        return characterRoot.listFiles()
            ?.filter(File::isDirectory)
            ?.flatMap { variant ->
                variant.listFiles()
                    ?.filter { it.isDirectory && File(it, "tts_models").isDirectory }
                    ?.map { character ->
                        CharacterRoot(
                            source = character,
                            relativeId = "${variant.name}/${character.name}",
                        )
                    }
                    .orEmpty()
            }
            ?.sortedBy(CharacterRoot::relativeId)
            .orEmpty()
    }

    private fun replaceDirectories(
        stagingRoot: File,
        replacements: List<DirectoryReplacement>,
    ) {
        val backupRoot = File(stagingRoot, BACKUP_DIRECTORY).apply { mkdirs() }
        val completed = mutableListOf<CompletedReplacement>()
        try {
            replacements.forEachIndexed { index, replacement ->
                val backup = File(backupRoot, "$index-${replacement.target.name}")
                val completedReplacement = CompletedReplacement(replacement, backup)
                completed += completedReplacement
                if (replacement.target.exists()) {
                    moveDirectory(replacement.target, backup)
                    completedReplacement.oldMoved = true
                }
                moveDirectory(replacement.source, replacement.target)
                completedReplacement.newMoved = true
            }
            completed.forEach { it.backup.deleteRecursively() }
        } catch (error: Throwable) {
            completed.asReversed().forEach { replacement ->
                runCatching { rollback(replacement) }
                    .onFailure(error::addSuppressed)
            }
            throw error
        }
    }

    private fun rollback(replacement: CompletedReplacement) {
        if (replacement.newMoved && replacement.replacement.target.exists()) {
            replacement.replacement.target.deleteRecursively()
        }
        if (replacement.oldMoved && replacement.backup.exists()) {
            moveDirectory(replacement.backup, replacement.replacement.target)
        }
    }

    private fun moveDirectory(source: File, target: File) {
        require(source.exists()) { "资源导入源目录不存在：${source.absolutePath}" }
        target.parentFile?.let { parent ->
            require(parent.exists() || parent.mkdirs()) {
                "无法创建资源目标目录：${parent.absolutePath}"
            }
        }
        try {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE)
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(source.toPath(), target.toPath())
        } catch (_: UnsupportedOperationException) {
            Files.move(source.toPath(), target.toPath())
        }
    }

    private fun isWithin(root: File, candidate: File): Boolean =
        candidate == root || candidate.path.startsWith(root.path + File.separator)

    private data class ExtractionStats(
        val fileCount: Int,
        val totalBytes: Long,
    )

    private data class CharacterRoot(
        val source: File,
        val relativeId: String,
    ) {
        val id: String
            get() = relativeId.substringAfter('/')
    }

    private data class DirectoryReplacement(
        val source: File,
        val target: File,
    )

    private class CompletedReplacement(
        val replacement: DirectoryReplacement,
        val backup: File,
        var oldMoved: Boolean = false,
        var newMoved: Boolean = false,
    )

    companion object {
        private const val RUNTIME_ROOT = "RuntimeAssets"
        private const val CHARACTER_ROOT = "CharacterModels"
        private const val BACKUP_DIRECTORY = ".genie-import-backups"
        private const val BUFFER_SIZE = 1024 * 1024
        private const val MAX_FILE_COUNT = 100_000
        private const val BYTES_PER_GIB = 1024L * 1024L * 1024L
        private const val MAX_UNCOMPRESSED_BYTES = 4L * BYTES_PER_GIB
    }
}
