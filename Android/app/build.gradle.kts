import org.gradle.api.DefaultTask
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.tasks.testing.Test
import org.gradle.api.tasks.Copy
import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.InputDirectory
import org.gradle.api.tasks.OutputFile
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.Sync
import org.gradle.api.tasks.TaskAction
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

plugins {
    id("com.android.application") version "9.2.1"
}

// Common inference models are shipped as one resource package, not copied
// into every character model package. The source path is configurable so
// generated model binaries do not need to live in the Git worktree.
val runtimeAssetSource = providers.gradleProperty("genieRuntimeAssets")
    .orElse(providers.environmentVariable("GENIE_TTS_RUNTIME_ASSETS"))
    .orElse("G:/AIGC/Genie-TTS GUI/GenieData")
val runtimeAssetOutput = layout.buildDirectory.dir("generated/runtimeAssets")
val runtimeAssetManifestOutput = layout.buildDirectory.file(
    "generated/runtimeAssetManifest/chinese-hubert-base_weights_fp16_manifest.json",
)

abstract class HighCompressionRuntimeAssetsTask : DefaultTask() {
    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val sourceDirectory: DirectoryProperty

    @get:OutputFile
    abstract val archiveFile: RegularFileProperty

    @TaskAction
    fun createArchive() {
        val sourceRoot = sourceDirectory.get().asFile
        val outputFile = archiveFile.get().asFile
        outputFile.parentFile.mkdirs()
        val temporaryFile = File(
            outputFile.parentFile,
            ".${outputFile.name}.${System.nanoTime()}.tmp",
        )
        try {
            temporaryFile.outputStream().buffered().use { output ->
                ZipOutputStream(output).use { archive ->
                    // Use the same high-compression policy as character packages.
                    archive.setLevel(Deflater.BEST_COMPRESSION)
                    sourceRoot.walkTopDown()
                        .filter(File::isFile)
                        .sortedBy { it.relativeTo(sourceRoot).invariantSeparatorsPath }
                        .forEach { file ->
                            val entryName = file.relativeTo(sourceRoot).invariantSeparatorsPath
                            archive.putNextEntry(ZipEntry(entryName))
                            file.inputStream().use { it.copyTo(archive) }
                            archive.closeEntry()
                        }
                }
            }
            Files.move(
                temporaryFile.toPath(),
                outputFile.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
            )
        } finally {
            temporaryFile.delete()
        }
    }
}

val generateRuntimeAssetManifest = tasks.register<Exec>("generateRuntimeAssetManifest") {
    val sourceRoot = file(runtimeAssetSource.get())
    inputs.file(File(sourceRoot, "chinese-hubert-base/chinese-hubert-base.onnx"))
    inputs.file(File(sourceRoot, "chinese-hubert-base/chinese-hubert-base_weights_fp16.bin"))
    outputs.file(runtimeAssetManifestOutput)
    workingDir(rootProject.projectDir.parentFile)
    commandLine(
        "python",
        "tools/generate_runtime_asset_manifest.py",
        File(sourceRoot, "chinese-hubert-base/chinese-hubert-base.onnx").absolutePath,
        File(sourceRoot, "chinese-hubert-base/chinese-hubert-base_weights_fp16.bin").absolutePath,
        runtimeAssetManifestOutput.get().asFile.absolutePath,
    )
}
val syncRuntimeAssets = tasks.register<Sync>("syncRuntimeAssets") {
    dependsOn(generateRuntimeAssetManifest)
    val sourceRoot = file(runtimeAssetSource.get())
    val requiredSources = listOf(
        File(sourceRoot, "chinese-hubert-base/chinese-hubert-base.onnx"),
        File(sourceRoot, "chinese-hubert-base/chinese-hubert-base_weights_fp16.bin"),
        File(sourceRoot, "speaker_encoder.onnx"),
        File(sourceRoot, "RoBERTa/RoBERTa.onnx"),
        File(sourceRoot, "RoBERTa/roberta_tokenizer/tokenizer.json"),
    )
    doFirst {
        val missing = requiredSources.filterNot(File::isFile)
        check(missing.isEmpty()) {
            "Missing Android runtime asset sources: ${missing.joinToString(", ")}. " +
                "Set -PgenieRuntimeAssets=<GenieData> or GENIE_TTS_RUNTIME_ASSETS."
        }
    }

    from(File(sourceRoot, "chinese-hubert-base/chinese-hubert-base.onnx")) {
        into("RuntimeAssets/chinese-hubert-base")
    }
    from(File(sourceRoot, "chinese-hubert-base/chinese-hubert-base_weights_fp16.bin")) {
        into("RuntimeAssets/chinese-hubert-base")
    }
    from(runtimeAssetManifestOutput) {
        into("RuntimeAssets/chinese-hubert-base")
    }
    from(File(sourceRoot, "speaker_encoder.onnx")) {
        into("RuntimeAssets")
    }
    from(File(sourceRoot, "RoBERTa/RoBERTa.onnx")) {
        into("RuntimeAssets/roberta-wwm-ext-large-onnx")
        rename { "model.onnx" }
    }
    from(File(sourceRoot, "RoBERTa/roberta_tokenizer")) {
        into("RuntimeAssets/roberta-wwm-ext-large-onnx/roberta_tokenizer")
    }
    into(runtimeAssetOutput)
}

tasks.register<HighCompressionRuntimeAssetsTask>("bundleRuntimeAssets") {
    dependsOn(syncRuntimeAssets)
    sourceDirectory.set(runtimeAssetOutput)
    archiveFile.set(
        layout.buildDirectory.file("outputs/runtime-assets/genie-tts-runtime-assets.zip"),
    )
}

android {
    namespace = "dev.rinstel.genie_tts"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "dev.rinstel.genie_tts"
        minSdk = 30
        targetSdk = 36
        versionCode = 2
        versionName = "0.1.1"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += "-DANDROID_STL=c++_shared"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    flavorDimensions += "runtime"
    productFlavors {
        create("cpu") {
            dimension = "runtime"
            ndk {
                // The production target is the connected arm64 Android device.
                abiFilters += "arm64-v8a"
            }
        }
    }

    androidResources {
        // Runtime assets are delivered by the separate resource package. Keep
        // the APK's small built-in frontend assets ordinary compressed assets.
        noCompress += listOf("onnx", "bin")
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

tasks.withType<Test>().configureEach {
    maxParallelForks = 1
    jvmArgs("-Xmx96m")
}

// Keep the historical app-debug.apk path for existing install scripts; the
// explicit cpuDebug output remains the source of truth.
tasks.register<Copy>("copyCpuDebugApkToLegacyDebugPath") {
    dependsOn("assembleCpuDebug")
    from(layout.buildDirectory.file("outputs/apk/cpu/debug/app-cpu-debug.apk"))
    into(layout.buildDirectory.dir("outputs/apk/debug"))
    rename { "app-debug.apk" }
}

afterEvaluate {
    tasks.named("assembleDebug") {
        dependsOn("assembleCpuDebug")
        finalizedBy("copyCpuDebugApkToLegacyDebugPath")
    }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("androidx.core:core-ktx:1.19.0")
    implementation("com.google.android.material:material:1.14.0")
    add("cpuImplementation", "com.microsoft.onnxruntime:onnxruntime-android:1.27.0")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
}
