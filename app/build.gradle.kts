import com.android.build.api.dsl.ApplicationExtension
import com.android.build.gradle.tasks.PackageAndroidArtifact
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

plugins {
    alias(libs.plugins.android.application)
}

val pattern: DateTimeFormatter? = DateTimeFormatter.ofPattern("yyMMdd_HHmm")
val now: String? = LocalDateTime.now().format(pattern)

configure<ApplicationExtension> {
    namespace = "top.bogey.ocr.baidu.paddle"
    compileSdk = common.versions.compileSdk.get().toInt()
    ndkVersion = common.versions.ndkVersion.get()
    buildToolsVersion = common.versions.buildToolsVersion.get()

    defaultConfig {
        applicationId = "top.bogey.ocr.baidu.paddle"
        minSdk = 24
        targetSdk = common.versions.compileSdk.get().toInt()
        versionCode = 1
        versionName = now

        externalNativeBuild {
            cmake {
                arguments.add("-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON")
            }
        }

        ndk {
            //noinspection ChromeOsAbiSupport
            abiFilters.add("arm64-v8a")
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = true
            applicationIdSuffix = ".debug"
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")

        }

        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = common.versions.cmakeVersion.get()
        }
    }

    buildFeatures {
        aidl = true
    }
}

tasks.withType<PackageAndroidArtifact>().configureEach {
    if (name.contains("release", true)) {
        doLast {
            val dir = outputDirectory.get().asFile
            val apk = dir.listFiles()?.firstOrNull { it.extension == "apk" } ?: return@doLast

            val target = File(dir, "点击助手_Ocr_By_Baidu_PaddleOcr_${now}.APK")
            apk.copyTo(target, true)
        }
    }
}

dependencies {
}