import com.android.build.gradle.internal.api.BaseVariantOutputImpl
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "top.bogey.ocr.baidu.paddle"
    compileSdk = 36
    buildToolsVersion = "36.0.0"
    ndkVersion = "27.2.12479018"

    val pattern = DateTimeFormatter.ofPattern("yyMMdd_HHmm")
    val now = LocalDateTime.now().format(pattern)

    defaultConfig {
        applicationId = "top.bogey.ocr.baidu.paddle"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = now

        externalNativeBuild {
            cmake {
                cppFlags.add("-std=c++14")
                cppFlags.add("-frtti")
                cppFlags.add("-fexceptions")
                cppFlags.add("-Wno-format")

                arguments.add("-DANDROID_PLATFORM=android-35")
                arguments.add("-DANDROID_STL=c++_shared")
                arguments.add("-DANDROID_ARM_NEON=TRUE")
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

    applicationVariants.all {
        outputs.all {
            if (buildType.name == "release") {
                val impl = this as BaseVariantOutputImpl
                impl.outputFileName = "点击助手_Ocr_By_Baidu_PaddleOcr_$now.APK"
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    buildFeatures {
        aidl = true
    }
}

dependencies {
}