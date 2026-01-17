import com.android.build.gradle.internal.api.BaseVariantOutputImpl
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "top.bogey.ocr.baidu.paddle"
    compileSdk = 36
    ndkVersion = "29.0.14206865"
    buildToolsVersion = "36.1.0"

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
                cppFlags.add("-std=c++17")
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
            version = "4.1.2"
        }
    }

    buildFeatures {
        aidl = true
    }
}

dependencies {
}