import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'ai.chaios.app',
  appName: 'chAIos',
  webDir: 'dist/app',
  server: {
    androidScheme: 'https'
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 3000,
      launchAutoHide: true,
      backgroundColor: "#3880ff",
      androidSplashResourceName: "splash",
      androidScaleType: "CENTER_CROP",
      showSpinner: true,
      androidSpinnerStyle: "large",
      iosSpinnerStyle: "small",
      spinnerColor: "#ffffff",
      splashFullScreen: true,
      splashImmersive: true,
    },
    StatusBar: {
      style: 'dark',
      backgroundColor: '#3880ff',
    },
    Keyboard: {
      resize: 'body',
      style: 'dark',
      resizeOnFullScreen: true,
    },
    App: {
      launchUrl: 'https://chaios.ai'
    },
    Device: {
      // Enable device info access
    },
    Network: {
      // Enable network status monitoring
    },
    Haptics: {
      // Enable haptic feedback
    },
    LocalNotifications: {
      smallIcon: "ic_stat_icon_config_sample",
      iconColor: "#3880ff",
      sound: "beep.wav",
    },
    PushNotifications: {
      presentationOptions: ["badge", "sound", "alert"],
    },
    Camera: {
      // Enable camera access for QR scanning, etc.
    },
    Filesystem: {
      // Enable file system access for exports
    },
    Share: {
      // Enable sharing functionality
    },
    Browser: {
      // Enable in-app browser
    }
  },
  android: {
    allowMixedContent: true,
    captureInput: true,
    webContentsDebuggingEnabled: true,
    appendUserAgent: 'chAIos-Android',
    overrideUserAgent: 'chAIos-Mobile-App',
    backgroundColor: '#3880ff',
    loggingBehavior: 'debug',
  },
  ios: {
    scheme: 'chAIos',
    contentInset: 'automatic',
    scrollEnabled: true,
    allowsLinkPreview: true,
    handleApplicationURL: true,
    appendUserAgent: 'chAIos-iOS',
    overrideUserAgent: 'chAIos-Mobile-App',
    backgroundColor: '#3880ff',
    preferredContentMode: 'mobile',
  }
};

export default config;