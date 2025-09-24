# chAIos Frontend Architecture Guide

This document outlines the architecture and guiding principles for the chAIos Ionic/Angular frontend, designed for a device-agnostic, scalable, and maintainable MEAN stack application.

## 1. Conceptual Overview

The core principle is a **device-agnostic core** with **platform-specific overlays**. This allows us to maintain a single codebase for UI primitives, business logic, and services, while applying small, targeted alterations for each platform (web, iOS, Android).

-   **Build-Time Replacements**: We heavily favor build-time file replacements over runtime detection for significant behavioral or stylistic changes. This keeps bundles lean and platform-specific.
-   **Runtime Detection**: Used minimally for minor adjustments, like enabling native plugins (e.g., haptics) where available.

### Architectural Layers

1.  **Core / Shared (Device-Agnostic)**
    -   **Location**: `src/app/core`, `src/app/shared`
    -   **Responsibilities**: Core services (`ApiService`, `UXService`), domain models, UI primitive components (`<app-button>`), and API contract interfaces.

2.  **Platform Overlays (Platform-Specific)**
    -   **Location**: `src/app/platforms/{platform}`
    -   **Responsibilities**: Small, composable modules and services that provide platform-native implementations (e.g., `SecureStorageIonic` vs. `SecureStorageWeb`). Styles are overridden in `src/theme/variables-{platform}.scss`.

3.  **Backend (Node/Express)**
    -   **Location**: `/server`
    -   **Responsibilities**: Serves the single-page application and provides versioned REST APIs.

## 2. File Structure

The project follows a strict file structure to enforce this separation of concerns.

```
/chaios-frontend
  /docker                 <-- Docker configuration
  /server                 <-- Node.js/Express backend
  /src
    /app
      /core               <-- Core services, models, interfaces
      /features           <-- Feature modules (pages)
      /platforms
        /ios              <-- iOS-specific components/services
        /android          <-- Android-specific components/services
        /web              <-- Web-specific components/services
      /shared             <-- Shared components, directives, pipes
      platform-providers.ts   <-- DI providers (replaced at build time)
    /assets
    /environments         <-- Environment files per platform
    /theme
      _tokens.scss        <-- Raw design tokens
      variables.scss      <-- Base theme variables (replaced at build time)
      variables-ios.scss
      variables-android.scss
      variables-web.scss
  angular.json
  capacitor.config.ts
  rules.yaml              <-- Definitive architectural rules
```

## 3. Build System & Platform Specialization

Our build process uses file replacements defined in `angular.json` to generate platform-specific builds.

### Build Configurations

-   `production-web`: Builds the Progressive Web App (PWA).
-   `production-ios`: Builds the app bundle for iOS.
-   `production-android`: Builds the app bundle for Android.

These are executed via npm scripts:

```bash
npm run build:web
npm run build:ios
npm run build:android
```

### File Replacements

The build system swaps out the following files per configuration:

1.  **Environment Configuration**: `src/environments/environment.ts` is replaced with `src/environments/environment.{platform}.ts`.
2.  **Theme Variables**: `src/theme/variables.scss` is replaced with `src/theme/variables-{platform}.scss`.
3.  **Platform Providers**: `src/app/platform-providers.ts` is replaced with `src/app/platform-providers.{platform}.ts` to provide the correct services via Dependency Injection.

## 4. Platform-Specific Services Pattern

To handle platform-specific dependencies (like native storage), we use an interface-driven approach with DI.

1.  **Define a Contract**: An interface is created in `src/app/core/interfaces` (e.g., `ISecureStorage`).
2.  **Create Implementations**: Platform-specific classes that implement the interface are created in `src/app/platforms/{platform}/services`.
3.  **Provide at Build Time**: The `platform-providers.ts` file maps the interface to the correct implementation. This file is then swapped out during the build process.

**Example: `ISecureStorage`**
-   **Interface**: `src/app/core/interfaces/secure-storage.interface.ts`
-   **Web Impl**: `src/app/platforms/web/services/secure-storage.web.service.ts` (uses `localStorage`).
-   **Ionic Impl**: `src/app/platforms/ionic/services/secure-storage.ionic.service.ts` (uses Capacitor Storage).
-   **Provider Files**: `platform-providers.web.ts` and `platform-providers.ionic.ts`.

## 5. SCSS Theming

Our theming is based on a token system that allows for easy platform-specific overrides.

-   `_tokens.scss`: Contains all raw, platform-agnostic design values (colors, spacing, etc.).
-   `variables-{platform}.scss`: Imports `_tokens.scss` and maps the tokens to Ionic's CSS custom properties, applying any platform-specific adjustments (e.g., different fonts or border-radius for iOS).

## 6. Developer Checklist

When adding new features, follow these rules:

-   **Core UI components** live in `src/app/shared`.
-   **Platform-specific files** live in `src/app/platforms/{platform}`.
-   **Theme tokens** are defined in `src/theme/_tokens.scss` only.
-   **Do not import platform-only plugins** in shared modules. Use the platform-specific service pattern instead.
-   **Update `angular.json` fileReplacements** when adding new platforms or provider types.
-   Refer to `rules.yaml` for all naming and structure conventions.
