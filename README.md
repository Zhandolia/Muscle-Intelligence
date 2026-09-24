# Muscle Intelligence

**[Open the interactive browser demo](https://zhandolia.github.io/Muscle-Intelligence/)** — a presentation of the mobile prototype's sample-video workflow. Uses prerecorded overlays and preset feedback; does not perform live analysis. See [demo documentation](docs/README.md).

<div align="center">

**A cutting-edge fitness application that leverages computer vision and machine learning to analyze exercise form in real-time, helping users perfect their technique and prevent injuries.**

[![React Native](https://img.shields.io/badge/React%20Native-0.75.3-61DAFB?logo=react)](https://reactnative.dev/)
[![Expo](https://img.shields.io/badge/Expo-51.0.32-000020?logo=expo)](https://expo.dev/)
[![Firebase](https://img.shields.io/badge/Firebase-10.14.1-FFCA28?logo=firebase)](https://firebase.google.com/)
[![Clerk](https://img.shields.io/badge/Clerk-Auth-000000?logo=clerk)](https://clerk.com/)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**Muscle Intelligence** is a sophisticated mobile fitness application designed to revolutionize how users approach their workouts. By combining advanced computer vision algorithms with machine learning models, the app provides real-time feedback on exercise form, identifies muscle groups being targeted, and helps users optimize their training routines while minimizing injury risk.

### What Makes It Special?

- **Real-Time Form Analysis**: Get instant feedback on your exercise technique using advanced pose estimation
- **Muscle Group Identification**: Automatically detect which muscle groups are being targeted during exercises
- **Visual Feedback**: Color-coded analysis showing perfect areas, areas needing work, and potential danger zones
- **Comprehensive Tracking**: Monitor workout history, progress, and body areas targeted over time
- **Professional-Grade**: Built with modern mobile development best practices and scalable architecture

---

## ✨ Key Features

### 🎥 Video Analysis
- **Upload & Analyze**: Upload workout videos from your device library
- **Real-Time Recording**: Record exercises directly through the app's camera interface
- **Skeleton Detection**: Advanced pose estimation with visual skeleton overlay
- **Form Scoring**: Get detailed feedback on exercise form with color-coded indicators:
  - 🟢 **Green**: Perfect form areas
  - 🟠 **Orange**: Areas needing improvement
  - 🔴 **Red**: Danger zones requiring attention

### 📊 Progress Tracking
- **Workout History**: Comprehensive log of all analyzed workouts
- **Body Areas Targeted**: Visual breakdown of muscle groups worked over time
- **Progress Analytics**: Track improvements in form and consistency
- **Exercise Library**: Browse and learn about different exercises

### 👤 User Management
- **Secure Authentication**: Powered by Clerk for enterprise-grade security
- **User Profiles**: Personalized profiles with workout statistics
- **Theme Support**: Light and dark mode for comfortable viewing
- **Device Management**: Track and manage connected fitness devices

### 🏠 Social Features
- **Feed**: Share progress and achievements with friends
- **Community**: Connect with other fitness enthusiasts

---

## 🛠 Technology Stack

### Frontend
- **React Native** `0.75.3` - Cross-platform mobile framework
- **Expo** `51.0.32` - Development platform and toolchain
- **React Navigation** `6.x` - Navigation library (Stack, Bottom Tabs)
- **React Native Reanimated** `3.15.3` - Smooth animations
- **React Native Gesture Handler** `2.20.0` - Touch gesture handling

### Backend & Services
- **Firebase** `10.14.1` - Backend-as-a-Service
  - Firestore - NoSQL database
  - Firebase Storage - Video and media storage
  - Firebase Authentication (via Clerk integration)
- **Clerk** `2.2.16` - Authentication and user management

### Media & Camera
- **Expo Camera** `15.0.16` - Camera access and video recording
- **Expo Image Picker** `15.0.7` - Image and video selection
- **Expo AV** `14.0.7` - Audio/video playback
- **React Native Video** `6.6.4` - Video player component

### Computer Vision & ML
- **TensorFlow.js** (via ML Playground) - Machine learning models
- **Pose Estimation Models** - Human pose detection
- **Three.js** `0.168.0` - 3D graphics and visualization
- **Expo GL** `14.0.2` - WebGL rendering

### Development Tools
- **TypeScript** (partial) - Type safety
- **ESLint** - Code linting
- **Metro Bundler** - React Native bundler

---

## 🏗 Architecture

### Application Structure

```
┌─────────────────────────────────────────┐
│         Navigation Container            │
├─────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────────┐ │
│  │  Auth Stack  │  │  Main App Tabs  │ │
│  │              │  │                 │ │
│  │ • Login      │  │ • Home          │ │
│  │ • Signup     │  │ • Record        │ │
│  │ • Verify     │  │ • Upload        │ │
│  └──────────────┘  │ • Profile       │ │
│                    └─────────────────┘ │
└─────────────────────────────────────────┘
```

### Data Flow

1. **User Authentication**: Clerk handles authentication → Firebase stores user data
2. **Video Upload**: User selects/records video → Uploaded to Firebase Storage
3. **Analysis Pipeline**: Video processed → ML models analyze form → Results stored in Firestore
4. **Feedback Display**: Analysis results → Visual feedback with skeleton overlay → User insights

### Key Components

- **Authentication Layer**: Clerk provider wrapping the entire app
- **Navigation**: Stack navigators for auth, upload flow, and profile sections
- **State Management**: React hooks and context (ThemeContext)
- **Media Processing**: Expo Camera and Image Picker for capture/selection
- **ML Integration**: ML playground models for pose estimation and form analysis

---

## 🚀 Installation & Setup

### Prerequisites

- **Node.js** >= 18.x
- **npm**, **yarn**, or **pnpm** (project uses pnpm)
- **Expo CLI** (install globally: `npm install -g expo-cli`)
- **iOS Development**: Xcode 14+ (for iOS builds)
- **Android Development**: Android Studio with Android SDK (for Android builds)
- **Firebase Account**: For backend services
- **Clerk Account**: For authentication

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Muscle-Intelligence
```

### Step 2: Install Dependencies

```bash
# Using pnpm (recommended)
pnpm install

# Or using npm
npm install

# Or using yarn
yarn install
```

### Step 3: Environment Configuration

Create a `.env` file in the root directory:

```env
EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
FIREBASE_API_KEY=your_firebase_api_key
FIREBASE_AUTH_DOMAIN=your_firebase_auth_domain
FIREBASE_PROJECT_ID=your_firebase_project_id
FIREBASE_STORAGE_BUCKET=your_firebase_storage_bucket
FIREBASE_MESSAGING_SENDER_ID=your_firebase_messaging_sender_id
FIREBASE_APP_ID=your_firebase_app_id
```

### Step 4: Firebase Setup

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Firestore Database
3. Enable Firebase Storage
4. Update `firebase/firebaseConfig.js` with your Firebase credentials
5. Deploy Firestore rules from `firestore.rules`
6. Deploy Firestore indexes from `firestore.indexes.json`

### Step 5: Clerk Setup

1. Create a Clerk account at [Clerk](https://clerk.com/)
2. Create a new application
3. Copy your publishable key to `.env`
4. Configure authentication methods (Email, Social, etc.)

### Step 6: iOS Setup (macOS only)

```bash
cd ios
pod install
cd ..
```

### Step 7: Start the Development Server

```bash
# Start Expo development server
npm start
# or
pnpm start
# or
yarn start
```

### Step 8: Run on Device/Emulator

- **iOS**: Press `i` in the terminal or scan QR code with Expo Go app
- **Android**: Press `a` in the terminal or scan QR code with Expo Go app
- **Web**: Press `w` in the terminal

---

## ⚙️ Configuration

### App Configuration (`app.json`)

```json
{
  "expo": {
    "name": "muscle-intelligence",
    "slug": "muscle-intelligence",
    "version": "1.0.0",
    "orientation": "portrait",
    "ios": {
      "bundleIdentifier": "com.yourcompany.muscleintelligence"
    },
    "android": {
      "package": "com.yourcompany.muscleintelligence"
    }
  }
}
```

### Firebase Configuration

Update `firebase/firebaseConfig.js` with your Firebase project credentials.

### Navigation Configuration

The app uses React Navigation with:
- **Bottom Tab Navigator**: Main app navigation (Home, Record, Upload, Profile)
- **Stack Navigators**: 
  - Auth Stack (Login, Signup, Verification)
  - Upload Stack (Upload flows for different analysis types)
  - Profile Stack (Profile and sub-screens)

---

## 📁 Project Structure

```
Muscle-Intelligence/
├── android/                 # Android native code
│   ├── app/
│   └── build.gradle
├── ios/                     # iOS native code
│   ├── postureAI/
│   └── Podfile
├── assets/                  # Static assets (images, videos, GIFs)
│   ├── *.gif               # Exercise demonstration GIFs
│   └── *.jpg               # Icons and images
├── firebase/                # Firebase configuration
│   ├── firebaseConfig.js   # Firebase initialization
│   ├── firestore.js        # Firestore utilities
│   └── firebaseStorage.js  # Storage utilities
├── hooks/                   # Custom React hooks
│   └── useProtectedRoute.js
├── ml_playground/           # ML models and playground
│   └── posts_code/         # Pose estimation models
├── screens/                 # Application screens
│   ├── auth/               # Authentication screens
│   │   ├── LoginScreen.js
│   │   ├── SignupScreen.js
│   │   └── VerificationScreen.js
│   ├── home/               # Home feed
│   │   └── HomeScreen.js
│   ├── record/             # Video recording
│   │   └── RecordScreen.js
│   ├── upload/             # Video upload and analysis
│   │   ├── UploadBetta.js
│   │   ├── UploadAlpha.js
│   │   ├── UploadGamma.js
│   │   ├── LoadingBettaOne.js
│   │   ├── LoadingBettaTwo.js
│   │   ├── LoadingAlphaOne.js
│   │   ├── LoadingAlphaTwo.js
│   │   ├── SkeletonBetta.js
│   │   ├── SkeletonAlpha.js
│   │   ├── FinalBetta.js
│   │   ├── FinalAlpha.js
│   │   └── StasSample.js
│   ├── profile/             # User profile and settings
│   │   ├── ProfileScreen.js
│   │   ├── WorkoutHistoryScreen.js
│   │   ├── BodyAreasTargetedScreen.js
│   │   ├── ProgressScreen.js
│   │   ├── SettingsScreen.js
│   │   ├── DevicesConnectedScreen.js
│   │   └── ThemeContext.js
│   ├── Button.js
│   ├── ExerciseLibraryScreen.js
│   ├── OnboardingScreen.js
│   ├── PostureAnalysisScreen.js
│   └── ProtectedScreen.js
├── App.js                   # Main application entry point
├── app.config.js           # Expo configuration
├── app.json                # App metadata
├── config.json             # Additional configuration
├── package.json            # Dependencies and scripts
├── firestore.rules         # Firestore security rules
└── firestore.indexes.json  # Firestore indexes
```

---

## 📖 Usage Guide

### Getting Started

1. **Sign Up/Login**: Create an account or sign in using Clerk authentication
2. **Onboarding**: Complete the initial setup process
3. **Grant Permissions**: Allow camera and storage access when prompted

### Recording a Workout

1. Navigate to the **Record** tab
2. Tap the camera button to start recording
3. Position yourself in frame
4. Perform your exercise
5. Stop recording when finished
6. The video will be automatically analyzed

### Uploading a Video

1. Navigate to the **Upload** tab
2. Tap "Upload Video"
3. Select a video from your device library
4. Wait for processing (video is converted to GIF format)
5. View the analysis results with:
   - Skeleton overlay showing pose detection
   - Color-coded feedback on form
   - Muscle groups targeted
   - Areas needing improvement

### Viewing Progress

1. Navigate to **Profile** tab
2. Access:
   - **Workout History**: View all past workouts
   - **Body Areas Targeted**: See muscle group breakdown
   - **Progress**: Track improvements over time
   - **Settings**: Customize app preferences

### Understanding Analysis Results

- **🟢 Green Areas**: Perfect form - keep doing what you're doing!
- **🟠 Orange Areas**: Needs work - focus on improving technique
- **🔴 Red Areas**: Danger zone - incorrect form that could lead to injury

---

## 💻 Development

### Available Scripts

```bash
# Start development server
npm start

# Run on iOS simulator
npm run ios

# Run on Android emulator
npm run android

# Run on web
npm run web
```

### Code Style

- Follow React Native best practices
- Use functional components with hooks
- Maintain consistent naming conventions
- Add comments for complex logic

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement changes following existing patterns
3. Test on both iOS and Android
4. Update documentation if needed
5. Submit pull request

### Testing

```bash
# Run tests (when test suite is added)
npm test
```

### Building for Production

```bash
# Build iOS
expo build:ios

# Build Android
expo build:android

# Or use EAS Build
eas build --platform ios
eas build --platform android
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Write clear, descriptive commit messages
- Follow the existing code style
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

---

## 📄 License

This project is private and proprietary. All rights reserved.

---

## 🔗 Additional Resources

- [React Native Documentation](https://reactnative.dev/docs/getting-started)
- [Expo Documentation](https://docs.expo.dev/)
- [Firebase Documentation](https://firebase.google.com/docs)
- [Clerk Documentation](https://clerk.com/docs)
- [React Navigation](https://reactnavigation.org/)

---

## 📞 Support

For issues, questions, or contributions, please open an issue on the repository or contact the development team.

---

## 🎯 Roadmap

### Upcoming Features

- [ ] Real-time form analysis during recording
- [ ] Exercise recommendations based on progress
- [ ] Social sharing and community features
- [ ] Integration with fitness wearables
- [ ] Advanced analytics and insights
- [ ] Custom workout plan generation
- [ ] Multi-exercise analysis in single session
- [ ] Export workout data
- [ ] Integration with popular fitness apps

---

<div align="center">

**Built with ❤️ using React Native and Expo**

*Helping you achieve perfect form, one rep at a time.*

</div>
