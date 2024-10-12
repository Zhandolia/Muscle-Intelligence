import React from 'react';
import { TouchableOpacity, Text, View, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import Icon from 'react-native-vector-icons/Ionicons';
import { ClerkProvider, SignedIn, SignedOut } from '@clerk/clerk-expo';

import HomeScreen from './screens/home/HomeScreen';
import RecordScreen from './screens/record/RecordScreen';
import UploadScreen from './screens/upload/UploadScreen';
import VideoUploadScreen from './screens/upload/VideoUploadScreen';
import ProfileScreen from './screens/profile/ProfileScreen';
import WorkoutHistoryScreen from './screens/profile/WorkoutHistoryScreen';
import BodyAreasTargetedScreen from './screens/profile/BodyAreasTargeted';
import ProgressScreen from './screens/profile/ProgressScreen';
import SettingsScreen from './screens/profile/SettingsScreen';
import DevicesConnectedScreen from './screens/profile/DevicesConnected';
import LoginScreen from './screens/auth/LoginScreen';
import SignupScreen from './screens/auth/SignupScreen';
import VerificationScreen from './screens/auth/VerificationScreen';
import { ThemeProvider } from './screens/profile/ThemeContext';

// Stack Navigator for Upload-related pages
const UploadStack = createStackNavigator();
const AuthStack = createStackNavigator();

// Authentication Stack (Login, Signup, Verification)
function AuthStackScreen() {
  return (
    <AuthStack.Navigator initialRouteName="Login">
      <AuthStack.Screen name="Login" component={LoginScreen} options={{ headerShown: false }} />
      <AuthStack.Screen name="Signup" component={SignupScreen} options={{ headerShown: false }} />
      <AuthStack.Screen name="Verification" component={VerificationScreen} options={{ headerShown: false }} />
    </AuthStack.Navigator>
  );
}

// Upload-related Stack
function UploadStackScreen() {
  return (
    <UploadStack.Navigator>
      <UploadStack.Screen name="Upload" component={UploadScreen} options={{ headerShown: false }} />
      <UploadStack.Screen name="VideoUpload" component={VideoUploadScreen} options={{ headerShown: false }} />
    </UploadStack.Navigator>
  );
}

// Custom header with back button for profile pages
const HeaderWithBackButton = ({ navigation, title }) => (
  <View style={styles.header}>
    <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
      <Icon name="arrow-back" size={24} color="#F34533" />
    </TouchableOpacity>
    <Text style={styles.headerTitle}>{title}</Text>
  </View>
);

// Profile-related Stack
const ProfileStack = createStackNavigator();

function ProfileStackScreen() {
  return (
    <ProfileStack.Navigator>
      <ProfileStack.Screen name="Profile" component={ProfileScreen} options={{ headerShown: false }} />
      <ProfileStack.Screen
        name="WorkoutHistory"
        component={WorkoutHistoryScreen}
        options={({ navigation }) => ({
          header: () => <HeaderWithBackButton navigation={navigation} title="Workout History" />,
        })}
      />
      <ProfileStack.Screen
        name="BodyAreasTargeted"
        component={BodyAreasTargetedScreen}
        options={({ navigation }) => ({
          header: () => <HeaderWithBackButton navigation={navigation} title="Body Areas Targeted" />,
        })}
      />
      <ProfileStack.Screen
        name="Progress"
        component={ProgressScreen}
        options={({ navigation }) => ({
          header: () => <HeaderWithBackButton navigation={navigation} title="Progress" />,
        })}
      />
      <ProfileStack.Screen
        name="Settings"
        component={SettingsScreen}
        options={({ navigation }) => ({
          header: () => <HeaderWithBackButton navigation={navigation} title="Settings" />,
        })}
      />
      <ProfileStack.Screen
        name="DevicesConnected"
        component={DevicesConnectedScreen}
        options={({ navigation }) => ({
          header: () => <HeaderWithBackButton navigation={navigation} title="Devices Connected" />,
        })}
      />
    </ProfileStack.Navigator>
  );
}

// Bottom Tab Navigator
const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <ClerkProvider publishableKey={process.env.EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY}>
      <ThemeProvider>
        <NavigationContainer>
          {/* Show different navigators based on authentication status */}
          <SignedIn>
            <Tab.Navigator
              initialRouteName="Home"
              screenOptions={({ route }) => ({
                tabBarIcon: ({ color, size }) => {
                  let iconName;

                  if (route.name === 'Home') {
                    iconName = 'home-outline';
                  } else if (route.name === 'Record') {
                    iconName = 'camera-outline';
                  } else if (route.name === 'Upload') {
                    iconName = 'cloud-upload-outline';
                  } else if (route.name === 'Profile') {
                    iconName = 'person-outline';
                  }

                  return <Icon name={iconName} size={size} color={color} />;
                },
                tabBarActiveTintColor: '#F34533',
                tabBarInactiveTintColor: '#636165',
                tabBarStyle: { backgroundColor: '#F4F6F8' },
              })}
            >
              <Tab.Screen name="Home" component={HomeScreen} />
              <Tab.Screen name="Record" component={RecordScreen} />
              <Tab.Screen name="Upload" component={UploadStackScreen} />
              <Tab.Screen name="Profile" component={ProfileStackScreen} />
            </Tab.Navigator>
          </SignedIn>

          {/* Auth Flow when user is not signed in */}
          <SignedOut>
            <AuthStackScreen />
          </SignedOut>
        </NavigationContainer>
      </ThemeProvider>
    </ClerkProvider>
  );
}

// Styles for custom header
const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 15,
    backgroundColor: '#F9FAFB',
    borderBottomWidth: 1,
    borderBottomColor: '#F4F6F8',
  },
  backButton: {
    marginRight: 10,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
});