import React, { useState } from 'react';
import { View, Text, Switch, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { useAuth } from '@clerk/clerk-expo';
import { useTheme } from './ThemeContext';  // Import the useTheme hook

const SettingsScreen = () => {
  const { signOut } = useAuth();
  const { theme, toggleTheme } = useTheme();  // Get theme and toggleTheme from context

  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [language, setLanguage] = useState('English');
  const [units, setUnits] = useState('Metric');

  const toggleLanguage = () => {
    setLanguage((prevLanguage) => (prevLanguage === 'English' ? 'Spanish' : 'English'));
  };

  const toggleUnits = () => {
    setUnits((prevUnits) => (prevUnits === 'Metric' ? 'Imperial' : 'Metric'));
  };

  const chooseTheme = () => {
    Alert.alert(
      "Choose Theme",
      "Select the app display theme:",
      [
        { text: "Black", onPress: () => toggleTheme('Black') },  // Use toggleTheme to update global theme
        { text: "White", onPress: () => toggleTheme('White') },
        { text: "Use Device Settings", onPress: () => toggleTheme('Use Device Settings'), style: "cancel" },
      ],
      { cancelable: true }
    );
  };

  const handleSignOut = () => {
    signOut();
  };

  const isDarkTheme = theme === 'dark';  // Check if the theme is dark

  return (
    <View style={[styles.container, isDarkTheme ? styles.darkContainer : styles.lightContainer]}>
      <Text style={[styles.title, isDarkTheme ? styles.darkText : styles.lightText]}>Settings</Text>

      <View style={[styles.setting, isDarkTheme ? styles.darkSetting : styles.lightSetting]}>
        <Text style={[styles.settingText, isDarkTheme ? styles.darkText : styles.lightText]}>Enable Notifications</Text>
        <Switch
          value={notificationsEnabled}
          onValueChange={setNotificationsEnabled}
        />
      </View>

      <TouchableOpacity style={[styles.setting, isDarkTheme ? styles.darkSetting : styles.lightSetting]} onPress={toggleLanguage}>
        <Text style={[styles.settingText, isDarkTheme ? styles.darkText : styles.lightText]}>Language</Text>
        <Text style={styles.unitText}>{language}</Text>
      </TouchableOpacity>

      <TouchableOpacity style={[styles.setting, isDarkTheme ? styles.darkSetting : styles.lightSetting]} onPress={toggleUnits}>
        <Text style={[styles.settingText, isDarkTheme ? styles.darkText : styles.lightText]}>Units</Text>
        <Text style={styles.unitText}>{units}</Text>
      </TouchableOpacity>

      <TouchableOpacity style={[styles.setting, isDarkTheme ? styles.darkSetting : styles.lightSetting]} onPress={chooseTheme}>
        <Text style={[styles.settingText, isDarkTheme ? styles.darkText : styles.lightText]}>Theme</Text>
        <Text style={styles.unitText}>{theme === 'dark' ? 'Black' : 'White'}</Text>
      </TouchableOpacity>

      {/* Sign Out Button */}
      <TouchableOpacity style={styles.signOutButton} onPress={handleSignOut}>
        <Text style={styles.signOutButtonText}>Sign Out</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  lightContainer: {
    backgroundColor: '#F9FAFB',
  },
  darkContainer: {
    backgroundColor: '#1D1C1F',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  lightText: {
    color: '#1D1C1F',
  },
  darkText: {
    color: '#F9FAFB',
  },
  setting: {
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  lightSetting: {
    backgroundColor: '#F4F6F8',
  },
  darkSetting: {
    backgroundColor: '#2D2D2D',
  },
  settingText: {
    fontSize: 16,
  },
  unitText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#F34533',
  },
  signOutButton: {
    backgroundColor: '#F34533',
    paddingVertical: 15,
    paddingHorizontal: 40,
    borderRadius: 10,
    marginTop: 20,
    alignItems: 'center',
  },
  signOutButtonText: {
    color: '#F9FAFB',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default SettingsScreen;
