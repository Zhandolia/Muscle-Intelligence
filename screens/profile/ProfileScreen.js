import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native'; // Import useNavigation
import { useTheme } from './ThemeContext';  // Import useTheme from ThemeContext

const ProfileScreen = () => {
  const navigation = useNavigation(); // Initialize navigation
  const { theme } = useTheme();  // Get the current theme from ThemeContext
  const isDarkTheme = theme === 'dark';  // Check if the theme is dark

  return (
    <ScrollView style={[styles.container, isDarkTheme ? styles.darkContainer : styles.lightContainer]}>
      <Text style={[styles.title, isDarkTheme ? styles.darkText : styles.lightText]}>Profile</Text>

      <TouchableOpacity style={[styles.section, isDarkTheme ? styles.darkSection : styles.lightSection]} onPress={() => navigation.navigate('BodyAreasTargeted')}>
        <Text style={[styles.sectionTitle, isDarkTheme ? styles.darkText : styles.lightText]}>Body Areas Targeted</Text>
        <Text style={[styles.sectionContent, isDarkTheme ? styles.darkContentText : styles.lightContentText]}>See which muscles you've been focusing on.</Text>
      </TouchableOpacity>

      <TouchableOpacity style={[styles.section, isDarkTheme ? styles.darkSection : styles.lightSection]} onPress={() => navigation.navigate('DevicesConnected')}>
        <Text style={[styles.sectionTitle, isDarkTheme ? styles.darkText : styles.lightText]}>Devices Connected</Text>
        <Text style={[styles.sectionContent, isDarkTheme ? styles.darkContentText : styles.lightContentText]}>View data from your connected devices.</Text>
      </TouchableOpacity>

      <TouchableOpacity style={[styles.section, isDarkTheme ? styles.darkSection : styles.lightSection]} onPress={() => navigation.navigate('WorkoutHistory')}>
        <Text style={[styles.sectionTitle, isDarkTheme ? styles.darkText : styles.lightText]}>Workout History</Text>
        <Text style={[styles.sectionContent, isDarkTheme ? styles.darkContentText : styles.lightContentText]}>Your past workouts will appear here.</Text>
      </TouchableOpacity>

      <TouchableOpacity style={[styles.section, isDarkTheme ? styles.darkSection : styles.lightSection]} onPress={() => navigation.navigate('Progress')}>
        <Text style={[styles.sectionTitle, isDarkTheme ? styles.darkText : styles.lightText]}>Progress</Text>
        <Text style={[styles.sectionContent, isDarkTheme ? styles.darkContentText : styles.lightContentText]}>Track your progress over time.</Text>
      </TouchableOpacity>

      <TouchableOpacity style={[styles.section, isDarkTheme ? styles.darkSection : styles.lightSection]} onPress={() => navigation.navigate('Settings')}>
        <Text style={[styles.sectionTitle, isDarkTheme ? styles.darkText : styles.lightText]}>Settings</Text>
        <Text style={[styles.sectionContent, isDarkTheme ? styles.darkContentText : styles.lightContentText]}>Customize your app experience.</Text>
      </TouchableOpacity>
    </ScrollView>
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
    fontSize: 26,
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
  section: {
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
  },
  lightSection: {
    backgroundColor: '#F4F6F8',
  },
  darkSection: {
    backgroundColor: '#2D2D2D',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  sectionContent: {
    fontSize: 14,
  },
  lightContentText: {
    color: '#636165',
  },
  darkContentText: {
    color: '#F9FAFB',
  },
});

export default ProfileScreen;
