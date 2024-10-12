import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useTheme } from './ThemeContext';  // Import the useTheme hook

const BodyAreasTargetedScreen = () => {
  const { theme } = useTheme();  // Access the current theme
  const isDarkTheme = theme === 'dark';  // Determine if the theme is dark

  const bodyAreas = ['Chest', 'Back', 'Arms', 'Legs', 'Core', 'Shoulders'];

  return (
    <ScrollView style={[styles.container, isDarkTheme ? styles.darkContainer : styles.lightContainer]}>
      <Text style={[styles.title, isDarkTheme ? styles.darkText : styles.lightText]}>Body Areas Targeted</Text>
      <View style={styles.grid}>
        {bodyAreas.map((area, index) => (
          <TouchableOpacity key={index} style={isDarkTheme ? styles.darkAreaContainer : styles.lightAreaContainer}>
            <Text style={[styles.areaText, isDarkTheme ? styles.darkText : styles.lightText]}>{area}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  lightContainer: {
    backgroundColor: '#F9FAFB',  // Light background for light theme
  },
  darkContainer: {
    backgroundColor: '#1D1C1F',  // Dark background for dark theme (darker than before)
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  lightText: {
    color: '#1D1C1F',  // Dark text for light theme
  },
  darkText: {
    color: '#E0E0E0',  // Light grey text for dark theme to make it readable
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  lightAreaContainer: {
    backgroundColor: '#F4F6F8',  // Light background for areas in light theme
    flexBasis: '48%',  // Each area container takes up 48% of the row width
    marginBottom: 10,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 15,
  },
  darkAreaContainer: {
    backgroundColor: '#2D2D2D',  // Darker background for areas in dark theme
    flexBasis: '48%',  // Each area container takes up 48% of the row width
    marginBottom: 10,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 15,
  },
  areaText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
});

export default BodyAreasTargetedScreen;
