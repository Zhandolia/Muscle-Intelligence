import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { useTheme } from './ThemeContext';  // Use the ThemeContext

const ProgressScreen = () => {
  const { theme } = useTheme();  // Get the current theme from context
  const isDarkTheme = theme === 'dark';  // Determine if the theme is dark

  const deviceData = [
    { label: 'Calories Burnt Today', value: '450' },
    { label: 'Steps Today', value: '10,000' },
    { label: 'Heart BPM', value: '72' },
  ];

  return (
    <ScrollView style={[styles.container, isDarkTheme ? styles.darkContainer : styles.lightContainer]}>
      <Text style={[styles.title, isDarkTheme ? styles.darkText : styles.lightText]}>Progress</Text>
      {deviceData.map((item, index) => (
        <View key={index} style={[styles.dataContainer, isDarkTheme ? styles.darkSetting : styles.lightSetting]}>
          <Text style={[styles.dataLabel, isDarkTheme ? styles.darkText : styles.lightText]}>{item.label}</Text>
          <Text style={styles.dataValue}>{item.value}</Text>
        </View>
      ))}
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
  dataContainer: {
    padding: 20,
    borderRadius: 10,
    marginBottom: 15,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  lightSetting: {
    backgroundColor: '#F4F6F8',
  },
  darkSetting: {
    backgroundColor: '#2D2D2D',
  },
  dataLabel: {
    fontSize: 16,
  },
  dataValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#F34533',
  },
});

export default ProgressScreen;
