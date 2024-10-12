import React from 'react';
import { View, Text, FlatList, StyleSheet } from 'react-native';
import { useTheme } from './ThemeContext';  // Use the ThemeContext

const WorkoutHistoryScreen = () => {
  const { theme } = useTheme();  // Get the current theme from context
  const isDarkTheme = theme === 'dark';  // Determine if the theme is dark

  const historyData = [
    { id: '1', date: '2024-09-18', result: 'Improved posture' },
    { id: '2', date: '2024-09-17', result: 'Needs improvement' },
    { id: '3', date: '2024-09-16', result: 'Good form' },
  ];

  return (
    <View style={[styles.container, isDarkTheme ? styles.darkContainer : styles.lightContainer]}>
      <Text style={[styles.title, isDarkTheme ? styles.darkText : styles.lightText]}>Workout History</Text>
      <FlatList
        data={historyData}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <View style={[styles.item, isDarkTheme ? styles.darkSetting : styles.lightSetting]}>
            <Text style={[styles.dateText, isDarkTheme ? styles.darkText : styles.lightText]}>{item.date}</Text>
            <Text style={styles.resultText}>{item.result}</Text>
          </View>
        )}
      />
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
  item: {
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
  dateText: {
    fontSize: 16,
  },
  resultText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#F34533',
  },
});

export default WorkoutHistoryScreen;
