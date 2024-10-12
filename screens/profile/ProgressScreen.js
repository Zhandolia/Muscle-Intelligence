import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';

const ProgressScreen = () => {
  const deviceData = [
    { label: 'Calories Burnt Today', value: '450' },
    { label: 'Steps Today', value: '10,000' },
    { label: 'Heart BPM', value: '72' },
    // Add more device data as necessary
  ];

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Progress</Text>
      {deviceData.map((item, index) => (
        <View key={index} style={styles.dataContainer}>
          <Text style={styles.dataLabel}>{item.label}</Text>
          <Text style={styles.dataValue}>{item.value}</Text>
        </View>
      ))}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1D1C1F',
    marginBottom: 20,
    textAlign: 'center',
  },
  dataContainer: {
    backgroundColor: '#F4F6F8',
    padding: 20,
    borderRadius: 10,
    marginBottom: 15,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  dataLabel: {
    fontSize: 16,
    color: '#636165',
  },
  dataValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#F34533',
  },
});

export default ProgressScreen;
