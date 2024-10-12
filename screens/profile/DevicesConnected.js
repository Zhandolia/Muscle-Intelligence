import React from 'react';
import { View, Text, ScrollView, StyleSheet } from 'react-native';

const DevicesConnectedScreen = () => {
  const deviceData = [
    { label: 'Garmin Watch', value: 'Connected' },
    { label: 'Apple Watch', value: 'Connected' },
    { label: 'Strava', value: 'Connected' },
  ];

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Connected Devices</Text>
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

export default DevicesConnectedScreen;
