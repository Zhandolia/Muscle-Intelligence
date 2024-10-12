import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';

const BodyAreasTargetedScreen = () => {
  const bodyAreas = ['Chest', 'Back', 'Arms', 'Legs', 'Core', 'Shoulders'];

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Body Areas Targeted</Text>
      <View style={styles.grid}>
        {bodyAreas.map((area, index) => (
          <TouchableOpacity key={index} style={styles.areaContainer}>
            <Text style={styles.areaText}>{area}</Text>
          </TouchableOpacity>
        ))}
      </View>
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
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  areaContainer: {
    width: '48%',
    backgroundColor: '#F4F6F8',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  areaText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#636165',
  },
});

export default BodyAreasTargetedScreen;
