import React from 'react';
import { View, Text, FlatList, StyleSheet } from 'react-native';

const historyData = [
  { id: '1', date: '2024-09-18', result: 'Improved posture' },
  { id: '2', date: '2024-09-17', result: 'Needs improvement' },
  { id: '3', date: '2024-09-16', result: 'Good form' },
];

const WorkoutHistoryScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Workout History</Text>
      <FlatList
        data={historyData}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <View style={styles.item}>
            <Text style={styles.dateText}>{item.date}</Text>
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
  item: {
    backgroundColor: '#F4F6F8',
    padding: 20,
    borderRadius: 10,
    marginBottom: 15,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  dateText: {
    fontSize: 16,
    color: '#636165',
  },
  resultText: {
    fontSize: 18,
    color: '#F34533',
    fontWeight: 'bold',
  },
});

export default WorkoutHistoryScreen;
