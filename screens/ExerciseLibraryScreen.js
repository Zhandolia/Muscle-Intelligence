// src/screens/ExerciseLibraryScreen.js
import React from 'react';
import { View, Text, Button, FlatList, StyleSheet } from 'react-native';

const exercises = [
  { id: '1', name: 'Squats' },
  { id: '2', name: 'Push-ups' },
  { id: '3', name: 'Deadlifts' },
  { id: '4', name: 'Planks' },
];

const ExerciseLibraryScreen = ({ navigation }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Exercise Library</Text>
      <FlatList
        data={exercises}
        keyExtractor={item => item.id}
        renderItem={({ item }) => (
          <View style={styles.item}>
            <Text style={styles.itemText}>{item.name}</Text>
            <Button
              title="Track Posture"
              onPress={() => navigation.navigate('PostureAnalysis')}
            />
          </View>
        )}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  item: {
    padding: 20,
    marginVertical: 8,
    borderWidth: 1,
    borderRadius: 5,
    borderColor: '#ddd',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  itemText: {
    fontSize: 18,
  },
});

export default ExerciseLibraryScreen;
