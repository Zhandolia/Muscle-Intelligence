import React from 'react';
import { View, Text, StyleSheet, Button, TouchableOpacity } from 'react-native';

const HomeScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Welcome to the Feed</Text>
      <Text style={styles.subtitle}>Share your progress with friends here.</Text>

      <TouchableOpacity style={styles.button}>
        <Text style={styles.buttonText}>Share Progress</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F9FAFB',
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#1D1C1F',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: '#636165',
    marginBottom: 30,
  },
  button: {
    backgroundColor: '#F34533',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  buttonText: {
    color: '#F9FAFB',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default HomeScreen;
