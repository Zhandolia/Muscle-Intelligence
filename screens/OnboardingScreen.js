// src/screens/OnboardingScreen.js
import React from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';

const OnboardingScreen = ({ navigation }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Welcome to Posture.ai</Text>
      <Text style={styles.subtitle}>Track your posture and improve your workouts!</Text>
      <Button 
        title="Get Started" 
        onPress={() => navigation.replace('Home')} 
        style={styles.button} 
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#333',
  },
  subtitle: {
    fontSize: 18,
    textAlign: 'center',
    marginBottom: 40,
    color: '#666',
  },
  button: {
    backgroundColor: '#4CAF50',
    color: 'white',
  },
});

export default OnboardingScreen;
