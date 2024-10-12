import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native'; // Import useNavigation

const ProfileScreen = () => {
  const navigation = useNavigation(); // Initialize navigation

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Profile</Text>

      <TouchableOpacity style={styles.section} onPress={() => navigation.navigate('BodyAreasTargeted')}>
        <Text style={styles.sectionTitle}>Body Areas Targeted</Text>
        <Text style={styles.sectionContent}>See which muscles you've been focusing on.</Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.section} onPress={() => navigation.navigate('DevicesConnected')}>
        <Text style={styles.sectionTitle}>Devices Connected</Text>
        <Text style={styles.sectionContent}>View data from your connected devices.</Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.section} onPress={() => navigation.navigate('WorkoutHistory')}>
        <Text style={styles.sectionTitle}>Workout History</Text>
        <Text style={styles.sectionContent}>Your past workouts will appear here.</Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.section} onPress={() => navigation.navigate('Progress')}>
        <Text style={styles.sectionTitle}>Progress</Text>
        <Text style={styles.sectionContent}>Track your progress over time.</Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.section} onPress={() => navigation.navigate('Settings')}>
        <Text style={styles.sectionTitle}>Settings</Text>
        <Text style={styles.sectionContent}>Customize your app experience.</Text>
      </TouchableOpacity>

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
    fontSize: 26,
    fontWeight: 'bold',
    color: '#1D1C1F',
    marginBottom: 20,
    textAlign: 'center',
  },
  section: {
    backgroundColor: '#F4F6F8',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1D1C1F',
    marginBottom: 5,
  },
  sectionContent: {
    fontSize: 14,
    color: '#636165',
  },
});

export default ProfileScreen;
