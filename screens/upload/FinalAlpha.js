import React from 'react';
import { View, Text, Image, StyleSheet, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';

const FinalAlpha = () => {
  const navigation = useNavigation();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Analyzed Video</Text>
      <View style={styles.videoContainer}>
        <Image 
          source={require('../../assets/correct_try_marked_colored.gif')} 
          style={styles.video} 
        />
      </View>
      <View style={styles.textContainer}>
        <Text style={[styles.text, styles.green]}>
          <Text style={styles.boldText}>Perfect Area: </Text>
          Triceps, Core, Chest, Shoulders
        </Text>
        <Text style={[styles.text, styles.orange]}>
          <Text style={styles.boldText}>Needs Work: </Text>
          Biceps
        </Text>
      </View>
      <TouchableOpacity 
        style={styles.button} 
        onPress={() => navigation.navigate('UploadBetta')}
        >
        <Text style={styles.buttonText}>Return Back</Text>
        </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1D1C1F',
    marginBottom: 20,
  },
  videoContainer: {
    width: 300,
    height: 300,
    backgroundColor: '#E0E0E0', // Grey placeholder for video
    marginBottom: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  video: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  textContainer: {
    marginBottom: 20,
  },
  text: {
    fontSize: 16,
    marginBottom: 5,
  },
  boldText: {
    fontWeight: 'bold',
  },
  green: {
    color: '#27AE60',
  },
  orange: {
    color: '#E67E22',
  },
  red: {
    color: '#E74C3C',
  },
  button: {
    backgroundColor: '#F34533',
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
    width: '70%',
  },
  buttonText: {
    color: '#F9FAFB',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default FinalAlpha;
