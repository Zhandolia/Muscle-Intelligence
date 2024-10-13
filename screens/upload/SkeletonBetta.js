import React from 'react';
import { View, Text, Image, StyleSheet, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';

const SkeletonBetta = () => {
  const navigation = useNavigation();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Confirm Skeleton</Text>
      <View style={styles.videoContainer}>
        <Image 
          source={require('../../assets/wrong_try_marked.gif')} 
          style={styles.video} 
        />
      </View>
      <Text style={styles.subtext}>
        Please confirm if the model matched skeleton
      </Text>
      <TouchableOpacity>
        <Text style={styles.regenerateText}>
          Didn’t match? Regenerate here!
        </Text>
      </TouchableOpacity>
      <TouchableOpacity 
        style={styles.button} 
        onPress={() => navigation.navigate('LoadingBettaTwo')}
      >
        <Text style={styles.buttonText}>Confirm →</Text>
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
    backgroundColor: '#E0E0E0', // Grey background as per design
    marginBottom: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  video: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  subtext: {
    fontSize: 16,
    color: '#1D1C1F',
    marginBottom: 10,
  },
  regenerateText: {
    fontSize: 16,
    color: '#F34533',
    marginBottom: 20,
    textDecorationLine: 'underline',
  },
  button: {
    backgroundColor: '#F34533',
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
    width: '60%',
  },
  buttonText: {
    color: '#F9FAFB',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default SkeletonBetta;
