import React, { useState } from 'react';
import { useNavigation } from '@react-navigation/native';
import { View, TouchableOpacity, Image, Text, StyleSheet } from 'react-native';

const ChooseVideo = () => {
  const [selected, setSelected] = useState(null);
  const navigation = useNavigation();

  const handleSelect = (index) => {
    setSelected(index);
    setTimeout(() => navigation.navigate('AnalyzingVideo'), 300); // navigate after 300 ms
  };

  const videos = Array(1).fill(require('../../assets/wrong_try.gif'));

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Choose Video</Text>
      <View style={styles.grid}>
        {videos.map((video, index) => (
          <TouchableOpacity
            key={index}
            onPress={() => handleSelect(index)}
            style={[
              styles.videoContainer,
              selected === index && styles.selectedContainer,
            ]}
          >
            <Image source={video} style={styles.video} />
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1D1C1F',
    marginBottom: 20,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  videoContainer: {
    margin: 10,
    borderWidth: 2,
    borderColor: '#F4F6F8',
  },
  selectedContainer: {
    borderColor: '#F34533',
  },
  video: {
    width: 100,
    height: 100,
  },
});

export default ChooseVideo;
