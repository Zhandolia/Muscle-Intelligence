// ChooseVideo.js
import React, { useState } from 'react';
import { 
  View, 
  TouchableOpacity, 
  Image, 
  Text, 
  StyleSheet, 
  ActivityIndicator, 
  Alert 
} from 'react-native';
import * as DocumentPicker from 'expo-image-picker';
import { useNavigation } from '@react-navigation/native';

const UploadAlpha = () => {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadHistory, setUploadHistory] = useState([]);
  const navigation = useNavigation();

  const pickVideo = async () => {
    const result = await DocumentPicker.launchImageLibraryAsync({
      mediaTypes: DocumentPicker.MediaTypeOptions.Videos,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      const video = result.assets[0];
      setSelectedVideo(video);
      setUploadHistory((prev) => [...prev, video]);
    }
  };

  const handleSelect = (index) => {
    setTimeout(() => navigation.navigate('LoadingAlphaOne'), 300);
  };

  return (
    <View style={styles.container}>
      {!selectedVideo && uploadHistory.length === 0 && (
        <View style={styles.initialView}>
          <TouchableOpacity onPress={pickVideo}>
            <Text style={styles.uploadText}>Upload Video</Text>
          </TouchableOpacity>
          <Text style={styles.subText}>It will be converted into GIF</Text>
        </View>
      )}

      {selectedVideo && (
        <View style={styles.uploadSection}>
          <Image 
            source={require('../../assets/correct_try.gif')} 
            style={styles.videoPreview} 
          />
          <TouchableOpacity 
            onPress={() => navigation.navigate('LoadingAlphaOne')} 
            style={styles.analyzeButton}
            disabled={uploading}
          >
            <Text style={styles.analyzeText}>Analyze</Text>
          </TouchableOpacity>

          {uploading && (
            <View style={styles.uploadingIndicator}>
              <ActivityIndicator size="large" color="#F34533" />
            </View>
          )}
        </View>
      )}

      {uploadHistory.length > 0 && (
        <View style={styles.historyContainer}>
          <Text style={styles.historyTitle}>Upload History</Text>
          <View style={styles.grid}>
            {uploadHistory.map((video, index) => (
              <TouchableOpacity
                key={index}
                onPress={() => handleSelect(index)}
                style={styles.historyItem}
              >
                <Image 
                  source={{ uri: video.uri }} 
                  style={styles.videoThumbnail} 
                />
              </TouchableOpacity>
            ))}
          </View>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  initialView: {
    alignItems: 'center',
  },
  uploadText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#4F46E5',
    textDecorationLine: 'underline',
    marginBottom: 10,
  },
  subText: {
    fontSize: 16,
    color: '#6B7280',
  },
  uploadSection: {
    alignItems: 'center',
  },
  videoPreview: {
    width: 300,
    height: 300,
    marginVertical: 20,
  },
  analyzeButton: {
    backgroundColor: '#F34533',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 10,
    marginTop: 10,
  },
  analyzeText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
    fontSize: 16,
  },
  uploadingIndicator: {
    marginTop: 10,
  },
  historyContainer: {
    width: '100%',
    marginTop: 20,
  },
  historyTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 10,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-around',
  },
  historyItem: {
    width: '45%',
    aspectRatio: 1,
    backgroundColor: '#E5E7EB',
    marginBottom: 10,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  videoThumbnail: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
});

export default UploadAlpha;
