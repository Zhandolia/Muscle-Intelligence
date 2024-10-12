// UploadScreen.js
import React, { useState } from 'react';
import { View, Button, Text, ActivityIndicator, Alert } from 'react-native';
import * as DocumentPicker from 'expo-image-picker';


const UploadScreen = ({ userId }) => {
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);

  const pickVideo = async () => {
    let result = await DocumentPicker.launchImageLibraryAsync({
      mediaTypes: DocumentPicker.MediaTypeOptions.Videos,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setFile(result.assets[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      Alert.alert('No video selected', 'Please select a video to upload.');
      return;
    }

    setUploading(true);
    try {
      const videoUrl = await uploadVideoToFirebase(file, setProgress); // Upload video and track progress
      await storeVideoMetadata(videoUrl, file.name, userId); // Store video metadata
      Alert.alert('Success', 'Video uploaded successfully');
    } catch (error) {
      console.error('Upload error:', error);
      Alert.alert('Error', 'Video upload failed.');
    } finally {
      setUploading(false);
      setProgress(0); // Reset progress after upload is complete
    }
  };

  return (
    <View style={{ flex: 1, justifyContent: 'center', padding: 20 }}>
      <Button title="Pick Video" onPress={pickVideo} />
      {file && <Text>Selected video: {file.name}</Text>}

      <Button title="Upload Video" onPress={handleUpload} disabled={uploading} />

      {uploading && (
        <View>
          <Text>Uploading... {progress}%</Text>
          <ActivityIndicator size="large" color="#F34533" />
        </View>
      )}
    </View>
  );
};

export default UploadScreen;
