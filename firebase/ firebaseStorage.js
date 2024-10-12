import { storage } from './firebaseConfig';
import { ref, uploadBytesResumable, getDownloadURL } from "firebase/storage";

// Function to upload video to Firebase Storage
export const uploadVideoToFirebase = async (file, setProgress) => {
  const storageRef = ref(storage, `videos/${file.name}`);
  const uploadTask = uploadBytesResumable(storageRef, file);

  return new Promise((resolve, reject) => {
    uploadTask.on(
      'state_changed',
      (snapshot) => {
        // Update progress
        const progress = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
        setProgress(progress); // This will update the progress bar or UI
        console.log(`Upload is ${progress}% done`);
      },
      (error) => {
        // Handle errors
        console.error('Upload failed:', error);
        reject(error);
      },
      () => {
        // Success - get the download URL
        getDownloadURL(uploadTask.snapshot.ref).then((downloadURL) => {
          console.log('Uploaded video successfully, URL:', downloadURL);
          resolve(downloadURL);
        });
      }
    );
  });
};