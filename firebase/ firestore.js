// firestore.js
import { firestore } from './firebaseConfig';
import { collection, addDoc, getDocs } from "firebase/firestore";

// Function to store video metadata in Firestore
export const storeVideoMetadata = async (downloadURL, title, userId) => {
  try {
    await addDoc(collection(firestore, 'videos'), {
      title: title,
      userId: userId,
      videoUrl: downloadURL,
      uploadDate: new Date(),
    });
    console.log('Video metadata stored successfully');
  } catch (error) {
    console.error('Error storing video metadata:', error);
  }
};

// Function to get all videos metadata from Firestore
export const getAllVideos = async () => {
  const querySnapshot = await getDocs(collection(firestore, 'videos'));
  return querySnapshot.docs.map(doc => doc.data());
};
